open Inference_compare_schema
module E = Element_codec

type env = {
  codecs : Element_codec.Registry.t;
  operations : Conversion_registry.t;
  canonical : Id.Canonical_value.t -> Tensor_spec.t option;
  boundary : Tensor_ref.t -> Tensor_spec.t option;
}

let fail e = Err.fail ~pp_error:Codec_error.pp e
let ( let* ) = Err.bind

module Node_map = Map.Make (struct
  type t = Id.Conversion_node.t

  let compare = Id.Conversion_node.compare
end)

module Checked = struct
  type t = {
    recipe : Recipe.t;
    order : Recipe.node list;
    specs : Tensor_spec.t Node_map.t;
    result_spec : Tensor_spec.t;
    lossless : bool;
    canonical_reads : Id.Canonical_value.t list;
    boundary_reads : Tensor_ref.t list;
  }

  let recipe t = t.recipe
  let order t = t.order
  let node_spec t id = Node_map.find_opt id t.specs
  let result_spec t = t.result_spec
  let lossless t = t.lossless
  let canonical_reads t = t.canonical_reads
  let boundary_reads t = t.boundary_reads
end

let shape_fault f = fail (Codec_error.Shape f)

let codec_of env (spec : Tensor_spec.t) =
  match Element_codec.Registry.find env.codecs spec.dtype.codec with
  | Some c -> Ok c
  | None -> fail (Codec_error.Unknown_codec spec.dtype.codec)

let with_shape (spec : Tensor_spec.t) shape =
  { spec with Tensor_spec.logical_shape = shape }

let rec nth_opt l i =
  match l with
  | [] -> None
  | x :: r -> if i = 0 then Some x else nth_opt r (i - 1)

(* Output spec and "this step preserves every value" for one step. *)
let infer_step env (step : Conversion_step.t) (args : Tensor_spec.t list) =
  let unary f =
    match args with
    | [ a ] -> f a
    | _ -> fail (Codec_error.Arity { expected = 1; actual = List.length args })
  in
  match step with
  | Reshape shape ->
      unary (fun a ->
          let* src = Shape_math.element_count a.logical_shape in
          let* dst = Shape_math.element_count shape in
          if Int64.equal src dst then Ok (with_shape a shape, true)
          else
            shape_fault (Element_count_changed { source = src; target = dst }))
  | Permute axes ->
      unary (fun a ->
          let rank = List.length a.logical_shape in
          if List.length axes <> rank then
            shape_fault
              (Permutation_arity { expected = rank; actual = List.length axes })
          else
            let rec check seen = function
              | [] -> Ok ()
              | x :: rest ->
                  if x < 0 || x >= rank then
                    shape_fault (Axis_out_of_range { axis = x; rank })
                  else if List.mem x seen then shape_fault (Duplicate_axis x)
                  else check (x :: seen) rest
            in
            let* () = check [] axes in
            let dims = Array.of_list a.logical_shape in
            Ok (with_shape a (List.map (fun x -> dims.(x)) axes), true))
  | Slice { axis; start; length } ->
      unary (fun a ->
          let rank = List.length a.logical_shape in
          match nth_opt a.logical_shape axis with
          | None -> shape_fault (Axis_out_of_range { axis; rank })
          | Some extent ->
              (* [start + length <= extent], tested as [start <= extent -
                 length] so the sum is never formed. *)
              if
                Int64.compare length extent > 0
                || Int64.compare start (Int64.sub extent length) > 0
              then
                shape_fault (Slice_out_of_range { axis; start; length; extent })
              else
                Ok
                  ( with_shape a
                      (List.mapi
                         (fun i d -> if i = axis then length else d)
                         a.logical_shape),
                    true ))
  | Concat axis -> (
      match args with
      | [] -> fail (Codec_error.Arity { expected = 1; actual = 0 })
      | first :: rest ->
          let rank = List.length first.logical_shape in
          if axis < 0 || axis >= rank then
            shape_fault (Axis_out_of_range { axis; rank })
          else
            (* Every other operand agrees with the first on dtype, rank and
               every extent off the concatenation axis. *)
            let compatible (s : Tensor_spec.t) =
              if not (Dtype.equal s.dtype first.dtype) then
                fail
                  (Codec_error.Codec_mismatch
                     { expected = first.dtype.codec; actual = s.dtype.codec })
              else if List.length s.logical_shape <> rank then
                shape_fault
                  (Rank_mismatch
                     { left = rank; right = List.length s.logical_shape })
              else
                let rec dims i a b =
                  match (a, b) with
                  | x :: ar, y :: br ->
                      if i <> axis && not (Int64.equal x y) then
                        shape_fault
                          (Extent_mismatch { axis = i; left = x; right = y })
                      else dims (i + 1) ar br
                  | _ -> Ok ()
                in
                dims 0 first.logical_shape s.logical_shape
            in
            let rec all_compatible = function
              | [] -> Ok ()
              | s :: more ->
                  let* () = compatible s in
                  all_compatible more
            in
            let* () = all_compatible rest in
            let extent (s : Tensor_spec.t) =
              Option.get (nth_opt s.logical_shape axis)
            in
            let rec total acc = function
              | [] -> Ok acc
              | s :: more ->
                  let e = extent s in
                  if Int64.compare e 0L < 0 then shape_fault (Negative_extent e)
                  else if Int64.compare e (Int64.sub Int64.max_int acc) > 0 then
                    fail Codec_error.Count_overflow
                  else total (Int64.add acc e) more
            in
            let* total = total 0L (first :: rest) in
            Ok
              ( with_shape first
                  (List.mapi
                     (fun i d -> if i = axis then total else d)
                     first.logical_shape),
                true ))
  | Cast { target; rounding; overflow } ->
      unary (fun a ->
          let* source = codec_of env a in
          let* tgt = codec_of env { a with dtype = target } in
          match Cast_law.classify source tgt with
          | Unsupported ->
              fail
                (Codec_error.Unsupported_cast
                   { source = source.id; target = tgt.id })
          | verdict ->
              if not (Cast_law.rounding_admitted ~source ~target:tgt rounding)
              then fail Codec_error.Unsupported_rounding
              else if
                not (Cast_law.overflow_admitted ~source ~target:tgt overflow)
              then fail Codec_error.Unsupported_overflow
              else
                Ok
                  ( { a with Tensor_spec.dtype = target },
                    verdict = Cast_law.Lossless ))
  | Codec_operation { name; implementation_digest } -> (
      match Conversion_registry.find env.operations name with
      | None -> fail (Codec_error.Unknown_operation name)
      | Some op -> (
          if
            not
              (Digest.Implementation.equal op.implementation_digest
                 implementation_digest)
          then fail (Codec_error.Unknown_operation name)
          else if op.arity <> List.length args then
            fail
              (Codec_error.Arity
                 { expected = op.arity; actual = List.length args })
          else
            match op.infer args with
            | Ok spec -> Ok (spec, op.lossless)
            | Error e -> fail e))

let check env (r : Recipe.t) =
  let nodes =
    List.fold_left
      (fun m (n : Recipe.node) -> Node_map.add n.id n m)
      Node_map.empty r.nodes
  in
  let* () =
    let rec dup seen = function
      | [] -> Ok ()
      | (n : Recipe.node) :: rest ->
          if List.exists (Id.Conversion_node.equal n.id) seen then
            fail (Codec_error.Duplicate_node n.id)
          else dup (n.id :: seen) rest
    in
    dup [] r.nodes
  in
  (* Depth-first, deterministic: arguments are visited in declared order, and
     a node is entered only through the result or another reachable node, so
     unreachable nodes are never inferred (and never executed). *)
  let specs = ref Node_map.empty in
  let lossless = ref true in
  let order = ref [] in
  let canon = ref [] and bound = ref [] in
  let visiting = ref [] in
  let rec arg_spec = function
    | Recipe.Canonical c -> (
        if not (List.exists (Id.Canonical_value.equal c) !canon) then
          canon := c :: !canon;
        match env.canonical c with
        | Some s -> Ok s
        | None -> fail (Codec_error.Unknown_canonical c))
    | Boundary t -> (
        if not (List.exists (Tensor_ref.equal t) !bound) then
          bound := t :: !bound;
        match env.boundary t with
        | Some s -> Ok s
        | None -> fail (Codec_error.Unknown_boundary t))
    | Conversion_result id -> node id
  and node id =
    match Node_map.find_opt id !specs with
    | Some s -> Ok s
    | None -> (
        if List.exists (Id.Conversion_node.equal id) !visiting then
          fail (Codec_error.Cyclic_recipe id)
        else
          match Node_map.find_opt id nodes with
          | None -> fail (Codec_error.Unknown_node id)
          | Some n ->
              visiting := id :: !visiting;
              let rec args acc = function
                | [] -> Ok (List.rev acc)
                | a :: rest ->
                    let* s = arg_spec a in
                    args (s :: acc) rest
              in
              let* arg_specs = args [] n.arguments in
              let* spec, preserved = infer_step env n.operation arg_specs in
              visiting := List.tl !visiting;
              if not preserved then lossless := false;
              specs := Node_map.add id spec !specs;
              order := n :: !order;
              Ok spec)
  in
  let* result_spec = arg_spec r.result in
  let sources_equal (a : Tensor_spec.t) (b : Tensor_spec.t) =
    Dtype.equal a.dtype b.dtype
    && List.equal Int64.equal a.logical_shape b.logical_shape
    && Bool.equal a.value_semantics b.value_semantics
  in
  let passthrough =
    match r.result with Conversion_result _ -> false | _ -> true
  in
  let declared_ok =
    if passthrough then Tensor_spec.equal result_spec r.output_spec
    else sources_equal result_spec r.output_spec
  in
  if not declared_ok then fail Codec_error.Spec_mismatch
  else if r.lossless && not !lossless then fail Codec_error.Lossy_recipe
  else
    let sorted_nodes = List.rev !order in
    Ok
      {
        Checked.recipe = r;
        order = sorted_nodes;
        specs = !specs;
        result_spec = r.output_spec;
        lossless = !lossless;
        canonical_reads = List.sort Id.Canonical_value.compare !canon;
        boundary_reads = List.sort Tensor_ref.compare !bound;
      }

module Lineage = struct
  type source =
    | Canonical_input of Id.Canonical_value.t * Digest.Logical_value.t
    | Boundary_input of Tensor_ref.t * Digest.Logical_value.t
    | Node_result of Id.Conversion_node.t

  type entry = {
    node : Id.Conversion_node.t;
    operation : Conversion_step.t;
    sources : source list;
    result : Digest.Logical_value.t;
  }

  type t = entry list
end

type inputs = {
  canonical : Id.Canonical_value.t -> Logical_value.t option;
  boundary : Tensor_ref.t -> Logical_value.t option;
}

let run_step (ctx : Logical_value.Context.t) env (step : Conversion_step.t)
    (out_spec : Tensor_spec.t) (args : Logical_value.t list) =
  let codec_of_value v = Logical_value.codec ctx v in
  match (step, args) with
  | Reshape _, [ a ] ->
      (* Row-major bytes are unchanged; the immutable string is shared. *)
      Logical_value.make ctx out_spec (Logical_value.contents a)
  | Permute axes, [ a ] ->
      let c = codec_of_value a in
      let shape = Shape_math.to_ints (Logical_value.spec a).logical_shape in
      Logical_value.make ctx out_spec
        (Recipe_kernels.permute ~width:c.byte_width ~axes ~shape
           (Logical_value.contents a))
  | Slice { axis; start; length }, [ a ] ->
      let c = codec_of_value a in
      let shape = Shape_math.to_ints (Logical_value.spec a).logical_shape in
      Logical_value.make ctx out_spec
        (Recipe_kernels.slice ~width:c.byte_width ~axis
           ~start:(Int64.to_int start) ~length:(Int64.to_int length) ~shape
           (Logical_value.contents a))
  | Concat axis, first :: _ ->
      let c = codec_of_value first in
      let operands =
        List.map
          (fun v ->
            ( Shape_math.to_ints (Logical_value.spec v).logical_shape,
              Logical_value.contents v ))
          args
      in
      Logical_value.make ctx out_spec
        (Recipe_kernels.concat ~width:c.byte_width ~axis operands)
  | Cast { rounding; overflow; _ }, [ a ] -> (
      let source = codec_of_value a in
      match Element_codec.Registry.find ctx.codecs out_spec.dtype.codec with
      | None -> fail (Codec_error.Unknown_codec out_spec.dtype.codec)
      | Some target -> (
          match
            Recipe_kernels.cast ~source ~target ~rounding ~overflow
              ~count:(Logical_value.element_count a)
              (Logical_value.contents a)
          with
          | Ok bytes -> Logical_value.make ctx out_spec bytes
          | Error e -> fail e))
  | Codec_operation { name; _ }, _ -> (
      match Conversion_registry.find env.operations name with
      | None -> fail (Codec_error.Unknown_operation name)
      | Some op -> op.apply ctx args)
  | _ -> fail (Codec_error.Arity { expected = 1; actual = List.length args })

let evaluate ctx env (checked : Checked.t) (inputs : inputs) =
  let recipe = Checked.recipe checked in
  let live = ref Node_map.empty in
  let owned = ref [] in
  let release_all () = List.iter Logical_value.drop !owned in
  let uses =
    (* Remaining reads of each node, over the reachable nodes and the result:
       an intermediate is dropped as soon as its count reaches zero. *)
    let tbl = ref Node_map.empty in
    let bump id =
      tbl :=
        Node_map.update id
          (function None -> Some 1 | Some n -> Some (n + 1))
          !tbl
    in
    List.iter
      (fun (n : Recipe.node) ->
        List.iter
          (function Recipe.Conversion_result id -> bump id | _ -> ())
          n.arguments)
      (Checked.order checked);
    (match recipe.result with Conversion_result id -> bump id | _ -> ());
    ref !tbl
  in
  let lineage = ref [] in
  let fetch_arg = function
    | Recipe.Canonical c -> (
        match inputs.canonical c with
        | Some v -> Ok (v, Lineage.Canonical_input (c, Logical_value.digest v))
        | None -> fail (Codec_error.Unknown_canonical c))
    | Boundary t -> (
        match inputs.boundary t with
        | Some v -> Ok (v, Lineage.Boundary_input (t, Logical_value.digest v))
        | None -> fail (Codec_error.Unknown_boundary t))
    | Conversion_result id -> (
        match Node_map.find_opt id !live with
        | Some v -> Ok (v, Lineage.Node_result id)
        | None -> fail (Codec_error.Unknown_node id))
  in
  let consume = function
    | Recipe.Conversion_result id -> (
        let n = Node_map.find id !uses - 1 in
        uses := Node_map.add id n !uses;
        if n = 0 then
          match Node_map.find_opt id !live with
          | Some v ->
              live := Node_map.remove id !live;
              owned := List.filter (fun x -> x != v) !owned;
              Logical_value.drop v
          | None -> ())
    | _ -> ()
  in
  let step (n : Recipe.node) =
    let rec fetch acc = function
      | [] -> Ok (List.rev acc)
      | a :: rest ->
          let* v, src = fetch_arg a in
          fetch ((v, src) :: acc) rest
    in
    let* args = fetch [] n.arguments in
    let out_spec = Option.get (Checked.node_spec checked n.id) in
    let* value = run_step ctx env n.operation out_spec (List.map fst args) in
    owned := value :: !owned;
    live := Node_map.add n.id value !live;
    lineage :=
      {
        Lineage.node = n.id;
        operation = n.operation;
        sources = List.map snd args;
        result = Logical_value.digest value;
      }
      :: !lineage;
    List.iter consume n.arguments;
    Ok ()
  in
  let run () =
    let rec go = function
      | [] -> Ok ()
      | n :: rest ->
          let* () = step n in
          go rest
    in
    let* () = go (Checked.order checked) in
    let* v, _ = fetch_arg recipe.result in
    (* The caller gets its own value: either the result node's (ownership
       moves) or a copy of a passthrough argument. *)
    match recipe.result with
    | Conversion_result id ->
        let v = Node_map.find id !live in
        live := Node_map.remove id !live;
        owned := List.filter (fun x -> x != v) !owned;
        Ok v
    | _ ->
        Logical_value.make ctx
          (Checked.result_spec checked)
          (Logical_value.contents v)
  in
  match run () with
  | Ok v ->
      release_all ();
      Ok (v, List.rev !lineage)
  | Error e ->
      release_all ();
      Error e
