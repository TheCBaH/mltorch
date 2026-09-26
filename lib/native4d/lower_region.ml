(* See lower_region.mli. *)

open Lower_engine_acc
open Graph_ir

type step = { op : Tensor_id.t -> Op.t; shape : Shape4.t }
type output = { id : Tensor_id.t; steps : step list }

type interleave = {
  inputs : Tensor_id.t list;
  reshape : bool;  (** whether each input needs a reshape onto [unit_shape] *)
  unit_shape : Shape4.t;
  axis : Axis4.t;
  concat_shape : Shape4.t;
}

type t = {
  trigger : Node_id.t;
  members : Node_id.t list;
  internal : Tensor_id.t list;
  input : Tensor_id.t;
  interleave : interleave option;
  outputs : output list;
}

let extent shape axis = Vec6.get shape axis
let is_one e = Dim.equal e Dim.one

let nonunit shape =
  List.filter (fun a -> not (is_one (extent shape a))) Axis.all

let dialect_valid shape =
  is_one (extent shape Axis.T) && is_one (extent shape Axis.D)

let is_f32 (sg : Tensor_sig.t) =
  match sg.Tensor_sig.fmt with Payload.Fmt Payload.F32 -> true | _ -> false

let rec last_n n = function
  | l when List.length l <= n -> l
  | _ :: rest -> last_n n rest
  | [] -> []

let index_of x l =
  let rec go i = function
    | [] -> None
    | y :: rest -> if y = x then Some i else go (i + 1) rest
  in
  go 0 l

(* Everything that varies per output is the SELECTED index [k]; the rest of the
   chain is fixed by the three shapes and the permutation.

   The region is [x --Reshape--> tgt --Permute--> po (--Select/Unbind sel--> y)],
   with [tgt] (and [po] when a select follows) outside the four-axis domain. What
   makes it expressible anyway is that Native tensors are flat row-major, so the
   selected index of an axis is a strided slice of the reshape's SOURCE whenever
   the source's [xi] axis splits that axis off: [pre] (the product of the target
   axes before it) must equal the product of the source axes before [xi], and
   [xi] must hold [k_extent] times what follows the selected axis in the target.
   The remainder is then a reshape to the target's surviving non-unit axes, in
   order, relabelled onto N/H/W/C, and a Permute4 onto the result's axes. Every
   step is pure data movement, so each output keeps its source id and claim.

   [None] means "outside this recognizer", never an error: the caller falls back
   to the ordinary path, which reports the real blocker. *)
let chain ~x_shape ~tgt ~perm ~po ~y_shape ~sel =
  let x4 = Shape4.of_vec6 x_shape in
  match x4 with
  | Error _ -> None
  | Ok x4 -> (
      let exception Abort in
      let get = function Some v -> v | None -> raise Abort in
      (* A product that does not fit declines the recognizer. *)
      let product es = get (Extent_product.bounded es) in
      try
        (* 1. the slice of [x], when a select follows. *)
        let selected_in =
          Option.map (fun (sa, _, _) -> Permute.Permute.lookup perm sa) sel
        in
        let slice, x_after =
          match (sel, selected_in) with
          | Some (_, k_extent, (k : Dim.index Dim.t)), Some a ->
              let before_a, after_a =
                let rec split acc = function
                  | [] -> (List.rev acc, [])
                  | ax :: rest ->
                      if ax = a then (List.rev acc, rest)
                      else split (ax :: acc) rest
                in
                split [] Axis.all
              in
              let pre = product (List.map (extent tgt) before_a) in
              let post = product (List.map (extent tgt) after_a) in
              if is_one k_extent || not (Dim.equal (extent tgt a) k_extent) then
                raise Abort;
              let dims = Axis4.all in
              let candidate i =
                let ex a4 = Shape4.get x4 a4 in
                let before, rest =
                  let rec split acc = function
                    | [] -> (List.rev acc, [])
                    | d :: r ->
                        if d = i then (List.rev acc, r) else split (d :: acc) r
                  in
                  split [] dims
                in
                let e = ex i in
                Dim.equal (product (List.map ex before)) pre
                && (not (is_one e))
                && Dim.divides ~by:k_extent e
                && Dim.equal
                     (product [ e; product (List.map ex rest) ])
                     (product [ k_extent; post ])
              in
              let xi = get (List.find_opt candidate dims) in
              let s = get (Dim.div_exact (Shape4.get x4 xi) ~by:k_extent) in
              let shape = Shape4.set x4 xi s in
              ( Some
                  {
                    op =
                      (fun x ->
                        Op.Slice4
                          {
                            Ops4.Slice4.params =
                              {
                                axis = xi;
                                start = (k :> int) * (s :> int);
                                stop = ((k :> int) + 1) * (s :> int);
                                step = Op_config.Pos.of_int 1;
                              };
                            x;
                          });
                    shape;
                  },
                shape )
          | _ -> (None, x4)
        in
        (* 2. the reshape onto the target's surviving non-unit axes. *)
        let tgt_axes =
          List.filter (fun ax -> Some ax <> selected_in) (nonunit tgt)
        in
        let m = List.length tgt_axes in
        if m > 4 then raise Abort;
        let labels = last_n m Axis4.all in
        let rho = List.combine tgt_axes labels in
        let reshaped =
          List.fold_left
            (fun shape (ax, label) -> Shape4.set shape label (extent tgt ax))
            (Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:1)
            rho
        in
        let reshape =
          if Shape4.equal reshaped x_after then None
          else
            Some
              {
                op =
                  (fun x ->
                    Op.Reshape4
                      { Ops4.Reshape4.params = { shape = reshaped }; x });
                shape = reshaped;
              }
        in
        (* 3. the permutation onto the result's own axes. *)
        let sa = Option.map (fun (sa, _, _) -> sa) sel in
        let lo = List.filter (fun ax -> Some ax <> sa) (nonunit po) in
        let ys = nonunit y_shape in
        if List.length lo <> List.length ys then raise Abort;
        let pairs =
          List.map2
            (fun o y ->
              let y4 = get (Axis4.of_axis y) in
              let src = Permute.Permute.lookup perm o in
              if not (Dim.equal (extent tgt src) (extent y_shape y)) then
                raise Abort;
              let idx = get (index_of src tgt_axes) in
              (y4, List.nth labels idx))
            lo ys
        in
        let claimed = List.map snd pairs in
        let spare = List.filter (fun a -> not (List.mem a claimed)) Axis4.all in
        let unset =
          List.filter
            (fun a -> not (List.exists (fun (o, _) -> o = a) pairs))
            Axis4.all
        in
        if List.length spare <> List.length unset then raise Abort;
        let perm4 =
          List.map
            (fun o ->
              match List.assoc_opt o pairs with
              | Some i -> (o, i)
              | None ->
                  let rec nth_unset i = function
                    | [] -> raise Abort
                    | a :: r -> if a = o then i else nth_unset (i + 1) r
                  in
                  (o, List.nth spare (nth_unset 0 unset)))
            Axis4.all
        in
        let y4 = get (Result.to_option (Shape4.of_vec6 y_shape)) in
        let permute =
          if List.for_all (fun (o, i) -> o = i) perm4 then None
          else
            Some
              {
                op = (fun x -> Op.Permute4 { Ops4.Permute4.perm = perm4; x });
                shape = y4;
              }
        in
        match List.filter_map Fun.id [ slice; reshape; permute ] with
        | [] -> None
        | steps ->
            (* The last step must land on [y_shape] itself; the slice or
               reshape only does when the permutation was the identity. *)
            let final = List.nth steps (List.length steps - 1) in
            if Shape4.equal final.shape y4 then Some steps else None
      with Abort -> None)

(* ---- finding regions ------------------------------------------------------ *)

let sole_use view id =
  if Graph_view.is_graph_output view id then None
  else match Graph_view.uses view id with [ n ] -> Some n | _ -> None

let find_at view (p : node) =
  let ( let* ) = Option.bind in
  match p.Node.op with
  | Permute { Permute.Permute.perm; x = rout } ->
      let* r0 = Graph_view.def view rout in
      (* A clone between the reshape and the permute moves nothing, so it is
         absorbed with the reshape; its input is then one more internal
         tensor. *)
      let* r, clone_members, clone_internal =
        match r0.Node.op with
        | Clone { Pointwise.Clone.x = reshaped } ->
            let* r = Graph_view.def view reshaped in
            let* user = sole_use view reshaped in
            if Node_id.equal user.Node.id r0.Node.id then
              Some (r, [ r0.Node.id ], [ reshaped ])
            else None
        | _ -> Some (r0, [], [])
      in
      let* rx =
        match r.Node.op with
        | Reshape { Reshape.Reshape.x; _ } -> Some x
        | _ -> None
      in
      let* consumer = sole_use view rout in
      if not (Node_id.equal consumer.Node.id p.Node.id) then None
      else
        let* pout = match p.Node.outputs with [ o ] -> Some o | _ -> None in
        let sig_of id =
          Option.map (fun s -> s.Tensor_sig.shape) (Graph_view.sig_of view id)
        in
        let* tgt = sig_of rout in
        let* po = sig_of pout in
        let* x_shape = sig_of rx in
        let* x_sig = Graph_view.sig_of view rx in
        let* pout_sig = Graph_view.sig_of view pout in
        if dialect_valid tgt || (not (is_f32 x_sig)) || not (is_f32 pout_sig)
        then None
        else if dialect_valid po then
          (* The permute's own result is in-domain, only the reshape target is
             not. *)
          let* steps = chain ~x_shape ~tgt ~perm ~po ~y_shape:po ~sel:None in
          Some
            {
              trigger = p.Node.id;
              members = (r.Node.id :: clone_members) @ [ p.Node.id ];
              internal = rout :: clone_internal;
              input = rx;
              interleave = None;
              outputs = [ { id = pout; steps } ];
            }
        else if Graph_view.is_graph_output view pout then None
        else
          let consumers = Graph_view.uses view pout in
          let axis_of (n : node) =
            match n.Node.op with
            | Unbind { Split.Unbind.params; _ } -> Some params.Split.Unbind.axis
            | Select { Split.Select.params; _ } -> Some params.Split.Select.axis
            | _ -> None
          in
          let* sa =
            match List.filter_map axis_of consumers with
            | a :: rest
              when List.length rest + 1 = List.length consumers
                   && List.for_all (fun b -> b = a) rest ->
                Some a
            | _ -> None
          in
          let k_extent = extent po sa in
          let outputs =
            List.map
              (fun (c : node) ->
                match c.Node.op with
                | Unbind _ -> List.mapi (fun k o -> (o, k)) c.Node.outputs
                | Select { Split.Select.params; _ } ->
                    List.map
                      (fun o -> (o, params.Split.Select.index))
                      c.Node.outputs
                | _ -> [])
              consumers
            |> List.concat
          in
          let* outs =
            List.fold_left
              (fun acc (o, k) ->
                let* acc = acc in
                let* k = Dim.index_of ~extent:k_extent (Dim.delta k) in
                let* y_shape = sig_of o in
                let* y_sig = Graph_view.sig_of view o in
                if (not (is_f32 y_sig)) || not (dialect_valid y_shape) then None
                else
                  let* steps =
                    chain ~x_shape ~tgt ~perm ~po ~y_shape
                      ~sel:(Some (sa, k_extent, k))
                  in
                  Some ({ id = o; steps } :: acc))
              (Some []) outputs
          in
          let first =
            List.fold_left
              (fun best (c : node) ->
                match best with
                | None -> Some c
                | Some b ->
                    if
                      Option.compare Graph_view.Position.compare
                        (Graph_view.topo_index view c.Node.id)
                        (Graph_view.topo_index view b.Node.id)
                      < 0
                    then Some c
                    else best)
              None consumers
          in
          let* first = first in
          Some
            {
              trigger = first.Node.id;
              members =
                (r.Node.id :: clone_members)
                @ (p.Node.id :: List.map (fun (c : node) -> c.Node.id) consumers);
              internal = (rout :: clone_internal) @ [ pout ];
              input = rx;
              interleave = None;
              outputs = List.rev outs;
            }
  | _ -> None

(* A run of reshape, clone and permute nodes between two in-domain tensors, each
   interior tensor read only by the next node and at least one of them outside
   the domain: one index permutation of the flat data, planned by
   [Wide_permute]. Walking back from the run's last node stops at the first
   in-domain tensor, so runs are disjoint. *)
let chain_op (n : node) =
  match n.Node.op with
  | Clone { Pointwise.Clone.x } -> Some (Wide_permute.Clone, x)
  | Permute { Permute.Permute.perm; x } -> Some (Wide_permute.Permute perm, x)
  | Reshape { Reshape.Reshape.params; x } ->
      Some (Wide_permute.Reshape params.Reshape.Reshape.shape, x)
  | _ -> None

let step_of = function
  | Wide_permute.Reshape4 shape ->
      {
        op = (fun x -> Op.Reshape4 { Ops4.Reshape4.params = { shape }; x });
        shape;
      }
  | Wide_permute.Permute4 (perm, shape) ->
      { op = (fun x -> Op.Permute4 { Ops4.Permute4.perm; x }); shape }

let find_wide view ~taken =
  let ( let* ) = Option.bind in
  let claimed = ref taken in
  let sig_of id = Graph_view.sig_of view id in
  let in_domain id =
    match sig_of id with
    | Some sg -> dialect_valid sg.Tensor_sig.shape
    | None -> false
  in
  let f32_in_domain id =
    match sig_of id with
    | Some sg -> is_f32 sg && dialect_valid sg.Tensor_sig.shape
    | None -> false
  in
  let rec back (n : node) chain =
    match chain_op n with
    | None -> None
    | Some (_, x) ->
        if in_domain x then Some (x, chain)
        else
          let* d = Graph_view.def view x in
          let* user = sole_use view x in
          if
            Node_id.equal user.Node.id n.Node.id
            && Option.is_some (chain_op d)
            && not (Node_id.Set.mem d.Node.id !claimed)
          then back d (d :: chain)
          else None
  in
  List.filter_map
    (fun (tail : node) ->
      let* o = match tail.Node.outputs with [ o ] -> Some o | _ -> None in
      if Node_id.Set.mem tail.Node.id !claimed || not (f32_in_domain o) then
        None
      else
        let* x, chain = back tail [ tail ] in
        let interior =
          List.concat_map (fun (n : node) -> n.Node.outputs) chain
          |> List.filter (fun id -> not (Tensor_id.equal id o))
        in
        let* x_sig = sig_of x in
        let* o_sig = sig_of o in
        let* x4 = Result.to_option (Shape4.of_vec6 x_sig.Tensor_sig.shape) in
        let* y4 = Result.to_option (Shape4.of_vec6 o_sig.Tensor_sig.shape) in
        if (not (is_f32 x_sig)) || List.for_all in_domain interior then None
        else
          let* steps =
            match
              Wide_permute.plan ~x:x4 ~y:y4
                (List.filter_map (fun n -> Option.map fst (chain_op n)) chain)
            with
            | Some (_ :: _ as steps) -> Some steps
            | Some [] | None -> None
          in
          List.iter
            (fun (n : node) -> claimed := Node_id.Set.add n.Node.id !claimed)
            chain;
          Some
            {
              trigger = tail.Node.id;
              members = List.map (fun (n : node) -> n.Node.id) chain;
              internal = interior;
              input = x;
              interleave = None;
              outputs = [ { id = o; steps = List.map step_of steps } ];
            })
    (Graph_ir.nodes (Graph_view.graph view))

(* [Stack] on an in-frame axis whose only reader is a [Reshape] back into the
   frame: the sin/cos interleave of a rotary embedding. Stacking pushes the
   operands' outer axes outward, so the stacked value carries T or D, but its
   flat layout is (prefix, operand, suffix) around the stacked position: each
   operand reshaped onto the stacked value's non-unit axes with the stacked one
   unit, then concatenated on it, is the same data. *)
let find_interleave view =
  let ( let* ) = Option.bind in
  let in_domain_f32 id =
    match Graph_view.sig_of view id with
    | Some sg -> is_f32 sg && dialect_valid sg.Tensor_sig.shape
    | None -> false
  in
  List.filter_map
    (fun (st : node) ->
      match (st.Node.op, st.Node.outputs) with
      | Stack { Concat.Stack.params; xs = first :: _ :: _ as xs }, [ so ] ->
          let* r = sole_use view so in
          let* ro = match r.Node.outputs with [ o ] -> Some o | _ -> None in
          let* () = match r.Node.op with Reshape _ -> Some () | _ -> None in
          let* stacked = Graph_view.sig_of view so in
          let* first_sig = Graph_view.sig_of view first in
          let* y_sig = Graph_view.sig_of view ro in
          let k = Dim.extent (List.length xs) in
          if
            dialect_valid stacked.Tensor_sig.shape
            || (not (List.for_all in_domain_f32 xs))
            || not (in_domain_f32 ro)
          then None
          else
            let u =
              Concat.Stack.unsqueezed_shape ~x_shape:first_sig.Tensor_sig.shape
                params
            in
            let axis = params.Concat.Stack.axis in
            let before, after =
              let rec split acc = function
                | [] -> (List.rev acc, [])
                | ax :: rest ->
                    if ax = axis then (List.rev acc, rest)
                    else split (ax :: acc) rest
              in
              split [] Axis.all
            in
            let extents axes =
              List.map (extent u)
                (List.filter (fun a -> not (is_one (extent u a))) axes)
            in
            let pre = extents before and suf = extents after in
            let m = List.length pre + 1 + List.length suf in
            if m > 4 then None
            else
              let labels = last_n m Axis4.all in
              let concat_axis = List.nth labels (List.length pre) in
              let shape_with kk =
                List.fold_left2
                  (fun sh label e -> Shape4.set sh label e)
                  (Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:1)
                  labels
                  (pre @ [ kk ] @ suf)
              in
              let unit_shape = shape_with Dim.one
              and concat_shape = shape_with k in
              let* y4 =
                Result.to_option (Shape4.of_vec6 y_sig.Tensor_sig.shape)
              in
              let* x4 =
                Result.to_option (Shape4.of_vec6 first_sig.Tensor_sig.shape)
              in
              let tail =
                if Shape4.equal concat_shape y4 then []
                else
                  [
                    {
                      op =
                        (fun x ->
                          Op.Reshape4
                            { Ops4.Reshape4.params = { shape = y4 }; x });
                      shape = y4;
                    };
                  ]
              in
              Some
                {
                  trigger = r.Node.id;
                  members = [ st.Node.id; r.Node.id ];
                  internal = [ so ];
                  input = first;
                  interleave =
                    Some
                      {
                        inputs = xs;
                        reshape = not (Shape4.equal x4 unit_shape);
                        unit_shape;
                        axis = concat_axis;
                        concat_shape;
                      };
                  outputs = [ { id = ro; steps = tail } ];
                }
      | _ -> None)
    (Graph_ir.nodes (Graph_view.graph view))

let find view =
  let regions =
    List.filter_map (find_at view) (Graph_ir.nodes (Graph_view.graph view))
  in
  let taken =
    List.fold_left
      (fun set r ->
        List.fold_left (fun set id -> Node_id.Set.add id set) set r.members)
      Node_id.Set.empty regions
  in
  regions @ find_wide view ~taken @ find_interleave view

(* ---- emitting a region ---------------------------------------------------- *)

let rec chain_steps acc ~from cur steps id =
  match steps with
  | [] -> acc
  | [ step ] -> emit acc ~from (step.op cur) [ id ]
  | step :: rest ->
      let mid, acc = fresh_tensor acc step.shape in
      let acc = emit acc ~from (step.op cur) [ mid ] in
      chain_steps acc ~from mid rest id

let emit_interleave acc r il (o : output) =
  let xs, acc =
    List.fold_left
      (fun (xs, acc) id ->
        let x = resolve acc id in
        if il.reshape then
          let y, acc = fresh_tensor acc il.unit_shape in
          let acc =
            emit acc ~from:r.trigger
              (Op.Reshape4
                 { Ops4.Reshape4.params = { shape = il.unit_shape }; x })
              [ y ]
          in
          (y :: xs, acc)
        else (x :: xs, acc))
      ([], acc) il.inputs
  in
  let concat =
    Op.Concat4 { Ops4.Concat4.params = { axis = il.axis }; xs = List.rev xs }
  in
  match o.steps with
  | [] -> emit acc ~from:r.trigger concat [ o.id ]
  | steps ->
      let mid, acc = fresh_tensor acc il.concat_shape in
      let acc = emit acc ~from:r.trigger concat [ mid ] in
      chain_steps acc ~from:r.trigger mid steps o.id

let emit_region acc r =
  match (r.interleave, r.outputs) with
  | Some il, [ o ] -> emit_interleave acc r il o
  | _ ->
      let x = resolve acc r.input in
      List.fold_left
        (fun acc (o : output) -> chain_steps acc ~from:r.trigger x o.steps o.id)
        acc r.outputs
