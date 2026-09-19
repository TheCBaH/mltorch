(* See kernel_eval.mli. *)

module Binding_mismatch = struct
  type kind =
    | Format
    | Quant
    | Shape
    | Storage_length of { expected : int; actual : int }

  type t = { id : Tensor_id.t; kind : kind }

  let pp fmt { id; kind } =
    match kind with
    | Format ->
        Fmt.pf fmt "%a: bound tensor has the wrong format" Tensor_id.pp id
    | Quant ->
        Fmt.pf fmt "%a: bound tensor has the wrong quantization" Tensor_id.pp id
    | Shape -> Fmt.pf fmt "%a: bound tensor has the wrong shape" Tensor_id.pp id
    | Storage_length { expected; actual } ->
        Fmt.pf fmt "%a: bound tensor holds %d cells, expected %d" Tensor_id.pp
          id actual expected
end

type error =
  [ Expr.Eval.error
  | `Binding_mismatch of Binding_mismatch.t
  | `Duplicate_group_ordinal of int
  | `Recursion_too_deep of int
  | Region_group.error
  | Region_partition.error
  | Region_program.error
  | `Unbound_input of Tensor_id.t
  | `Unknown_value of Tensor_id.t ]

let pp_error fmt : [< error ] -> unit = function
  | #Expr.Eval.error as e -> Expr.Eval.pp_error fmt e
  | `Binding_mismatch m -> Binding_mismatch.pp fmt m
  | `Duplicate_group_ordinal ordinal ->
      Fmt.pf fmt "group run repeats emitter ordinal %d" ordinal
  | `Recursion_too_deep n ->
      Fmt.pf fmt "recursive evaluation nested more than %d producers deep" n
  | #Region_group.error as e -> Region_group.pp_error fmt e
  | #Region_partition.error as e -> Region_partition.pp_error fmt e
  | #Region_program.error as e -> Region_program.pp_error fmt e
  | `Unbound_input id -> Fmt.pf fmt "no binding for input %a" Tensor_id.pp id
  | `Unknown_value id -> Fmt.pf fmt "%a names no value" Tensor_id.pp id

(* ---- the per-run input environment ----------------------------------------

   Resolved and validated BEFORE anything is evaluated, which is what makes
   [`Unbound_input] and [`Binding_mismatch] expressible at all:
   [Expr.Eval.Env.t.load] has the fixed result type
   [(float, Expr.Eval.error) Err.t], and this row is wider. With a
   validated total map, [Expr_bridge.env] can only ever report a genuine
   [`Unknown_source]. *)

let same_shape a b =
  List.for_all
    (fun ax -> Dim.equal (Vec6.get a ax) (Vec6.get b ax))
    Expr.Axis.all

let storage_cells (Tensor.Tensor t) =
  Bigarray.Array1.dim t.Tensor.payload.Payload.data

let payload_quant (Tensor.Tensor t) =
  match t.Tensor.payload.Payload.quant with
  | Payload.No_quant -> None
  | Payload.Quant q -> Some q

(* Shape, format and quantization must match what the kernel DECLARES, and the
   dense payload must actually hold numel cells.

   None of this is bookkeeping. [Expr_bridge.env] reads the BOUND tensor's shape
   and payload, never the declared [Tensor_sig.t], so a wrong-but-present
   binding is silently decoded under a different format or quantization rule and
   the bounds proved for the signature never apply to the tensor actually read.
   The length check is what keeps reads exception-safe: there is no tensor.mli,
   so [Tensor.packed] is a public record and a caller can build one whose shape
   matches while its Bigarray is short — [Tensor.read] bounds-checks the
   COORDINATE against the shape and then indexes [data.{i}], which raises out of
   [Payload.get_float] at an otherwise in-range offset. *)
let check_binding id (sg : Tensor_sig.t) tensor =
  let mismatch kind =
    Err.fail (`Binding_mismatch { Binding_mismatch.id; kind })
  in
  let (Tensor.Tensor t) = tensor in
  if not (same_shape sg.Tensor_sig.shape t.Tensor.shape) then
    mismatch Binding_mismatch.Shape
  else
    let fmt_name (Payload.Fmt f) = Payload.fmt_name f in
    if
      not
        (String.equal
           (fmt_name sg.Tensor_sig.fmt)
           (Payload.fmt_name t.Tensor.payload.Payload.fmt))
    then mismatch Binding_mismatch.Format
    else
      let quant_ok =
        match (sg.Tensor_sig.quant, payload_quant tensor) with
        | None, None -> true
        | Some a, Some b -> Quant.equal a b
        | Some _, None | None, Some _ -> false
      in
      if not quant_ok then mismatch Binding_mismatch.Quant
      else
        let expected = (Vec6.numel sg.Tensor_sig.shape :> int) in
        let actual = storage_cells tensor in
        if expected <> actual then
          mismatch (Binding_mismatch.Storage_length { expected; actual })
        else Err.return ()

let input_env (k : Kernel.t) ~bind =
  let open Err.Syntax in
  List.fold_left
    (fun acc (i : Kernel.Input.t) ->
      let* m = acc in
      match i.Kernel.Input.binding with
      | Kernel.Binding.Filled v ->
          (* Materialised here, exactly as [Stage_program.ground] does, NOT
             handed through as an OCaml float: the store is f32, so a fill that
             is not representable there must be rounded before any consumer
             reads it. *)
          let shape = i.Kernel.Input.sg.Tensor_sig.shape in
          let filled =
            match i.Kernel.Input.sg.Tensor_sig.fmt with
            | Payload.Fmt Payload.Bool ->
                Tensor.materialize_bool shape (fun _ -> v <> 0.)
            | _ -> Tensor.materialize shape (fun _ -> v)
          in
          Err.return (Tensor_id.Map.add i.Kernel.Input.id filled m)
      | Kernel.Binding.Caller | Kernel.Binding.Captured_constant -> (
          match bind i.Kernel.Input.id with
          | None -> Err.fail (`Unbound_input i.Kernel.Input.id)
          | Some tensor ->
              let+ () =
                check_binding i.Kernel.Input.id i.Kernel.Input.sg tensor
              in
              Tensor_id.Map.add i.Kernel.Input.id tensor m))
    (Err.return Tensor_id.Map.empty)
    k.Kernel.inputs

(* ---- evaluation ------------------------------------------------------------

   [Tensor.materialize] takes [Vec6.coord -> float] and cannot carry a result,
   so the callback signals through [Err.Escape] and the boundary converts once.
   Reusing [Schedule.ground] instead would put [Err.or_raise ~pp_error:] on the
   path and let [Expr.Eval] failures escape as exceptions through an API
   promising [Err.t]. [Escape.or_throw] re-returns the original error, never
   rebuilding it by hand, so its detection backtrace survives. *)

let widen r = Err.map_error (fun (e : Expr.Eval.error) -> (e :> error)) r

let widen_region r =
  Err.map_error (fun (e : Region_eval.error) -> (e :> error)) r

let widen_program r =
  Err.map_error (fun (e : Region_program.error) -> (e :> error)) r

let widen_group r =
  Err.map_error (fun (e : Region_group.error) -> (e :> error)) r

let coord_key (c : int Expr.Coord.t) =
  ( c.Expr.Coord.n,
    c.Expr.Coord.t,
    c.Expr.Coord.d,
    c.Expr.Coord.h,
    c.Expr.Coord.w,
    c.Expr.Coord.c )

let values_by_id (k : Kernel.t) =
  List.fold_left
    (fun m (v : Kernel.Value.t) -> Tensor_id.Map.add v.Kernel.Value.id v m)
    Tensor_id.Map.empty k.Kernel.values

(* The value as its consumers see it: its emitter wrapped in its result
   conversion, applied exactly once. Built once per value rather than per
   coordinate. A Pixel program stays on the existing direct expression path;
   this distinction is intentional, because it preserves the tight per-cell
   evaluator for the overwhelmingly common singleton case.

   [Result_conversion.apply] rewrites the emitter via [with_output] AFTER
   [v.Kernel.Value.computation] was already checked at construction --
   [with_output] is a raw record update, not [Region_program.create] -- so
   [lower_region]'s own re-validation against the kernel's limits is what
   covers this rewritten expression; nothing upstream of it does. *)
let converted esc ?region_counters ~(limits : Kernel.Limits.t)
    (v : Kernel.Value.t) =
  match Region_group.Ref.pixel_expression v.Kernel.Value.computation with
  | Some body ->
      `Pixel (Kernel.Result_conversion.apply v.Kernel.Value.result body)
  | None ->
      (* [v.Kernel.Value.computation] is PROJECTED to a concrete
         [Region_program.t] first (a [Grouped] value's shared locals are
         rebuilt/re-checked here independently of any sibling -- project
         step 19's Section C milestone 2 is what removes this per-value
         duplication for real), then the result conversion is spliced in
         and re-lowered exactly as before. *)
      let program =
        Err.Escape.or_throw esc
          (widen_group
             (Region_group.Ref.project ~max_size:limits.Kernel.Limits.max_size
                ~max_depth:limits.Kernel.Limits.max_depth
                v.Kernel.Value.computation))
      in
      let lowered =
        Err.Escape.or_throw esc
          (widen_program
             (Region_execution.lower_region
                ~max_size:limits.Kernel.Limits.max_size
                ~max_depth:limits.Kernel.Limits.max_depth
                ~max_local_slots:limits.Kernel.Limits.max_local_slots
                ~scan_limits:(Kernel.Limits.scan_limits limits)
                ~output_shape:v.Kernel.Value.sg.Tensor_sig.shape
                (Region_program.with_output program
                   (Kernel.Result_conversion.apply v.Kernel.Value.result
                      (Region_program.output program)))))
      in
      `Region
        ( lowered,
          Option.bind region_counters (fun counters ->
              Tensor_id.Map.find_opt v.Kernel.Value.id counters) )

(* A value computed on the float path becomes its declared storage: a
   Bool-declared one is written as canonical Bool bytes, everything else stays
   the F32 tensor the float path produced. Applied at the one place a value is
   stored, so a consumer reads the Bool payload whether the producer ran solo
   or in a group. *)
let stored (v : Kernel.Value.t) tensor =
  match v.Kernel.Value.sg.Tensor_sig.fmt with
  | Payload.Fmt Payload.Bool -> Tensor.bool_of_float_cells tensor
  | _ -> tensor

let in_shape (sg : Tensor_sig.t) (c : int Expr.Coord.t) =
  List.find_opt
    (fun a ->
      let i = Expr.Coord.get c a in
      i < 0 || i >= Dim.to_int (Vec6.get sg.Tensor_sig.shape a))
    Expr.Axis.all

(* One engine for both placements. A value in [stores] is materialised; a load
   on an edge in [virtual_uses] recurses into its producer instead of reading a
   buffer. [run] is this with nothing virtual and everything stored.

   Returns the [values_i64] results too, alongside the three float-side
   closures: [execute] needs them for its own final union, and computing them
   here (rather than a second time in [execute]) is what makes seeding
   [bound] below and returning the same map to the caller one materialization,
   not two. *)
let machine esc ?on_load ?region_counters (k : Kernel.t) ~bind ~virtual_uses =
  let inputs = Err.Escape.or_throw esc (input_env k ~bind) in
  let values = values_by_id k in
  let scan_limits = Kernel.Limits.scan_limits k.Kernel.limits in
  let bodies =
    Tensor_id.Map.map
      (converted esc ?region_counters ~limits:k.Kernel.limits)
      values
  in
  (* Every stored tensor, inputs included. An int64 value and a float value may
     each read the other ([Kernel.create] proved the combined graph acyclic),
     so neither list runs "first": a load that misses here asks [ensure] to
     produce the tensor on demand, memoised in this same map. [ensure] is
     filled in once [materialize]/[materialize_group] exist. *)
  let bound = ref inputs in
  let ensure = ref (fun (_ : Tensor_id.t) -> ()) in
  let binding i =
    match Tensor_id.Map.find_opt i !bound with
    | Some _ as found -> found
    | None ->
        !ensure i;
        Tensor_id.Map.find_opt i !bound
  in
  (* An edge is virtual only for its NOMINATED consumer, so the question is
     always "does this consumer recurse into that producer", never "is that
     producer virtual". A producer both virtual and stored is materialised as
     well, and read from its buffer by everyone else. *)
  let is_virtual ~consumer ~producer =
    Kernel.Use.Set.mem { Kernel.Use.producer; consumer } virtual_uses
  in
  let memo = Hashtbl.create 64 in
  (* Depth is carried, not measured: the guard has to fire BEFORE the frame it
     would have pushed, which a post-hoc count cannot do. *)
  let rec eval_value ~depth id coord =
    if depth > Kernel.Limits.Hard.eval_recursion then
      Err.Escape.throw esc
        (`Recursion_too_deep Kernel.Limits.Hard.eval_recursion);
    match Tensor_id.Map.find_opt id values with
    | None -> Err.Escape.throw esc (`Unknown_value id)
    | Some (v : Kernel.Value.t) -> (
        match in_shape v.Kernel.Value.sg coord with
        | Some a ->
            Err.Escape.throw esc
              (`Coord_out_of_range
                 (Expr_bridge.source_of_id id, a, Expr.Coord.get coord a, coord))
        | None -> (
            let key = (Tensor_id.to_int id, coord_key coord) in
            match Hashtbl.find_opt memo key with
            | Some x -> x
            | None ->
                let x =
                  match Tensor_id.Map.find id bodies with
                  | `Pixel body ->
                      Err.Escape.or_throw esc
                        (widen
                           (Expr.Eval.value
                              ~scan_meter:
                                (Expr.Scan_meter.create ~limits:scan_limits)
                              (env_for ~depth id) ~output:coord body))
                  | `Region (lowered, _) ->
                      Err.Escape.or_throw esc
                        (widen_region
                           (Region_execution.value_at lowered
                              ~env:(env_for ~depth id)
                              ~output:
                                (Vec6.map Dim.index
                                   (Expr_bridge.vec6_of_coord coord))))
                in
                Hashtbl.add memo key x;
                x))
  and env_for ~depth consumer =
    let bridge = Expr_bridge.env ~binding in
    let load_bound =
      match on_load with
      | None -> bridge.Expr.Eval.Env.load
      | Some observe ->
          fun src coord ->
            observe (Expr_bridge.id_of_source src) coord;
            bridge.Expr.Eval.Env.load src coord
    in
    {
      Expr.Eval.Env.load =
        (fun src c ->
          let producer = Expr_bridge.id_of_source src in
          if is_virtual ~consumer ~producer then
            Err.return (eval_value ~depth:(depth + 1) producer c)
          else load_bound src c);
      Expr.Eval.Env.load_index =
        (fun src c ->
          let producer = Expr_bridge.id_of_source src in
          if is_virtual ~consumer ~producer then
            (* Dead: a [Data] index component is never fusion-admitted --
               [kernel_elab.ml]'s pointwise-admission predicate rejects it via
               its existing catch-all, so no consumer can ever recurse into a
               virtual producer through this arm. But [Env.t]'s [load_index]
               field has a fixed, non-polymorphic row ([Expr.Eval.error]), so
               this function must still type-check against it regardless.
               Reuses the same [`Data_index_unexpected_here] tag the public
               [Eval.index]'s own "impossible resolver" uses (rather than a
               second, separately-typed tag) -- both call sites mean exactly
               the same thing: a [Data] index reached a context that
               structurally cannot produce one. *)
            Err.fail `Data_index_unexpected_here
          else bridge.Expr.Eval.Env.load_index src c);
    }
  in
  (* Materialising one value, and evaluating one cell on demand: the same
     recursion, so the depth guard cannot cover one path and miss the other. *)
  let materialize_fresh (v : Kernel.Value.t) =
    let env = env_for ~depth:0 v.Kernel.Value.id in
    let t =
      match Tensor_id.Map.find v.Kernel.Value.id bodies with
      | `Pixel body ->
          Tensor.materialize v.Kernel.Value.sg.Tensor_sig.shape (fun c ->
              Err.Escape.or_throw esc
                (widen
                   (Expr.Eval.value
                      ~scan_meter:(Expr.Scan_meter.create ~limits:scan_limits)
                      env
                      ~output:
                        (Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int c))
                      body)))
      | `Region (lowered, counters) ->
          Err.Escape.or_throw esc
            (widen_region (Region_execution.materialize ?counters lowered ~env))
    in
    let t = stored v t in
    bound := Tensor_id.Map.add v.Kernel.Value.id t !bound;
    t
  in
  (* Idempotent: a float value an int64 value read on demand is already here
     when [execute] reaches its own turn. *)
  let materialize (v : Kernel.Value.t) =
    match Tensor_id.Map.find_opt v.Kernel.Value.id !bound with
    | Some t -> t
    | None -> materialize_fresh v
  in
  (* Materialises several sibling [Kernel.Value.t]s that share one physically
     identical [Region_group.t] in ONE shared evaluation (project step 19,
     Section C milestone 2 -- the Kernel-side counterpart of
     [Stage_program.ground]'s own [execute_run] [Group_run] arm). Every
     [members] entry is guaranteed never-virtual: [Kernel_elab.admit]'s very
     first check rejects any [Use] naming a [Regional_computation] endpoint,
     and [Region_group.Ref.pixel_expression] is unconditionally [None] for a
     [Grouped] value, so no fusion plan can ever place one on either side of
     [virtual_uses] -- which is what makes ANY single member's own id a safe
     [~consumer] for [env_for]'s virtual-load lookups here, regardless of
     which one is chosen. Updates the SAME [bound] ref [materialize]'s solo
     path does, so a later solo value that loads a grouped sibling as an
     ordinary source resolves it correctly. *)
  let materialize_group_fresh (g : Region_group.t)
      (members : (int * Kernel.Value.t) list) =
    (match
       Region_group.Run.duplicate_ordinal (Region_group.Run.Group (g, members))
     with
    | Some ordinal -> Err.Escape.throw esc (`Duplicate_group_ordinal ordinal)
    | None -> ());
    match members with
    | [] -> []
    | (_, first) :: _ ->
        let lowered_group =
          Err.Escape.or_throw esc
            (widen_group
               (Region_execution.lower_group
                  ~max_size:k.Kernel.limits.Kernel.Limits.max_size
                  ~max_depth:k.Kernel.limits.Kernel.Limits.max_depth
                  ~max_local_slots:k.Kernel.limits.Kernel.Limits.max_local_slots
                  ~scan_limits g))
        in
        let env = env_for ~depth:0 first.Kernel.Value.id in
        let counters =
          Option.bind region_counters (fun m ->
              Tensor_id.Map.find_opt first.Kernel.Value.id m)
        in
        let results =
          Err.Escape.or_throw esc
            (widen_region
               (Region_execution.materialize_group ?counters lowered_group ~env
                  ~selected:(List.map fst members)))
        in
        List.map
          (fun (ordinal, tensor) ->
            let st = List.assoc ordinal members in
            let tensor = stored st tensor in
            bound := Tensor_id.Map.add st.Kernel.Value.id tensor !bound;
            (st.Kernel.Value.id, tensor))
          results
  in
  let materialize_group g members =
    if
      members <> []
      && List.for_all
           (fun (_, (st : Kernel.Value.t)) ->
             Tensor_id.Map.mem st.Kernel.Value.id !bound)
           members
    then
      List.map
        (fun (_, (st : Kernel.Value.t)) ->
          (st.Kernel.Value.id, Tensor_id.Map.find st.Kernel.Value.id !bound))
        members
    else materialize_group_fresh g members
  in
  (* An int64 value is always stored, never virtual: it is computed once, from
     whatever it reads, the first time anything asks for it. *)
  let materialize_i64 (v : Kernel.Value_i64.t) =
    match Tensor_id.Map.find_opt v.Kernel.Value_i64.id !bound with
    | Some t -> t
    | None ->
        let tensor =
          Err.Escape.or_throw esc
            (widen_region
               (Region_eval.materialize_i64
                  ~output_shape:v.Kernel.Value_i64.sg.Tensor_sig.shape
                  ~env:(Expr_bridge.env ~binding) v.Kernel.Value_i64.pixel))
        in
        bound := Tensor_id.Map.add v.Kernel.Value_i64.id tensor !bound;
        tensor
  in
  let run_of =
    lazy
      (List.fold_left
         (fun m run ->
           let members =
             match run with
             | Region_group.Run.Solo v -> [ v ]
             | Region_group.Run.Group (_, members) -> List.map snd members
           in
           List.fold_left
             (fun m (v : Kernel.Value.t) ->
               Tensor_id.Map.add v.Kernel.Value.id run m)
             m members)
         Tensor_id.Map.empty
         (Region_group.runs
            ~computation:(fun (v : Kernel.Value.t) ->
              v.Kernel.Value.computation)
            k.Kernel.values))
  in
  (ensure :=
     fun id ->
       match Tensor_id.Map.find_opt id k.Kernel.by_id_i64 with
       | Some v -> ignore (materialize_i64 v)
       | None -> (
           match Tensor_id.Map.find_opt id (Lazy.force run_of) with
           | Some (Region_group.Run.Solo v) -> ignore (materialize v)
           | Some (Region_group.Run.Group (g, members)) ->
               ignore (materialize_group g members)
           | None -> ()));
  let finish () =
    List.fold_left
      (fun results (v : Kernel.Value_i64.t) ->
        Tensor_id.Map.add v.Kernel.Value_i64.id (materialize_i64 v) results)
      Tensor_id.Map.empty k.Kernel.values_i64
  in
  ( materialize,
    materialize_group,
    (fun id coord -> eval_value ~depth:0 id coord),
    finish )

let execute esc ?on_load ?region_counters (k : Kernel.t) ~bind ~virtual_uses
    ~stores =
  let materialize, materialize_group, _, finish =
    machine esc ?on_load ?region_counters k ~bind ~virtual_uses
  in
  let runs =
    Region_group.runs
      ~computation:(fun (v : Kernel.Value.t) -> v.Kernel.Value.computation)
      k.Kernel.values
  in
  let results =
    List.fold_left
      (fun results run ->
        match run with
        | Region_group.Run.Solo v ->
            if not (Tensor_id.Set.mem v.Kernel.Value.id stores) then results
            else Tensor_id.Map.add v.Kernel.Value.id (materialize v) results
        | Region_group.Run.Group (g, members) ->
            let selected =
              List.filter
                (fun (_, v) -> Tensor_id.Set.mem v.Kernel.Value.id stores)
                members
            in
            List.fold_left
              (fun results (id, tensor) -> Tensor_id.Map.add id tensor results)
              results
              (materialize_group g selected))
      Tensor_id.Map.empty runs
  in
  Tensor_id.Map.union (fun _ _ i64 -> Some i64) results (finish ())

let run ?on_load ?region_counters k ~bind =
  Err.Escape.with_escape @@ fun esc ->
  execute esc ?on_load ?region_counters k ~bind
    ~virtual_uses:Kernel.Use.Set.empty
    ~stores:
      (List.fold_left
         (fun s (v : Kernel.Value.t) -> Tensor_id.Set.add v.Kernel.Value.id s)
         Tensor_id.Set.empty k.Kernel.values)

let run_plan (p : Fusion_plan.t) ~bind =
  Err.Escape.with_escape @@ fun esc ->
  execute esc p.Fusion_plan.kernel ~bind
    ~virtual_uses:p.Fusion_plan.virtual_uses ~stores:p.Fusion_plan.stores

(* Every value virtual and nothing stored: the fully on-demand reading, and the
   one a virtual placement uses per edge. Shares [execute]'s guarded recursion
   rather than repeating it, so the depth bound cannot apply to one path and not
   the other. *)
let value_at k ~bind id coord =
  Err.Escape.with_escape @@ fun esc ->
  match Kernel.value k id with
  | None -> Err.Escape.throw esc (`Unknown_value id)
  | Some (v : Kernel.Value.t) -> (
      match in_shape v.Kernel.Value.sg coord with
      | Some a ->
          Err.Escape.throw esc
            (`Coord_out_of_range
               (Expr_bridge.source_of_id id, a, Expr.Coord.get coord a, coord))
      | None ->
          (* [Kernel.uses] is exactly "every internal source dependency", which
             is what fully virtual execution means, and it builds its value-id
             set once. Refolding the bodies here and asking [Kernel.value] per
             source made setup O(values x source occurrences) before evaluation
             or memoisation had begun — the same linear-lookup-per-source shape
             already removed from [edges_of]. *)
          let _, _, eval, _ =
            machine esc k ~bind ~virtual_uses:(Kernel.uses k)
          in
          eval id coord)
