(* One global SSA environment; structural groups do not affect evaluation. *)

open Graph_ir

type context = Operand | Sig_shape
type missing_tensor = { context : context; id : Tensor_id.t }
type arity_mismatch = { expected : int; actual : int }

type mixed_dtype = {
  mixed_op : string;
  a_fmt : Payload.packed_fmt;
  b_fmt : Payload.packed_fmt;
}

type scalar_op = { scalar_op : string; fmt : Payload.packed_fmt }

type error =
  [ Eval_direct_compute.error
  | Graph_shape.error
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Missing_tensor of missing_tensor
  | `Output_arity_mismatch of arity_mismatch
  | `Region_construction of Region_computation.error
  | `Region_execution of Region_eval.error
  | `Unsupported_bool_arithmetic of mixed_dtype
  | `Unsupported_bool_scalar_arithmetic of scalar_op
  | `Unsupported_mixed_dtype of mixed_dtype ]

type hooks =
  | Hooks : { on_start : node -> 'a; on_end : node -> 'a -> unit } -> hooks

let pp_context ppf = function
  | Operand -> Format.pp_print_string ppf "operand"
  | Sig_shape -> Format.pp_print_string ppf "shape lookup"

let pp_error ppf : [< error ] -> unit = function
  | #Eval_direct_compute.error as e -> Eval_direct_compute.pp_error ppf e
  | #Graph_shape.error as e -> Graph_shape.pp_error ppf e
  | `Missing_constant id ->
      Format.fprintf ppf "missing constant tensor t%d" (Tensor_id.to_int id)
  | `Missing_input id ->
      Format.fprintf ppf "missing input tensor t%d" (Tensor_id.to_int id)
  | `Missing_tensor { context; id } ->
      Format.fprintf ppf "missing %a tensor t%d" pp_context context
        (Tensor_id.to_int id)
  | `Output_arity_mismatch { expected; actual } ->
      Format.fprintf ppf
        "node output arity mismatch: %d output shapes for %d output ids"
        expected actual
  | `Region_construction error -> Region_computation.pp_error ppf error
  | `Region_execution error -> Region_eval.pp_error ppf error
  | `Unsupported_bool_arithmetic
      { mixed_op; a_fmt = Payload.Fmt a_fmt; b_fmt = Payload.Fmt b_fmt } ->
      Format.fprintf ppf
        "%s: arithmetic on a Bool operand is not supported, a=%s b=%s" mixed_op
        (Payload.fmt_name a_fmt) (Payload.fmt_name b_fmt)
  | `Unsupported_bool_scalar_arithmetic { scalar_op; fmt = Payload.Fmt fmt } ->
      Format.fprintf ppf
        "%s: arithmetic on a Bool operand is not supported, x=%s" scalar_op
        (Payload.fmt_name fmt)
  | `Unsupported_mixed_dtype
      { mixed_op; a_fmt = Payload.Fmt a_fmt; b_fmt = Payload.Fmt b_fmt } ->
      Format.fprintf ppf "%s: unsupported mixed dtype, a=%s b=%s" mixed_op
        (Payload.fmt_name a_fmt) (Payload.fmt_name b_fmt)

let is_bool = function Payload.Fmt Payload.Bool -> true | _ -> false

let find_tensor map id ~context =
  Tensor_id.Map.find_opt id map
  |> Err.of_option (`Missing_tensor { context; id })

let widen (r : ('a, [< error ]) Err.t) : ('a, error) Err.t =
  (r :> ('a, error) Err.t)

let sig_shape (g : graph) r =
  let open Err.Syntax in
  let+ sg = find_tensor g.Graph.tensors r ~context:Sig_shape in
  sg.Tensor_sig.shape

let input_ids g =
  List.filter (fun id -> Graph_ir.input_kind g id = Input.Input) g.Graph.inputs

(* An exported program can retain captured state that no lowered operation
   consumes (for example BatchNorm's int64 num_batches_tracked).  Constants are
   therefore required only when they occur as an actual graph operand. *)
let constant_is_used g id =
  List.exists
    (fun node -> List.mem id (Graph_ir.operands node.Node.op))
    g.Graph.nodes

let bind_constants g constants env =
  Err.List.fold_left
    (fun env id ->
      match Graph_ir.input_kind g id with
      | Input.Constant when not (constant_is_used g id) -> Err.return env
      | Input.Constant -> (
          match List.assoc_opt id constants with
          | None -> Err.fail (`Missing_constant id)
          | Some tensor -> Err.return (Tensor_id.Map.add id tensor env))
      | Input.Input -> Err.return env)
    env g.Graph.inputs

let fresh_synthetic_ids g =
  let rec fresh candidate used =
    let id = Tensor_id.of_int candidate in
    if Tensor_id.Set.mem id used then fresh (candidate + 1) used
    else (id, Tensor_id.Set.add id used)
  in
  List.fold_left
    (fun (ids, used, candidate) role ->
      let id, used = fresh candidate used in
      ((role, id) :: ids, used, candidate + 1))
    ( [],
      Tensor_id.Map.fold
        (fun id _ ids -> Tensor_id.Set.add id ids)
        g.Graph.tensors Tensor_id.Set.empty,
      0 )
    [
      Region_computation.Layer_bias;
      Region_computation.Layer_weight;
      Region_computation.Rms_weight;
      Region_computation.Sdpa_mask;
    ]
  |> fun (ids, _, _) -> ids

let region_result ~limits ~region_counters
    ?(region_executor = Region_executor.default) g ~op ~output ~out_shape
    ~operand_env ~synthetic_ids =
  let open Err.Syntax in
  let id_for role = List.assoc role synthetic_ids in
  (* [fill] is the only place a synthetic operand's default value and shape
     are decided, so it records them here rather than have the caller
     re-derive the same pair afterwards by re-matching [op] -- a second copy
     that can only drift from this one. *)
  let filled = ref Tensor_id.Map.empty in
  let fill role value shape =
    let id = id_for role in
    filled := Tensor_id.Map.add id (value, shape) !filled;
    Tensor_sig.create ~id ~name:"direct optional operand" ~shape
      ~fmt:(Payload.Fmt Payload.F32) ()
  in
  let* program =
    Region_computation.program ~limits ~op ~output ~output_shape:out_shape
      ~operand:(fun id -> Tensor_id.Map.find_opt id g.Graph.tensors)
      ~fill
    |> Err.map_error (fun error -> `Region_construction error)
  in
  let sources = Region_program.Fold.sources program in
  let synthetic_bindings =
    Tensor_id.Map.filter_map
      (fun id (value, shape) ->
        if Expr.Source.Set.mem (Expr_bridge.source_of_id id) sources then
          Some (Tensor.materialize shape (fun _ -> value))
        else None)
      !filled
  in
  let env =
    Expr_bridge.env ~binding:(fun id ->
        match Tensor_id.Map.find_opt id operand_env with
        | Some tensor -> Some tensor
        | None -> Tensor_id.Map.find_opt id synthetic_bindings)
  in
  let bindings =
    Tensor_id.Map.union
      (fun _ tensor _ -> Some tensor)
      operand_env synthetic_bindings
  in
  let* lowered =
    Region_execution.lower ~max_size:limits.Kernel.Limits.max_size
      ~max_depth:limits.Kernel.Limits.max_depth
      ~max_local_slots:limits.Kernel.Limits.max_local_slots
      ~scan_limits:(Kernel.Limits.scan_limits limits)
      ~output_shape:out_shape program
    |> Err.map_error (fun error ->
        `Region_construction (Region_computation.Invalid_program error))
  in
  match lowered with
  | Region_execution.Pixel_loop _ -> assert false
  | Region_execution.Region_loop lowered ->
      region_executor ?counters:region_counters lowered ~env ~bindings
      |> Err.map_error (fun error -> `Region_execution error)

(* The multi-output counterpart of [region_result] (project step 19): builds
   the shared [Region_group.t] ONCE for the whole node and materializes every
   ordinal in [outs] from one shared per-key evaluation, instead of folding
   [region_result] (which would rebuild the whole recurrence) once per
   ordinal. No synthetic [fill]/[synthetic_ids] machinery is needed here --
   [Region_computation.group] only ever builds [Lstm], which (unlike
   [Rms_norm]/[Layer_norm]/[Sdpa]) has no optional operand with a synthetic
   default. *)
let region_group_result ~limits ~region_counters
    ?(region_group_executor = Region_executor.default_group) g ~op ~outs
    ~operand_env =
  let open Err.Syntax in
  let* group =
    Region_computation.group ~limits ~op ~operand:(fun id ->
        Tensor_id.Map.find_opt id g.Graph.tensors)
    |> Err.map_error (fun error -> `Region_construction error)
  in
  let env =
    Expr_bridge.env ~binding:(fun id -> Tensor_id.Map.find_opt id operand_env)
  in
  let* lowered_group =
    Region_execution.lower_group ~max_size:limits.Kernel.Limits.max_size
      ~max_depth:limits.Kernel.Limits.max_depth
      ~max_local_slots:limits.Kernel.Limits.max_local_slots
      ~scan_limits:(Kernel.Limits.scan_limits limits)
      group
    |> Err.map_error (fun error ->
        `Region_construction (Region_computation.Invalid_group error))
  in
  region_group_executor ?counters:region_counters lowered_group ~env
    ~bindings:operand_env
    ~selected:
      (List.map
         (fun (output, _, _) -> Region_computation.emitter_of_output output)
         outs)
  |> Err.map_error (fun error -> `Region_execution error)

(* Edges something reads: a graph output, or an operand of a node that is not a
   [Discard] sink. An index output nothing reads is not worth allocating. *)
let live_edges (g : graph) =
  let add = List.fold_left (fun s id -> Tensor_id.Set.add id s) in
  List.fold_left
    (fun live (node : node) ->
      match node.Node.op with
      | Discard _ -> live
      | op -> add live (Graph_ir.operands op))
    (add Tensor_id.Set.empty g.Graph.outputs)
    g.Graph.nodes

(* The second output of the argmax-style ops. *)
let is_index_output (op : op) output =
  Output_ordinal.equal output Output_ordinal.one
  &&
  match op with
  | Adaptive_max_pool2d_with_indices _ | Max_dim _ | Max_pool2d_with_indices _
    ->
      true
  | _ -> false

let is_i64 = function Payload.Fmt Payload.I64 -> true | _ -> false

(* The Bool and mixed-dtype checked-admission arms (design §4.2, plan T2.1):
   runs once per node, before [Eval_direct_compute.compute] is ever called
   for any of its outputs, so a rejected node reaches neither the direct nor
   (later) a generated-JS path, and both see the identical error. This is
   [Eval_symbolic.check_mixed_dtype]'s own check list, restated as a returned
   [Err.t] rather than a raise -- the two functions cannot share code because
   their error rows differ (this one reuses [Eval_direct]'s own
   [mixed_dtype]/[scalar_op] types), but they must never drift apart, since a
   rejection here and there is what keeps the Symbolic/Direct dispatch
   agreeing at checked admission, not just at the default pixel formula. Does
   NOT cover [To_copy]'s [Long]/[Bool] source-format rejections: those are
   "no accepted cast exists for this source", a different kind of gap, and
   stay inside [Eval_direct_compute.compute]'s own (unchanged) match. *)
let admit (g : graph) (op : op) : (unit, [> error ]) Err.t =
  let open Err.Syntax in
  let fmt_of r = (Tensor_id.Map.find r g.Graph.tensors).Tensor_sig.fmt in
  (* Bool checked FIRST, matching every inline check this replaces: a Bool
     paired with I64 must report the Bool reason, not the unrelated
     I64-mixing one. [Bool.equal (is_i64 a) (is_i64 b)] accepts an
     (I64, I64) pair and any pair of non-Bool non-I64 formats alike, and
     rejects only an exactly-one-I64 pair -- the same three-way split
     [Eval_direct_compute.compute]'s own nested match on [Add]/[Sub]/[Mul]
     makes, restated as one boolean condition instead of three pattern
     cases. *)
  let check_pair mixed_op a b =
    let a_fmt = fmt_of a and b_fmt = fmt_of b in
    if is_bool a_fmt || is_bool b_fmt then
      Err.fail (`Unsupported_bool_arithmetic { mixed_op; a_fmt; b_fmt })
    else if Bool.equal (is_i64 a_fmt) (is_i64 b_fmt) then Err.return ()
    else Err.fail (`Unsupported_mixed_dtype { mixed_op; a_fmt; b_fmt })
  in
  let check_scalar_op scalar_op x =
    let fmt = fmt_of x in
    if is_bool fmt then
      Err.fail (`Unsupported_bool_scalar_arithmetic { scalar_op; fmt })
    else Err.return ()
  in
  match op with
  | Add { Pointwise.Bin.a; b } -> check_pair "add" a b
  | Sub { Pointwise.Bin.a; b } -> check_pair "sub" a b
  | Mul { Pointwise.Bin.a; b } -> check_pair "mul" a b
  | Mul_scalar { Pointwise.Scalar_bin.x; _ } -> check_scalar_op "mul_scalar" x
  | Add_scalar { Pointwise.Scalar_bin.x; _ } -> check_scalar_op "add_scalar" x
  | Div_scalar { Pointwise.Scalar_bin.x; _ } -> check_scalar_op "div_scalar" x
  | Floor_div_scalar { Pointwise.Scalar_bin.x; _ } ->
      check_scalar_op "floor_div_scalar" x
  | Pow { Pointwise.Scalar_bin.x; _ } -> check_scalar_op "pow" x
  | Rpow_scalar { Pointwise.Scalar_bin.x; _ } -> check_scalar_op "rpow_scalar" x
  | Rsub_scalar { Pointwise.Rsub_scalar.x; _ } ->
      check_scalar_op "rsub_scalar" x
  (* Reports whichever of the three operands is Bool first, in that order,
     matching the three separate inline arms this replaces. *)
  | Addcmul { Pointwise.Addcmul.self; tensor1; tensor2; _ } ->
      let* () = check_scalar_op "addcmul" self in
      let* () = check_scalar_op "addcmul" tensor1 in
      check_scalar_op "addcmul" tensor2
  | _ -> Err.return ()

let rec run_graph ?hooks ?region_counters ?region_executor
    ?region_group_executor ?node_executor ?(limits = Kernel.Limits.default)
    ~constants (g : graph) (env : Tensor.packed Tensor_id.Map.t) :
    (Tensor.packed Tensor_id.Map.t, error) Err.t =
  let open Err.Syntax in
  let* env = bind_constants g constants env in
  let synthetic_ids = fresh_synthetic_ids g in
  let live = live_edges g in
  Err.List.fold_left
    (fun env node ->
      match hooks with
      (* A [Discard] produces nothing, and the edge it sinks may be an
         unallocated dead index output: there is nothing to look up. *)
      | _ when match node.Node.op with Discard _ -> true | _ -> false ->
          Err.return env
      | None ->
          eval_node ?region_counters ?region_executor ?region_group_executor
            ?node_executor ~limits ~synthetic_ids ~live g env node
      | Some (Hooks h) ->
          let state = h.on_start node in
          let* env =
            eval_node ?region_counters ?region_executor ?region_group_executor
              ?node_executor ~limits ~synthetic_ids ~live g env node
          in
          h.on_end node state;
          Err.return env)
    env g.Graph.nodes

and eval_node ?region_counters ?region_executor ?region_group_executor
    ?node_executor ~limits ~synthetic_ids ~live (g : graph)
    (env : Tensor.packed Tensor_id.Map.t) (node : node) :
    (Tensor.packed Tensor_id.Map.t, error) Err.t =
  let open Err.Syntax in
  let op = node.Node.op in
  let node_executor =
    Option.value node_executor ~default:Node_executor.default
  in
  let operand r = find_tensor env r ~context:Operand in
  let shape_of r = sig_shape g r in
  let fill v shape = Tensor.materialize shape (fun _ -> v) in
  let* () = admit g op in
  let* shapes =
    widen
      (Graph_shape.output_shape op ~sig_of:(fun r ->
           Tensor_id.Map.find_opt r g.Graph.tensors
           |> Err.of_option (`Missing_tensor_sig r)))
  in
  let* operand_env =
    Err.List.fold_left
      (fun acc r ->
        let+ t = operand r in
        Tensor_id.Map.add r t acc)
      Tensor_id.Map.empty (Graph_ir.operands op)
  in
  let* shape_env =
    Err.List.fold_left
      (fun acc r ->
        let+ sh = shape_of r in
        Tensor_id.Map.add r sh acc)
      Tensor_id.Map.empty (Graph_ir.operands op)
  in
  (* One materialisation per output edge: [Graph_shape] and [Node.outputs]
         agree in length by construction (single-output ops give one of each; a
         [Discard]-style zero-output op gives none, so the fold is empty). *)
  let* pairs =
    Err.List.map2
      ~unequal_lengths:(fun actual expected ->
        `Output_arity_mismatch { expected; actual })
      (fun oid out_shape -> Err.return (oid, out_shape))
      node.Node.outputs shapes
  in
  (* A dead index output is neither computed nor allocated: nothing reads it,
     so [env] never needs it. *)
  let outs =
    List.mapi
      (fun output (oid, out_shape) ->
        (Output_ordinal.of_int output, oid, out_shape))
      pairs
    |> List.filter (fun (output, oid, _) ->
        not (is_index_output op output && not (Tensor_id.Set.mem oid live)))
  in
  (* A multi-output region-authored node (today, only [Lstm]) shares one
     recurrence across all its outputs (project step 19) instead of folding
     [region_result] -- which would rebuild the shared computation once per
     ordinal -- over [outs]. Every other multi-output op (`Unbind`,
     `Split_with_sizes`, ...) has its own dedicated arm above the
     [is_region_authored] branch, so this length check only ever selects a
     region-authored multi-output op; single-output region-authored ops
     (`Rms_norm`/`Layer_norm`/`Sdpa`/`Softmax`) keep the exact existing path,
     unchanged. *)
  match outs with
  | _ :: _ :: _ when Region_computation.is_region_authored op ->
      let* results =
        region_group_result ~limits
          ~region_counters:
            (let _, first_oid, _ = List.hd outs in
             Option.bind region_counters (fun counters ->
                 Tensor_id.Map.find_opt first_oid counters))
          ?region_group_executor g ~op ~outs ~operand_env
      in
      Err.return
        (List.fold_left
           (fun env (output, oid, _) ->
             Tensor_id.Map.add oid
               (List.assoc
                  (Region_computation.emitter_of_output output)
                  results)
               env)
           env outs)
  | _ ->
      Err.List.fold_left
        (fun env (output, oid, out_shape) ->
          let* result =
            if Region_computation.is_region_authored op then
              region_result ~limits
                ~region_counters:
                  (Option.bind region_counters (fun counters ->
                       Tensor_id.Map.find_opt oid counters))
                ?region_executor g ~op ~output ~out_shape ~operand_env
                ~synthetic_ids
            else
              widen
                (node_executor.Node_executor.run g node ~output ~out_shape
                   ~operands:operand_env ~direct:(fun () ->
                     Eval_direct_compute.compute g op ~output ~out_shape
                       ~operand_env ~shape_env ~fill))
          in
          Err.return (Tensor_id.Map.add oid result env))
        env outs

let run ?hooks ?region_counters ?region_executor ?region_group_executor
    ?node_executor ?(limits = Kernel.Limits.default) ?(constants = [])
    (g : graph) ~(inputs : (Tensor_id.t * Tensor.packed) list) =
  let provided =
    List.fold_left
      (fun e (id, t) -> Tensor_id.Map.add id t e)
      Tensor_id.Map.empty inputs
  in
  let open Err.Syntax in
  let* env0 =
    Err.List.fold_left
      (fun env id ->
        match Tensor_id.Map.find_opt id provided with
        | None -> Err.fail (`Missing_input id)
        | Some tensor -> Err.return (Tensor_id.Map.add id tensor env))
      Tensor_id.Map.empty (input_ids g)
  in
  run_graph ?hooks ?region_counters ?region_executor ?region_group_executor
    ?node_executor ~limits ~constants g env0
