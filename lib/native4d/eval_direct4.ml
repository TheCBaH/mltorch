(* Direct evaluation of a Native4D graph. The twin of [Eval_direct], and
   structurally identical to it: one global SSA environment, one materialisation
   per output edge, structural groups ignored.

   It is a twin rather than a shared driver because [Eval_direct] is typed on
   [Graph_ir.graph] and the two records differ. Stage 4 makes the record shared;
   the driver could follow, but the arithmetic that matters — [Eval_op4]'s
   delegation to Native's [Compute (S)] — is already shared, which is the reuse
   the design asks for. *)

module E = Eval_op4.Make (Direct)

type context = Operand | Sig_shape
type missing_tensor = { context : context; id : Tensor_id.t }
type arity_mismatch = { expected : int; actual : int }

type error =
  [ Graph_shape4.error
  | `Arange_i64_overflow of Factory.Arange.Overflow.t
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Missing_tensor of missing_tensor
  | `Output_arity_mismatch of arity_mismatch
  | `Region_construction of Region_computation4.error
  | `Region_execution of Region_eval.error ]

let pp_context ppf = function
  | Operand -> Fmt.string ppf "operand"
  | Sig_shape -> Fmt.string ppf "shape lookup"

let pp_error ppf : [< error ] -> unit = function
  | #Graph_shape4.error as e -> Graph_shape4.pp_error ppf e
  | `Arange_i64_overflow { Factory.Arange.Overflow.start; step; i } ->
      Fmt.pf ppf
        "arange4: exact int64 generation overflows at start=%Ld step=%Ld i=%d"
        start step i
  | `Missing_constant id ->
      Fmt.pf ppf "missing constant tensor %a" Tensor_id.pp id
  | `Missing_input id -> Fmt.pf ppf "missing input tensor %a" Tensor_id.pp id
  | `Missing_tensor { context; id } ->
      Fmt.pf ppf "missing %a tensor %a" pp_context context Tensor_id.pp id
  | `Output_arity_mismatch { expected; actual } ->
      Fmt.pf ppf
        "node output arity mismatch: %d output shapes for %d output ids"
        expected actual
  | `Region_construction error -> Region_computation.pp_error ppf error
  | `Region_execution error -> Region_eval.pp_error ppf error

let find_tensor map id ~context =
  Tensor_id.Map.find_opt id map
  |> Err.of_option (`Missing_tensor { context; id })

let widen (r : ('a, [< error ]) Err.t) : ('a, error) Err.t =
  (r :> ('a, error) Err.t)

let sig_shape (g : Graph.graph) r =
  let open Err.Syntax in
  let+ sg = find_tensor g.Graph.Graph.tensors r ~context:Sig_shape in
  sg.Tensor_sig.shape

let input_ids (g : Graph.graph) =
  List.filter
    (fun id -> Graph.input_kind g id = Graph_ir.Input.Input)
    g.Graph.Graph.inputs

(* As in Native: an exported program can retain captured state no lowered
   operation consumes, so a constant is required only where it is an operand. *)
let constant_is_used (g : Graph.graph) id =
  List.exists
    (fun (n : Graph.node) -> List.mem id (Op.operands n.Graph.Node.op))
    g.Graph.Graph.nodes

let bind_constants (g : Graph.graph) constants env =
  Err.List.fold_left
    (fun env id ->
      match Graph.input_kind g id with
      | Graph_ir.Input.Constant when not (constant_is_used g id) ->
          Err.return env
      | Graph_ir.Input.Constant -> (
          match List.assoc_opt id constants with
          | None -> Err.fail (`Missing_constant id)
          | Some tensor -> Err.return (Tensor_id.Map.add id tensor env))
      | Graph_ir.Input.Input -> Err.return env)
    env g.Graph.Graph.inputs

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
        g.Graph.Graph.tensors Tensor_id.Set.empty,
      0 )
    [
      Region_computation.Rms_weight;
      Region_computation.Layer_weight;
      Region_computation.Layer_bias;
      Region_computation.Sdpa_mask;
    ]
  |> fun (ids, _, _) -> ids

let region_result ~limits ~region_counters g ~op ~output ~out_shape ~operand_env
    ~synthetic_ids =
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
    Region_computation4.program ~limits ~op ~output
      ~output_shape:(Shape4.to_vec6 out_shape)
      ~operand:(fun id -> Tensor_id.Map.find_opt id g.Graph.Graph.tensors)
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
  let* lowered =
    Region_execution.lower ~max_size:limits.Kernel.Limits.max_size
      ~max_depth:limits.Kernel.Limits.max_depth
      ~max_local_slots:limits.Kernel.Limits.max_local_slots
      ~scan_limits:(Kernel.Limits.scan_limits limits)
      ~output_shape:(Shape4.to_vec6 out_shape) program
    |> Err.map_error (fun error ->
        `Region_construction (Region_computation.Invalid_program error))
  in
  match lowered with
  | Region_execution.Pixel_loop _ -> assert false
  | Region_execution.Region_loop lowered ->
      Region_execution.materialize ?counters:region_counters lowered ~env
      |> Err.map_error (fun error -> `Region_execution error)

(* The multi-output counterpart of [region_result] (project step 19), the
   twin of [Eval_direct.region_group_result]: builds the shared
   [Region_group.t] ONCE for the whole node and materializes every ordinal in
   [outs] from one shared per-key evaluation. No [Shape4] conversion is
   needed here beyond this function's own boundary -- [Region_group.Emitter.t]
   values are already [Vec6.shape]s, produced by [Lstm.Lstm.Computation.group]
   the same way regardless of caller (Native4D's [Lstm] payload is reused
   verbatim, per the Native4D design record). No synthetic [fill] machinery
   either, for the same reason as the Native twin: [Lstm] has no optional
   operand with a synthetic default. *)
let region_group_result ~limits ~region_counters (g : Graph.graph) ~op ~outs
    ~operand_env =
  let open Err.Syntax in
  let* group =
    Region_computation4.group ~limits ~op ~operand:(fun id ->
        Tensor_id.Map.find_opt id g.Graph.Graph.tensors)
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
  Region_execution.materialize_group ?counters:region_counters lowered_group
    ~env
    ~selected:(List.map (fun (output, _, _) -> output) outs)
  |> Err.map_error (fun error -> `Region_execution error)

let eval_node ?region_counters ~limits ~synthetic_ids (g : Graph.graph) env
    (node : Graph.node) =
  let open Err.Syntax in
  let op = node.Graph.Node.op in
  let fill v shape = Tensor.materialize shape (fun _ -> v) in
  let* shapes =
    widen
      (Graph_shape4.output_shape op ~sig_of:(fun r ->
           Tensor_id.Map.find_opt r g.Graph.Graph.tensors
           |> Err.of_option (`Missing_tensor_sig r)))
  in
  let* operand_env =
    Err.List.fold_left
      (fun acc r ->
        let+ t = find_tensor env r ~context:Operand in
        Tensor_id.Map.add r t acc)
      Tensor_id.Map.empty (Op.operands op)
  in
  let* shape_env =
    Err.List.fold_left
      (fun acc r ->
        let+ sh = sig_shape g r in
        Tensor_id.Map.add r sh acc)
      Tensor_id.Map.empty (Op.operands op)
  in
  let* pairs =
    Err.List.map2
      ~unequal_lengths:(fun actual expected ->
        `Output_arity_mismatch { expected; actual })
      (fun oid out_shape -> Err.return (oid, out_shape))
      node.Graph.Node.outputs shapes
  in
  let outs =
    List.mapi (fun output (oid, out_shape) -> (output, oid, out_shape)) pairs
  in
  (* See Eval_direct.eval_node's identical branch for the full rationale:
     a multi-output region-authored node (today, only [Lstm]) shares one
     recurrence across all its outputs instead of folding [region_result]
     once per ordinal. *)
  match outs with
  | _ :: _ :: _ when Region_computation4.is_region_authored op ->
      let* results =
        region_group_result ~limits
          ~region_counters:
            (let _, first_oid, _ = List.hd outs in
             Option.bind region_counters (fun counters ->
                 Tensor_id.Map.find_opt first_oid counters))
          g ~op ~outs ~operand_env
      in
      Err.return
        (List.fold_left
           (fun env (output, oid, _) ->
             Tensor_id.Map.add oid (List.assoc output results) env)
           env outs)
  | _ ->
      Err.List.fold_left
        (fun env (output, oid, out_shape) ->
          let* result =
            match op with
            | Op.Unbind { Ops4.Unbind.params; x } ->
                Err.return
                  (Tensor.unbind
                     (Tensor_id.Map.find x operand_env)
                     ~axis:(Axis4.to_axis params.axis)
                     ~output ~shape:(Shape4.to_vec6 out_shape))
            | Op.Zeros4 { Ops4.Zeros4.params } ->
                Err.return
                  (Tensor.materialize_fmt params.fmt (Shape4.to_vec6 out_shape)
                     (fun _ -> 0.))
            | Op.Arange4 { Ops4.Arange4.params } -> (
                let params =
                  Factory.Arange.
                    {
                      start = params.start;
                      stop = params.stop;
                      step = params.step;
                      fmt = params.fmt;
                      exact = params.exact;
                    }
                in
                match params.fmt with
                | Payload.Fmt Payload.I64 -> (
                    match params.exact with
                    | Some e ->
                        (* Exact int64 arithmetic, no float round trip -- the
                           Native4D twin of [Eval_direct]'s own D02 fix. *)
                        Err.Escape.with_escape (fun esc ->
                            Tensor.materialize_i64 (Shape4.to_vec6 out_shape)
                              (fun coord ->
                                Err.Escape.or_throw esc
                                  (Factory.Arange.value_i64_exact e
                                     (Dim.to_int coord.Vec6.c))))
                    | None ->
                        Err.return
                          (Tensor.materialize_i64 (Shape4.to_vec6 out_shape)
                             (fun coord ->
                               Int64.of_float
                                 (Factory.Arange.value params
                                    (Dim.to_int coord.Vec6.c)))))
                | _ ->
                    Err.return
                      (Tensor.materialize_fmt params.fmt
                         (Shape4.to_vec6 out_shape) (fun coord ->
                           Factory.Arange.value params (Dim.to_int coord.Vec6.c)))
                )
            | Op.Eye4 { Ops4.Eye4.params } ->
                Err.return
                  (Tensor.materialize_fmt params.fmt (Shape4.to_vec6 out_shape)
                     (fun coord ->
                       if Dim.to_int coord.Vec6.w = Dim.to_int coord.Vec6.c then
                         1.
                       else 0.))
            | _ when Region_computation4.is_region_authored op ->
                region_result ~limits
                  ~region_counters:
                    (Option.bind region_counters (fun counters ->
                         Tensor_id.Map.find_opt oid counters))
                  g ~op ~output ~out_shape ~operand_env ~synthetic_ids
            | _ ->
                Err.return
                  (Schedule.evaluate (Shape4.to_vec6 out_shape)
                     (E.pixel op ~output
                        ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                        ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                        ~fill))
          in
          Err.return (Tensor_id.Map.add oid result env))
        env outs

let run ?region_counters ?(limits = Kernel.Limits.default) ?(constants = [])
    (g : Graph.graph) ~(inputs : (Tensor_id.t * Tensor.packed) list) =
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
  let* env = bind_constants g constants env0 in
  let synthetic_ids = fresh_synthetic_ids g in
  Err.List.fold_left
    (eval_node ?region_counters ~limits ~synthetic_ids g)
    env g.Graph.Graph.nodes
