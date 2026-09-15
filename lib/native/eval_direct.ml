(* One global SSA environment; structural groups do not affect evaluation. *)

open Graph_ir
module E = Eval_op.Make (Direct)

type context = Operand | Sig_shape
type missing_tensor = { context : context; id : Tensor_id.t }
type arity_mismatch = { expected : int; actual : int }

type error =
  [ Graph_shape.error
  | `Arange_i64_overflow of Factory.Arange.Overflow.t
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Missing_tensor of missing_tensor
  | `Output_arity_mismatch of arity_mismatch
  | `Region_construction of Region_computation.error
  | `Region_execution of Region_eval.error ]

type hooks =
  | Hooks : { on_start : node -> 'a; on_end : node -> 'a -> unit } -> hooks

let pp_context ppf = function
  | Operand -> Format.pp_print_string ppf "operand"
  | Sig_shape -> Format.pp_print_string ppf "shape lookup"

let pp_error ppf : [< error ] -> unit = function
  | #Graph_shape.error as e -> Graph_shape.pp_error ppf e
  | `Arange_i64_overflow { Factory.Arange.Overflow.start; step; i } ->
      Format.fprintf ppf
        "arange: exact int64 generation overflows at start=%Ld step=%Ld i=%d"
        start step i
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
      Region_execution.materialize ?counters:region_counters lowered ~env
      |> Err.map_error (fun error -> `Region_execution error)

(* The multi-output counterpart of [region_result] (project step 19): builds
   the shared [Region_group.t] ONCE for the whole node and materializes every
   ordinal in [outs] from one shared per-key evaluation, instead of folding
   [region_result] (which would rebuild the whole recurrence) once per
   ordinal. No synthetic [fill]/[synthetic_ids] machinery is needed here --
   [Region_computation.group] only ever builds [Lstm], which (unlike
   [Rms_norm]/[Layer_norm]/[Sdpa]) has no optional operand with a synthetic
   default. *)
let region_group_result ~limits ~region_counters g ~op ~outs ~operand_env =
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
  Region_execution.materialize_group ?counters:region_counters lowered_group
    ~env
    ~selected:(List.map (fun (output, _, _) -> output) outs)
  |> Err.map_error (fun error -> `Region_execution error)

let rec run_graph ?hooks ?region_counters ?(limits = Kernel.Limits.default)
    ~constants (g : graph) (env : Tensor.packed Tensor_id.Map.t) :
    (Tensor.packed Tensor_id.Map.t, error) Err.t =
  let open Err.Syntax in
  let* env = bind_constants g constants env in
  let synthetic_ids = fresh_synthetic_ids g in
  Err.List.fold_left
    (fun env node ->
      match hooks with
      | None -> eval_node ?region_counters ~limits ~synthetic_ids g env node
      | Some (Hooks h) ->
          let state = h.on_start node in
          let* env =
            eval_node ?region_counters ~limits ~synthetic_ids g env node
          in
          h.on_end node state;
          Err.return env)
    env g.Graph.nodes

and eval_node ?region_counters ~limits ~synthetic_ids (g : graph)
    (env : Tensor.packed Tensor_id.Map.t) (node : node) :
    (Tensor.packed Tensor_id.Map.t, error) Err.t =
  let open Err.Syntax in
  let op = node.Node.op in
  let operand r = find_tensor env r ~context:Operand in
  let shape_of r = sig_shape g r in
  let fill v shape = Tensor.materialize shape (fun _ -> v) in
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
  let outs =
    List.mapi (fun output (oid, out_shape) -> (output, oid, out_shape)) pairs
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
            | Unbind { Split.Unbind.params; x } ->
                Err.return
                  (Tensor.unbind
                     (Tensor_id.Map.find x operand_env)
                     ~axis:params.axis ~output ~shape:out_shape)
            (* Same dtype-preserving bypass as [Unbind], and for the same reason:
           [offset] is the sum of every earlier piece's size, computed the
           same way [Eval_op]'s arm computes it for the generic path. *)
            | Split_with_sizes { Split.Split_with_sizes.params; x } ->
                let offset =
                  Split.Split_with_sizes.offset_of ~output
                    params.Split.Split_with_sizes.sizes
                in
                Err.return
                  (Tensor.split_with_sizes
                     (Tensor_id.Map.find x operand_env)
                     ~axis:params.Split.Split_with_sizes.axis ~offset
                     ~shape:out_shape)
            | Zeros { Factory.Zeros.params } ->
                Err.return
                  (Tensor.materialize_fmt params.fmt out_shape (fun _ -> 0.))
            | Eye { Factory.Eye.params } ->
                Err.return
                  (Tensor.materialize_fmt params.fmt out_shape (fun coord ->
                       if Dim.to_int coord.Vec6.w = Dim.to_int coord.Vec6.c then
                         1.
                       else 0.))
            | Arange { Factory.Arange.params } -> (
                match params.fmt with
                | Payload.Fmt Payload.I64 -> (
                    match params.exact with
                    | Some e ->
                        (* Exact int64 arithmetic, no float round trip: fixes
                           the truncation [Int64.of_float (Arange.value ...)]
                           below performs whenever an ATen call actually
                           supplied exact integer bounds (see the
                           implementation tracker's D02). *)
                        Err.Escape.with_escape (fun esc ->
                            Tensor.materialize_i64 out_shape (fun coord ->
                                Err.Escape.or_throw esc
                                  (Factory.Arange.value_i64_exact e
                                     (Dim.to_int coord.Vec6.c))))
                    | None ->
                        Err.return
                          (Tensor.materialize_i64 out_shape (fun coord ->
                               Int64.of_float
                                 (Factory.Arange.value params
                                    (Dim.to_int coord.Vec6.c)))))
                | _ ->
                    Err.return
                      (Tensor.materialize_fmt params.fmt out_shape (fun coord ->
                           Factory.Arange.value params (Dim.to_int coord.Vec6.c)))
                )
            (* Dtype-preserving Reshape: the default arm below reaches
               [Reshape.Reshape.Compute(Direct).pixel], whose final [S.load]
               reads through [Payload.get_float] regardless of the source
               format -- exact for F32 but silently lossy above 2^53 for an
               I64 source. Branching on the OPERAND's declared signature
               format (not the runtime payload, though they agree by
               construction) routes an I64 reshape through
               [Compute_i64]/[Tensor.i64_load] instead, matching this node's
               Arange arm just above. Every other format keeps the existing
               float pixel path, unchanged. *)
            | Reshape { Reshape.Reshape.params; x } -> (
                let x_sig = Tensor_id.Map.find x g.Graph.tensors in
                match x_sig.Tensor_sig.fmt with
                | Payload.Fmt Payload.I64 ->
                    let module C = Reshape.Reshape.Compute_i64 (Direct) (Direct)
                    in
                    let x_t = Tensor_id.Map.find x operand_env in
                    let x_shape = Tensor_id.Map.find x shape_env in
                    Err.return
                      (Tensor.materialize_i64 out_shape (fun coord ->
                           C.pixel params ~x_shape ~x:x_t coord))
                | _ ->
                    Err.return
                      (Schedule.evaluate out_shape
                         (E.pixel op ~output
                            ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                            ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                            ~fill)))
            (* Dtype-preserving Permute, the same shape as Reshape just above:
               the default arm's [Permute.Compute(S).pixel] reads through
               [S.load], exact for F32 but silently lossy above 2^53 for an
               I64 source. Branch on the OPERAND's declared format, matching
               Reshape/Arange's own precedent. *)
            | Permute { Permute.Permute.perm; x } -> (
                let x_sig = Tensor_id.Map.find x g.Graph.tensors in
                match x_sig.Tensor_sig.fmt with
                | Payload.Fmt Payload.I64 ->
                    let module C = Permute.Permute.Compute_i64 (Direct) (Direct)
                    in
                    let x_t = Tensor_id.Map.find x operand_env in
                    Err.return
                      (Tensor.materialize_i64 out_shape (fun coord ->
                           C.pixel perm ~x:x_t coord))
                | _ ->
                    Err.return
                      (Schedule.evaluate out_shape
                         (E.pixel op ~output
                            ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                            ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                            ~fill)))
            (* Explicit int64-input promotion for [Mul_scalar]: the default
               arm's [Pointwise.Mul_scalar.Compute(S).pixel] reads through
               [S.load], which is numerically exact for this promotion
               ([Payload.get_float]'s I64 case is [Int64.to_float]) but
               incidental -- branch on the operand's declared format so the
               cast is the explicit [i64_to_float] step the plan requires,
               matching Reshape/Permute's own precedent. Unlike those two,
               the output stays the ordinary float pixel path: [Mul_scalar]'s
               output format is F32 regardless of operand format, so only the
               read changes, not the write-back. *)
            | Mul_scalar { Pointwise.Scalar_bin.x; scalar } -> (
                let x_sig = Tensor_id.Map.find x g.Graph.tensors in
                match x_sig.Tensor_sig.fmt with
                | Payload.Fmt Payload.I64 ->
                    let module C =
                      Pointwise.Mul_scalar.Compute_i64 (Direct) (Direct)
                    in
                    let x_t = Tensor_id.Map.find x operand_env in
                    Err.return
                      (Schedule.evaluate out_shape (fun coord ->
                           C.pixel ~scalar x_t coord))
                | _ ->
                    Err.return
                      (Schedule.evaluate out_shape
                         (E.pixel op ~output
                            ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                            ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                            ~fill)))
            (* Dtype-preserving tensor-tensor Add/Sub/Mul: the default arm's
               [Pointwise.{Add,Sub,Mul}.Compute(S).pixel] reads both operands
               through [S.load], exact for F32 but silently lossy above 2^53
               for I64 operands, same defect class as Reshape/Permute before
               their own fixes. Dispatch only when BOTH operands declare I64
               (the only case [Graph_builder.{add,sub,mul}] threads an I64
               output edge for, and the only case a real broadcasted binary op
               is safe to promote wholesale to -- a mismatched pair is
               unsupported mixed promotion, left to the ordinary float path
               unchanged, per the plan's P5.4). *)
            | Add { Pointwise.Bin.a; b } -> (
                let a_sig = Tensor_id.Map.find a g.Graph.tensors in
                let b_sig = Tensor_id.Map.find b g.Graph.tensors in
                match (a_sig.Tensor_sig.fmt, b_sig.Tensor_sig.fmt) with
                | Payload.(Fmt I64, Fmt I64) ->
                    let module C = Pointwise.Add.Compute_i64 (Direct) (Direct)
                    in
                    let a_t = Tensor_id.Map.find a operand_env in
                    let b_t = Tensor_id.Map.find b operand_env in
                    let a_shape = Tensor_id.Map.find a shape_env in
                    let b_shape = Tensor_id.Map.find b shape_env in
                    Err.return
                      (Tensor.materialize_i64 out_shape (fun coord ->
                           C.pixel ~a_shape ~b_shape a_t b_t coord))
                | _ ->
                    Err.return
                      (Schedule.evaluate out_shape
                         (E.pixel op ~output
                            ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                            ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                            ~fill)))
            | Sub { Pointwise.Bin.a; b } -> (
                let a_sig = Tensor_id.Map.find a g.Graph.tensors in
                let b_sig = Tensor_id.Map.find b g.Graph.tensors in
                match (a_sig.Tensor_sig.fmt, b_sig.Tensor_sig.fmt) with
                | Payload.(Fmt I64, Fmt I64) ->
                    let module C = Pointwise.Sub.Compute_i64 (Direct) (Direct)
                    in
                    let a_t = Tensor_id.Map.find a operand_env in
                    let b_t = Tensor_id.Map.find b operand_env in
                    let a_shape = Tensor_id.Map.find a shape_env in
                    let b_shape = Tensor_id.Map.find b shape_env in
                    Err.return
                      (Tensor.materialize_i64 out_shape (fun coord ->
                           C.pixel ~a_shape ~b_shape a_t b_t coord))
                | _ ->
                    Err.return
                      (Schedule.evaluate out_shape
                         (E.pixel op ~output
                            ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                            ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                            ~fill)))
            | Mul { Pointwise.Bin.a; b } -> (
                let a_sig = Tensor_id.Map.find a g.Graph.tensors in
                let b_sig = Tensor_id.Map.find b g.Graph.tensors in
                match (a_sig.Tensor_sig.fmt, b_sig.Tensor_sig.fmt) with
                | Payload.(Fmt I64, Fmt I64) ->
                    let module C = Pointwise.Mul.Compute_i64 (Direct) (Direct)
                    in
                    let a_t = Tensor_id.Map.find a operand_env in
                    let b_t = Tensor_id.Map.find b operand_env in
                    let a_shape = Tensor_id.Map.find a shape_env in
                    let b_shape = Tensor_id.Map.find b shape_env in
                    Err.return
                      (Tensor.materialize_i64 out_shape (fun coord ->
                           C.pixel ~a_shape ~b_shape a_t b_t coord))
                | _ ->
                    Err.return
                      (Schedule.evaluate out_shape
                         (E.pixel op ~output
                            ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                            ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                            ~fill)))
            (* Explicit int64-input promotion for [To_copy]'s [Float] target
               only -- the EdgeNeXt/mvitv2 "I64 Arange -> Float cast"
               acceptance pattern's own promoted-consumer step. Same rationale
               as [Mul_scalar] above: [Compute(S).pixel]'s [S.load] already
               computes the identical value for this specific case
               ([Payload.get_float]'s I64 case is [Int64.to_float]), so this
               is architecture-only, not a value-level fix. [Long]/[Bool]
               targets are untouched -- an I64 input reaching [Long] needs no
               cast at all (I64->I64 copy), and [Bool] needs storage this plan
               has not opened yet, so both keep the existing float pixel path. *)
            | To_copy { Pointwise.To_copy.target = Pointwise.To_copy.Float; x }
              -> (
                let x_sig = Tensor_id.Map.find x g.Graph.tensors in
                match x_sig.Tensor_sig.fmt with
                | Payload.Fmt Payload.I64 ->
                    let module C =
                      Pointwise.To_copy.Compute_i64 (Direct) (Direct)
                    in
                    let x_t = Tensor_id.Map.find x operand_env in
                    Err.return
                      (Schedule.evaluate out_shape (fun coord ->
                           C.pixel x_t coord))
                | _ ->
                    Err.return
                      (Schedule.evaluate out_shape
                         (E.pixel op ~output
                            ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                            ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                            ~fill)))
            | _ when Region_computation.is_region_authored op ->
                region_result ~limits
                  ~region_counters:
                    (Option.bind region_counters (fun counters ->
                         Tensor_id.Map.find_opt oid counters))
                  g ~op ~output ~out_shape ~operand_env ~synthetic_ids
            | _ ->
                Err.return
                  (Schedule.evaluate out_shape
                     (E.pixel op ~output
                        ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                        ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                        ~fill))
          in
          Err.return (Tensor_id.Map.add oid result env))
        env outs

let run ?hooks ?region_counters ?(limits = Kernel.Limits.default)
    ?(constants = []) (g : graph) ~(inputs : (Tensor_id.t * Tensor.packed) list)
    =
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
  run_graph ?hooks ?region_counters ~limits ~constants g env0
