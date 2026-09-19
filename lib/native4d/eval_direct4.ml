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

type mixed_dtype = {
  mixed_op : string;
  a_fmt : Payload.packed_fmt;
  b_fmt : Payload.packed_fmt;
}

type scalar_op = { scalar_op : string; fmt : Payload.packed_fmt }

type error =
  [ Graph_shape4.error
  | `Arange_i64_overflow of Factory.Arange.Overflow.t
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Missing_tensor of missing_tensor
  | `Output_arity_mismatch of arity_mismatch
  | `Region_construction of Region_computation4.error
  | `Region_execution of Region_eval.error
  | `Unsupported_bool_arithmetic of mixed_dtype
  | `Unsupported_bool_scalar_arithmetic of scalar_op
  | `Unsupported_mixed_dtype of mixed_dtype
  | `Unsupported_to_copy_bool_source of Payload.packed_fmt
  | `Unsupported_to_copy_long_source of Payload.packed_fmt ]

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
  | `Unsupported_bool_arithmetic
      { mixed_op; a_fmt = Payload.Fmt a_fmt; b_fmt = Payload.Fmt b_fmt } ->
      Fmt.pf ppf "%s: arithmetic on a Bool operand is not supported, a=%s b=%s"
        mixed_op (Payload.fmt_name a_fmt) (Payload.fmt_name b_fmt)
  | `Unsupported_bool_scalar_arithmetic { scalar_op; fmt = Payload.Fmt fmt } ->
      Fmt.pf ppf "%s: arithmetic on a Bool operand is not supported, x=%s"
        scalar_op (Payload.fmt_name fmt)
  | `Unsupported_mixed_dtype
      { mixed_op; a_fmt = Payload.Fmt a_fmt; b_fmt = Payload.Fmt b_fmt } ->
      Fmt.pf ppf "%s: unsupported mixed dtype, a=%s b=%s" mixed_op
        (Payload.fmt_name a_fmt) (Payload.fmt_name b_fmt)
  | `Unsupported_to_copy_bool_source (Payload.Fmt f) ->
      Fmt.pf ppf "to_copy: Bool target has no exact Bool output for a %s source"
        (Payload.fmt_name f)
  | `Unsupported_to_copy_long_source (Payload.Fmt f) ->
      Fmt.pf ppf "to_copy: Long target has no exact I64 output for a %s source"
        (Payload.fmt_name f)

let is_bool = function Payload.Fmt Payload.Bool -> true | _ -> false

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

(* Edges something reads: a graph output or any node's operand. An index output
   nothing reads is not worth allocating. *)
let live_edges (g : Graph.graph) =
  let add = List.fold_left (fun s id -> Tensor_id.Set.add id s) in
  List.fold_left
    (fun live (node : Graph.node) -> add live (Op.operands node.Graph.Node.op))
    (add Tensor_id.Set.empty g.Graph.Graph.outputs)
    g.Graph.Graph.nodes

(* The second output of the argmax-style ops. *)
let is_index_output (op : Op.t) output =
  output = 1
  &&
  match op with
  | Op.Adaptive_max_pool2d_with_indices _ | Op.Max_pool2d_with_indices _ -> true
  | _ -> false

let eval_node ?region_counters ~limits ~synthetic_ids ~live (g : Graph.graph)
    env (node : Graph.node) =
  let open Err.Syntax in
  let op = node.Graph.Node.op in
  let fmt_of r = (Tensor_id.Map.find r g.Graph.Graph.tensors).Tensor_sig.fmt in
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
  (* A dead index output is neither computed nor allocated: nothing reads it,
     so [env] never needs it. *)
  let outs =
    List.mapi (fun output (oid, out_shape) -> (output, oid, out_shape)) pairs
    |> List.filter (fun (output, oid, _) ->
        not (is_index_output op output && not (Tensor_id.Set.mem oid live)))
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
            (* Dtype-preserving Reshape4/Permute4, the Native4D twin of
               [Eval_direct]'s own P5.2 arms: the default arm's
               [Eval_op4.pixel]'s delegation to Native's
               [Reshape.Reshape.Compute(S).pixel]/[Permute.Permute.Compute(S)
               .pixel] reads through [S.load], which round-trips every format
               through [Payload.get_float] and is silently lossy for an I64
               source above 2^53. Branch on the OPERAND's declared format
               (paired with [Builder.reshape4]/[permute4]'s own I64-only fmt
               threading), routing an I64 source through Native's own
               [Compute_i64] functors instead. *)
            | Op.Reshape4 { Ops4.Reshape4.params; x } -> (
                let x_sig = Tensor_id.Map.find x g.Graph.Graph.tensors in
                match x_sig.Tensor_sig.fmt with
                | Payload.Fmt Payload.I64 ->
                    let module C = Reshape.Reshape.Compute_i64 (Direct) (Direct)
                    in
                    let x_t = Tensor_id.Map.find x operand_env in
                    let x_shape = Tensor_id.Map.find x shape_env in
                    Err.return
                      (Tensor.materialize_i64 (Shape4.to_vec6 out_shape)
                         (fun coord ->
                           C.pixel
                             {
                               Reshape.Reshape.shape =
                                 Shape4.to_vec6 params.shape;
                             }
                             ~x_shape ~x:x_t coord))
                | _ ->
                    Err.return
                      (Schedule.evaluate (Shape4.to_vec6 out_shape)
                         (E.pixel op ~output
                            ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                            ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                            ~fill)))
            | Op.Permute4 { Ops4.Permute4.perm; x } -> (
                let x_sig = Tensor_id.Map.find x g.Graph.Graph.tensors in
                match x_sig.Tensor_sig.fmt with
                | Payload.Fmt Payload.I64 ->
                    let module C = Permute.Permute.Compute_i64 (Direct) (Direct)
                    in
                    let x_t = Tensor_id.Map.find x operand_env in
                    Err.return
                      (Tensor.materialize_i64 (Shape4.to_vec6 out_shape)
                         (fun coord ->
                           C.pixel (Graph_shape4.perm6 perm) ~x:x_t coord))
                | _ ->
                    Err.return
                      (Schedule.evaluate (Shape4.to_vec6 out_shape)
                         (E.pixel op ~output
                            ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                            ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                            ~fill)))
            (* Dtype-preserving tensor-tensor Add/Sub/Mul and the explicit
               int64-input promotion for Mul_scalar -- the Native4D twins of
               [Eval_direct]'s own P5.3/P5.4 arms, same rationale: dispatch
               only when BOTH operands declare I64 (the only case
               [Builder.{add,sub,mul}] threads an I64 output edge for); an
               exactly-one-I64 pair fails at checked admission rather than
               silently computing through a double-rounded float path. *)
            | Op.Add { Pointwise.Bin.a; b } -> (
                let a_sig = Tensor_id.Map.find a g.Graph.Graph.tensors in
                let b_sig = Tensor_id.Map.find b g.Graph.Graph.tensors in
                match (a_sig.Tensor_sig.fmt, b_sig.Tensor_sig.fmt) with
                | Payload.(Fmt I64, Fmt I64) ->
                    let module C = Pointwise.Add.Compute_i64 (Direct) (Direct)
                    in
                    let a_t = Tensor_id.Map.find a operand_env in
                    let b_t = Tensor_id.Map.find b operand_env in
                    let a_shape = Tensor_id.Map.find a shape_env in
                    let b_shape = Tensor_id.Map.find b shape_env in
                    Err.return
                      (Tensor.materialize_i64 (Shape4.to_vec6 out_shape)
                         (fun coord -> C.pixel ~a_shape ~b_shape a_t b_t coord))
                (* Arithmetic on Bool stays rejected -- the Native4D twin of
                   [Eval_direct]'s own fix, checked BEFORE the I64 guard below
                   so a Bool paired with I64 reports the Bool reason. *)
                | a_fmt, b_fmt when is_bool a_fmt || is_bool b_fmt ->
                    Err.fail
                      (`Unsupported_bool_arithmetic
                         { mixed_op = "add"; a_fmt; b_fmt })
                | a_fmt, b_fmt
                  when (match a_fmt with
                         | Payload.Fmt Payload.I64 -> true
                         | _ -> false)
                       ||
                       match b_fmt with
                       | Payload.Fmt Payload.I64 -> true
                       | _ -> false ->
                    Err.fail
                      (`Unsupported_mixed_dtype
                         { mixed_op = "add"; a_fmt; b_fmt })
                | _ ->
                    Err.return
                      (Schedule.evaluate (Shape4.to_vec6 out_shape)
                         (E.pixel op ~output
                            ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                            ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                            ~fill)))
            | Op.Sub { Pointwise.Bin.a; b } -> (
                let a_sig = Tensor_id.Map.find a g.Graph.Graph.tensors in
                let b_sig = Tensor_id.Map.find b g.Graph.Graph.tensors in
                match (a_sig.Tensor_sig.fmt, b_sig.Tensor_sig.fmt) with
                | Payload.(Fmt I64, Fmt I64) ->
                    let module C = Pointwise.Sub.Compute_i64 (Direct) (Direct)
                    in
                    let a_t = Tensor_id.Map.find a operand_env in
                    let b_t = Tensor_id.Map.find b operand_env in
                    let a_shape = Tensor_id.Map.find a shape_env in
                    let b_shape = Tensor_id.Map.find b shape_env in
                    Err.return
                      (Tensor.materialize_i64 (Shape4.to_vec6 out_shape)
                         (fun coord -> C.pixel ~a_shape ~b_shape a_t b_t coord))
                (* See the matching [Add] arm's own comment. *)
                | a_fmt, b_fmt when is_bool a_fmt || is_bool b_fmt ->
                    Err.fail
                      (`Unsupported_bool_arithmetic
                         { mixed_op = "sub"; a_fmt; b_fmt })
                | a_fmt, b_fmt
                  when (match a_fmt with
                         | Payload.Fmt Payload.I64 -> true
                         | _ -> false)
                       ||
                       match b_fmt with
                       | Payload.Fmt Payload.I64 -> true
                       | _ -> false ->
                    Err.fail
                      (`Unsupported_mixed_dtype
                         { mixed_op = "sub"; a_fmt; b_fmt })
                | _ ->
                    Err.return
                      (Schedule.evaluate (Shape4.to_vec6 out_shape)
                         (E.pixel op ~output
                            ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                            ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                            ~fill)))
            | Op.Mul { Pointwise.Bin.a; b } -> (
                let a_sig = Tensor_id.Map.find a g.Graph.Graph.tensors in
                let b_sig = Tensor_id.Map.find b g.Graph.Graph.tensors in
                match (a_sig.Tensor_sig.fmt, b_sig.Tensor_sig.fmt) with
                | Payload.(Fmt I64, Fmt I64) ->
                    let module C = Pointwise.Mul.Compute_i64 (Direct) (Direct)
                    in
                    let a_t = Tensor_id.Map.find a operand_env in
                    let b_t = Tensor_id.Map.find b operand_env in
                    let a_shape = Tensor_id.Map.find a shape_env in
                    let b_shape = Tensor_id.Map.find b shape_env in
                    Err.return
                      (Tensor.materialize_i64 (Shape4.to_vec6 out_shape)
                         (fun coord -> C.pixel ~a_shape ~b_shape a_t b_t coord))
                (* See the matching [Add] arm's own comment. *)
                | a_fmt, b_fmt when is_bool a_fmt || is_bool b_fmt ->
                    Err.fail
                      (`Unsupported_bool_arithmetic
                         { mixed_op = "mul"; a_fmt; b_fmt })
                | a_fmt, b_fmt
                  when (match a_fmt with
                         | Payload.Fmt Payload.I64 -> true
                         | _ -> false)
                       ||
                       match b_fmt with
                       | Payload.Fmt Payload.I64 -> true
                       | _ -> false ->
                    Err.fail
                      (`Unsupported_mixed_dtype
                         { mixed_op = "mul"; a_fmt; b_fmt })
                | _ ->
                    Err.return
                      (Schedule.evaluate (Shape4.to_vec6 out_shape)
                         (E.pixel op ~output
                            ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                            ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                            ~fill)))
            | Op.Mul_scalar { Pointwise.Scalar_bin.x; scalar } -> (
                let x_sig = Tensor_id.Map.find x g.Graph.Graph.tensors in
                match x_sig.Tensor_sig.fmt with
                | Payload.Fmt Payload.I64 ->
                    let module C =
                      Pointwise.Mul_scalar.Compute_i64 (Direct) (Direct)
                    in
                    let x_t = Tensor_id.Map.find x operand_env in
                    Err.return
                      (Schedule.evaluate (Shape4.to_vec6 out_shape)
                         (fun coord -> C.pixel ~scalar x_t coord))
                (* Arithmetic on Bool stays rejected here too -- the
                   Native4D twin of [Eval_direct]'s own [Mul_scalar] fix. *)
                | fmt when is_bool fmt ->
                    Err.fail
                      (`Unsupported_bool_scalar_arithmetic
                         { scalar_op = "mul_scalar"; fmt })
                | _ ->
                    Err.return
                      (Schedule.evaluate (Shape4.to_vec6 out_shape)
                         (E.pixel op ~output
                            ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                            ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                            ~fill)))
            (* Explicit int64-input promotion for [To_copy]'s [Float] target,
               and the reverse "Float to I64" cast for its [Long] target --
               the Native4D twins of [Eval_direct]'s own P5.3 arms. [Long]/
               [Float]/[Bool] are each their own arm rather than one match on
               [target] because the three directions need entirely different
               dispatch shapes (a read-side cast vs. a genuine checked
               write). *)
            | Op.To_copy
                { Pointwise.To_copy.target = Pointwise.To_copy.Float; x } -> (
                let x_sig = Tensor_id.Map.find x g.Graph.Graph.tensors in
                match x_sig.Tensor_sig.fmt with
                | Payload.Fmt Payload.I64 ->
                    let module C =
                      Pointwise.To_copy.Compute_i64 (Direct) (Direct)
                    in
                    let x_t = Tensor_id.Map.find x operand_env in
                    Err.return
                      (Schedule.evaluate (Shape4.to_vec6 out_shape)
                         (fun coord -> C.pixel x_t coord))
                | _ ->
                    Err.return
                      (Schedule.evaluate (Shape4.to_vec6 out_shape)
                         (E.pixel op ~output
                            ~operand:(fun r -> Tensor_id.Map.find r operand_env)
                            ~shape_of:(fun r -> Tensor_id.Map.find r shape_env)
                            ~fill)))
            | Op.To_copy
                { Pointwise.To_copy.target = Pointwise.To_copy.Long; x } -> (
                let x_sig = Tensor_id.Map.find x g.Graph.Graph.tensors in
                match x_sig.Tensor_sig.fmt with
                | Payload.Fmt Payload.F32 ->
                    let module C =
                      Pointwise.To_copy.Compute_to_long (Direct) (Direct)
                    in
                    let x_t = Tensor_id.Map.find x operand_env in
                    Err.return
                      (Tensor.materialize_i64 (Shape4.to_vec6 out_shape)
                         (fun coord -> C.pixel x_t coord))
                | Payload.Fmt Payload.I64 ->
                    let x_t = Tensor_id.Map.find x operand_env in
                    Err.return
                      (Tensor.materialize_i64 (Shape4.to_vec6 out_shape)
                         (fun coord -> Direct.i64_load x_t coord))
                | Payload.Fmt other ->
                    Err.fail
                      (`Unsupported_to_copy_long_source (Payload.Fmt other)))
            (* [To_copy]'s [Bool] target now writes real [Payload.Bool]
               storage too (P6.3), the Native4D twin of [Eval_direct]'s own
               arm -- [Builder.to_copy]'s [Bool] case now declares the
               output edge [Bool] unconditionally (this session), so
               leaving this to the generic default arm below (F32-only)
               would reproduce the exact declared/actual mismatch hazard
               the [Long] arm above already guards against. Reuses
               [Compute(Direct).pixel]'s own formula UNCHANGED, same as
               Native's own arm. *)
            | Op.To_copy
                { Pointwise.To_copy.target = Pointwise.To_copy.Bool; x } -> (
                let x_sig = Tensor_id.Map.find x g.Graph.Graph.tensors in
                match x_sig.Tensor_sig.fmt with
                | Payload.Fmt Payload.F32 ->
                    let module C = Pointwise.To_copy.Compute (Direct) in
                    let x_t = Tensor_id.Map.find x operand_env in
                    Err.return
                      (Tensor.materialize_bool (Shape4.to_vec6 out_shape)
                         (fun coord ->
                           C.pixel Pointwise.To_copy.Bool x_t coord <> 0.0))
                | Payload.Fmt other ->
                    Err.fail
                      (`Unsupported_to_copy_bool_source (Payload.Fmt other)))
            (* [Bitwise_not] now writes real [Payload.Bool] storage too
               (P6.3), the Native4D twin of [Eval_direct]'s own arm --
               [Builder.bitwise_not] now declares the output edge [Bool]
               unconditionally (this session), matching [To_copy(Bool)]'s
               own reasoning above. No operand-format branch is needed,
               unlike [To_copy]'s casts: [Compute(Direct).pixel]'s existing
               formula already reads ANY operand format. *)
            | Op.Bitwise_not { Pointwise.Bitwise_not.x } ->
                let module C = Pointwise.Bitwise_not.Compute (Direct) in
                let x_t = Tensor_id.Map.find x operand_env in
                Err.return
                  (Tensor.materialize_bool (Shape4.to_vec6 out_shape)
                     (fun coord -> C.pixel x_t coord <> 0.0))
            (* [Eq_scalar] mirrors [Gt_scalar]'s own split exactly (P6.4),
               the Native4D twin of [Eval_direct]'s own arm --
               [Eval_op4.Make(S).pixel]'s [Eq_scalar] case is
               [SEMANTICS]-generic and still writes a plain float 0./1., so
               only this early-intercept arm lands genuine [Payload.Bool]
               storage, matching [Builder.eq_scalar]'s own unconditional
               [Bool] declaration. *)
            | Op.Eq_scalar { Pointwise.Scalar_bin.x; scalar } ->
                let module C = Pointwise.Eq_scalar.Compute (Direct) in
                let x_t = Tensor_id.Map.find x operand_env in
                Err.return
                  (Tensor.materialize_bool (Shape4.to_vec6 out_shape)
                     (fun coord -> C.pixel ~scalar x_t coord <> 0.0))
            (* [Eq_tensor] mirrors [Eq_scalar]'s own split, broadcast via
               [Binary] instead of [Scalar_binary] since both operands are
               runtime tensors, matching [Builder.eq_tensor]'s own
               unconditional [Bool] declaration. *)
            | Op.Eq_tensor { Pointwise.Bin.a; b } ->
                let module C = Pointwise.Eq_tensor.Compute (Direct) in
                let a_t = Tensor_id.Map.find a operand_env in
                let b_t = Tensor_id.Map.find b operand_env in
                let a_shape = Tensor_id.Map.find a shape_env in
                let b_shape = Tensor_id.Map.find b shape_env in
                Err.return
                  (Tensor.materialize_bool (Shape4.to_vec6 out_shape)
                     (fun coord ->
                       C.pixel ~a_shape ~b_shape a_t b_t coord <> 0.0))
            (* [Gt_scalar] mirrors [Bitwise_not]'s own split, the Native4D
               twin of [Eval_direct]'s own arm -- [Eval_op4.Make(S).pixel]'s
               [Gt_scalar] case (added this session) is [SEMANTICS]-generic
               and still writes a plain float 0./1., so only this
               early-intercept arm lands genuine [Payload.Bool] storage,
               matching [Builder.gt_scalar]'s own unconditional [Bool]
               declaration. *)
            | Op.Gt_scalar { Pointwise.Scalar_bin.x; scalar } ->
                let module C = Pointwise.Gt_scalar.Compute (Direct) in
                let x_t = Tensor_id.Map.find x operand_env in
                Err.return
                  (Tensor.materialize_bool (Shape4.to_vec6 out_shape)
                     (fun coord -> C.pixel ~scalar x_t coord <> 0.0))
            (* [Ne_scalar] mirrors [Eq_scalar]'s own split exactly (P6.4,
               negated), the Native4D twin of [Eval_direct]'s own arm --
               [Eval_op4.Make(S).pixel]'s [Ne_scalar] case is
               [SEMANTICS]-generic and still writes a plain float 0./1., so
               only this early-intercept arm lands genuine [Payload.Bool]
               storage, matching [Builder.ne_scalar]'s own unconditional
               [Bool] declaration. *)
            | Op.Ne_scalar { Pointwise.Scalar_bin.x; scalar } ->
                let module C = Pointwise.Ne_scalar.Compute (Direct) in
                let x_t = Tensor_id.Map.find x operand_env in
                Err.return
                  (Tensor.materialize_bool (Shape4.to_vec6 out_shape)
                     (fun coord -> C.pixel ~scalar x_t coord <> 0.0))
            (* [Ne_tensor] mirrors [Eq_tensor]'s own split exactly (negated),
               matching [Builder.ne_tensor]'s own unconditional [Bool]
               declaration. *)
            | Op.Ne_tensor { Pointwise.Bin.a; b } ->
                let module C = Pointwise.Ne_tensor.Compute (Direct) in
                let a_t = Tensor_id.Map.find a operand_env in
                let b_t = Tensor_id.Map.find b operand_env in
                let a_shape = Tensor_id.Map.find a shape_env in
                let b_shape = Tensor_id.Map.find b shape_env in
                Err.return
                  (Tensor.materialize_bool (Shape4.to_vec6 out_shape)
                     (fun coord ->
                       C.pixel ~a_shape ~b_shape a_t b_t coord <> 0.0))
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
            (* The Native4D twin of [Eval_direct]'s own extension of the
               `*_scalar` family's Bool-rejection to the ops with no
               per-format admission point at all -- see that file's own
               comment. None of these seven has ANY Native4D dispatch today
               (confirmed by `grep -n`), so every operand format reaches the
               generic default arm below unchecked; each guard intercepts
               ONLY the Bool case, letting every other format (including
               I64) fall through unchanged. *)
            | Op.Add_scalar { Pointwise.Scalar_bin.x; _ }
              when is_bool (fmt_of x) ->
                Err.fail
                  (`Unsupported_bool_scalar_arithmetic
                     { scalar_op = "add_scalar"; fmt = fmt_of x })
            | Op.Div_scalar { Pointwise.Scalar_bin.x; _ }
              when is_bool (fmt_of x) ->
                Err.fail
                  (`Unsupported_bool_scalar_arithmetic
                     { scalar_op = "div_scalar"; fmt = fmt_of x })
            | Op.Floor_div_scalar { Pointwise.Scalar_bin.x; _ }
              when is_bool (fmt_of x) ->
                Err.fail
                  (`Unsupported_bool_scalar_arithmetic
                     { scalar_op = "floor_div_scalar"; fmt = fmt_of x })
            | Op.Pow { Pointwise.Scalar_bin.x; _ } when is_bool (fmt_of x) ->
                Err.fail
                  (`Unsupported_bool_scalar_arithmetic
                     { scalar_op = "pow"; fmt = fmt_of x })
            | Op.Rpow_scalar { Pointwise.Scalar_bin.x; _ }
              when is_bool (fmt_of x) ->
                Err.fail
                  (`Unsupported_bool_scalar_arithmetic
                     { scalar_op = "rpow_scalar"; fmt = fmt_of x })
            | Op.Rsub_scalar { Pointwise.Rsub_scalar.x; _ }
              when is_bool (fmt_of x) ->
                Err.fail
                  (`Unsupported_bool_scalar_arithmetic
                     { scalar_op = "rsub_scalar"; fmt = fmt_of x })
            | Op.Addcmul { Pointwise.Addcmul.self; _ }
              when is_bool (fmt_of self) ->
                Err.fail
                  (`Unsupported_bool_scalar_arithmetic
                     { scalar_op = "addcmul"; fmt = fmt_of self })
            | Op.Addcmul { Pointwise.Addcmul.tensor1; _ }
              when is_bool (fmt_of tensor1) ->
                Err.fail
                  (`Unsupported_bool_scalar_arithmetic
                     { scalar_op = "addcmul"; fmt = fmt_of tensor1 })
            | Op.Addcmul { Pointwise.Addcmul.tensor2; _ }
              when is_bool (fmt_of tensor2) ->
                Err.fail
                  (`Unsupported_bool_scalar_arithmetic
                     { scalar_op = "addcmul"; fmt = fmt_of tensor2 })
            (* The index output of [Max_pool2d_with_indices] and its adaptive
               twin is declared I64 by [Builder]; see [Eval_direct]'s own arm
               for why the conversion from the double-carried flat index is
               exact. *)
            | Op.Max_pool2d_with_indices { Pool.MaxPool2dWithIndices.params; x }
              when output = 1 ->
                let module C = Pool.MaxPool2dWithIndices.Compute (Direct) in
                let x_shape = Tensor_id.Map.find x shape_env
                and x = Tensor_id.Map.find x operand_env in
                Err.return
                  (Tensor.materialize_i64 (Shape4.to_vec6 out_shape)
                     (fun coord ->
                       Int64.of_float (C.index_pixel params ~x_shape ~x coord)))
            | Op.Adaptive_max_pool2d_with_indices
                { Pool.AdaptiveMaxPool2dWithIndices.params; x }
              when output = 1 ->
                let module C = Pool.AdaptiveMaxPool2dWithIndices.Compute (Direct)
                in
                let x_shape = Tensor_id.Map.find x shape_env
                and x = Tensor_id.Map.find x operand_env in
                Err.return
                  (Tensor.materialize_i64 (Shape4.to_vec6 out_shape)
                     (fun coord ->
                       Int64.of_float (C.index_pixel params ~x_shape ~x coord)))
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
  let live = live_edges g in
  Err.List.fold_left
    (eval_node ?region_counters ~limits ~synthetic_ids ~live g)
    env g.Graph.Graph.nodes
