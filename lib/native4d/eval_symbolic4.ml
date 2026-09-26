(* Symbolic evaluation of a Native4D graph, into the SAME [Stage_program.t] a
   Native graph produces. The twin of [Eval_symbolic].

   That the result type is shared and unparameterised is what makes cross-dialect
   verification cheap: [Stage_program.t] is typed on [Tensor_id.t],
   [Tensor_sig.t] and [float Expr.Value.t], none of which are dialect-specific, so the whole
   grounding and comparison machinery below [Map_verify] works on a Native4D
   program with no change at all. Design §9.3 assumes this; it holds only because
   Native4D reuses [Tensor_sig.t] verbatim (correction C3). *)

let f32 = Payload.Fmt Payload.F32

type mixed_dtype = {
  mixed_op : string;
  a_fmt : Payload.packed_fmt;
  b_fmt : Payload.packed_fmt;
}

let pp_mixed_dtype fmt
    { mixed_op; a_fmt = Payload.Fmt a_fmt; b_fmt = Payload.Fmt b_fmt } =
  Format.fprintf fmt "%s: unsupported mixed dtype, a=%s b=%s" mixed_op
    (Payload.fmt_name a_fmt) (Payload.fmt_name b_fmt)

let pp_bool_arithmetic fmt
    { mixed_op; a_fmt = Payload.Fmt a_fmt; b_fmt = Payload.Fmt b_fmt } =
  Format.fprintf fmt
    "%s: arithmetic on a Bool operand is not supported, a=%s b=%s" mixed_op
    (Payload.fmt_name a_fmt) (Payload.fmt_name b_fmt)

type scalar_op = { scalar_op : string; fmt : Payload.packed_fmt }

let pp_bool_scalar_arithmetic fmt { scalar_op; fmt = Payload.Fmt f } =
  Format.fprintf fmt "%s: arithmetic on a Bool operand is not supported, x=%s"
    scalar_op (Payload.fmt_name f)

let is_i64 = function Payload.Fmt Payload.I64 -> true | _ -> false

(* The ops whose second output is an argmax-style index. *)
let is_index_output (op : Op.t) output =
  Output_ordinal.equal output Output_ordinal.one
  &&
  match op with
  | Op.Adaptive_max_pool2d_with_indices _ | Op.Max_dim4 _
  | Op.Max_pool2d_with_indices _ ->
      true
  | _ -> false

let is_bool = function Payload.Fmt Payload.Bool -> true | _ -> false

(* The Native4D twin of [Eval_symbolic]'s own fix, same rationale: closes the
   mixed I64/F32 checked-admission gap for Native4D's Symbolic route (Direct
   already rejects this pair); Symbolic still has no
   I64,I64 [Compute_i64] dispatch of its own, so a matching pair is
   unaffected. *)
let check_mixed_dtype (g : Graph.graph) op =
  let fmt_of r = (Tensor_id.Map.find r g.Graph.Graph.tensors).Tensor_sig.fmt in
  let check_pair mixed_op a b =
    let a_fmt = fmt_of a and b_fmt = fmt_of b in
    (* Arithmetic on Bool stays rejected, checked FIRST so a Bool paired with
       I64 reports the Bool reason, not the unrelated I64-mixing one below --
       the Native4D twin of [Eval_symbolic]'s own fix/precedence. *)
    if is_bool a_fmt || is_bool b_fmt then
      Err.or_raise ~pp_error:pp_bool_arithmetic
        (Err.fail ~pos:__POS__ { mixed_op; a_fmt; b_fmt })
    else if Bool.equal (is_i64 a_fmt) (is_i64 b_fmt) then ()
    else
      Err.or_raise ~pp_error:pp_mixed_dtype
        (Err.fail ~pos:__POS__ { mixed_op; a_fmt; b_fmt })
  in
  let check_scalar_op scalar_op x =
    let fmt = fmt_of x in
    if is_bool fmt then
      Err.or_raise ~pp_error:pp_bool_scalar_arithmetic
        (Err.fail ~pos:__POS__ { scalar_op; fmt })
  in
  match op with
  | Op.Add { Pointwise.Bin.a; b } -> check_pair "add" a b
  | Op.Sub { Pointwise.Bin.a; b } -> check_pair "sub" a b
  | Op.Mul { Pointwise.Bin.a; b } -> check_pair "mul" a b
  | Op.Mul_scalar { Pointwise.Scalar_bin.x; _ } ->
      check_scalar_op "mul_scalar" x
  (* The Native4D twin of [Eval_symbolic]'s own extension of this same check
     to the rest of the `*_scalar` family -- see that file's own comment. *)
  | Op.Add_scalar { Pointwise.Scalar_bin.x; _ } ->
      check_scalar_op "add_scalar" x
  | Op.Div_scalar { Pointwise.Scalar_bin.x; _ } ->
      check_scalar_op "div_scalar" x
  | Op.Floor_div_scalar { Pointwise.Scalar_bin.x; _ } ->
      check_scalar_op "floor_div_scalar" x
  | Op.Pow { Pointwise.Scalar_bin.x; _ } -> check_scalar_op "pow" x
  | Op.Rpow_scalar { Pointwise.Scalar_bin.x; _ } ->
      check_scalar_op "rpow_scalar" x
  | Op.Rsub_scalar { Pointwise.Rsub_scalar.x; _ } ->
      check_scalar_op "rsub_scalar" x
  | Op.Addcmul { Pointwise.Addcmul.self; tensor1; tensor2; _ } ->
      check_scalar_op "addcmul" self;
      check_scalar_op "addcmul" tensor1;
      check_scalar_op "addcmul" tensor2
  | _ -> ()

let first_free_tid (g : Graph.graph) =
  Tensor_id.Map.fold
    (fun k _ acc -> max acc (Tensor_id.to_int k + 1))
    g.Graph.Graph.tensors 0

(* The Native4D twin of [Eval_symbolic.i64_arange_pixel] -- see its own
   comment for the full rationale (same [Expr.Value] typed constructors, same
   [length_exact]-backed overflow-safety argument, same "no [Eval_op.Make]"
   reason). [Symbolic] here is [Native.Symbolic], reused unqualified per this
   library's own `wrapped`/[include_subdirs] convention (see this file's dune
   comment), not a Native4D-specific redefinition. *)
let i64_arange_pixel
    ({ Factory.Arange.Exact.start; step; _ } : Factory.Arange.Exact.t) =
  Expr.Value.i64_add
    (Expr.Value.i64_const start)
    (Expr.Value.i64_mul
       (Expr.Value.i64_const step)
       (Expr.Value.i64_of_index (Symbolic.of_index Symbolic.out_vec.Vec6.c)))

let run (g : Graph.graph) : Stage_program.t =
  (* [Symbolic] is stateless, so there is no instance to create. Each stage body
     is a construction computation, run below from [Expr.Builder.initial]: stage
     expressions therefore reuse reducer ordinals, which is correct because a
     reducer identity means nothing outside the expression that binds it. *)
  let module E = Eval_op4.Make (Symbolic) in
  let consts = ref [] in
  let next_const = ref (first_free_tid g) in
  let fill v shape =
    let id = Tensor_id.of_int !next_const in
    incr next_const;
    let sg = Tensor_sig.create ~id ~name:"" ~shape ~fmt:f32 () in
    consts := (sg, v) :: !consts;
    sg
  in
  let process_node (env, stages, stages_i64) (node : Graph.node) =
    let op = node.Graph.Node.op in
    check_mixed_dtype g op;
    let operand r = Tensor_id.Map.find r env in
    let shape_of r = (Tensor_id.Map.find r env).Tensor_sig.shape in
    let outs = Output_ordinal.indexed node.Graph.Node.outputs in
    match (op, outs) with
    (* Mirrors [Eval_symbolic]'s own exact-Arange special case. *)
    | ( Op.Arange4 { Ops4.Arange4.params = { fmt; exact = Some e; _ } },
        [ (_, oid) ] )
      when is_i64 fmt ->
        let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
        let st =
          {
            Stage_program.Stage_i64.id = oid;
            sg = out_sig;
            pixel = i64_arange_pixel e;
          }
        in
        (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
    (* Mirrors [Eval_symbolic]'s own dtype-preserving [Reshape] arm -- same
       rationale, same [Compute_i64 (Symbolic) (Symbolic)] instantiation
       ([Symbolic] here is [Native.Symbolic], reused unqualified per this
       library's own convention, not a Native4D-specific redefinition); only
       the [Shape4.to_vec6] conversion for [params.shape] differs, matching
       [Eval_direct4]'s own [Reshape4] arm. *)
    | Op.Reshape4 { Ops4.Reshape4.params; x }, [ (_, oid) ]
      when is_i64 (operand x).Tensor_sig.fmt ->
        let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
        let x_sig = operand x in
        let module C = Reshape.Reshape.Compute_i64 (Symbolic) (Symbolic) in
        let pixel =
          Expr.Builder.run
            (C.pixel
               { Reshape.Reshape.shape = Shape4.to_vec6 params.shape }
               ~x_shape:x_sig.Tensor_sig.shape ~x:x_sig Symbolic.out_vec)
        in
        let st = { Stage_program.Stage_i64.id = oid; sg = out_sig; pixel } in
        (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
    (* Mirrors [Eval_symbolic]'s own dtype-preserving [Permute] arm -- same
       rationale/instantiation as this file's own [Reshape4] arm just above;
       only the [Graph_shape4.perm6] conversion differs, matching
       [Eval_direct4]'s own [Permute4] arm. *)
    | Op.Permute4 { Ops4.Permute4.perm; x }, [ (_, oid) ]
      when is_i64 (operand x).Tensor_sig.fmt ->
        let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
        let x_sig = operand x in
        let module C = Permute.Permute.Compute_i64 (Symbolic) (Symbolic) in
        let pixel =
          Expr.Builder.run
            (C.pixel (Graph_shape4.perm6 perm) ~x:x_sig Symbolic.out_vec)
        in
        let st = { Stage_program.Stage_i64.id = oid; sg = out_sig; pixel } in
        (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
    (* Mirrors [Eval_symbolic]'s own dtype-preserving tensor-tensor
       [Add]/[Sub]/[Mul] arms -- same rationale (checking just [a]'s format
       suffices since [check_mixed_dtype] above already rejects a mismatched
       pair), same [Compute_i64 (Symbolic) (Symbolic)] instantiation. *)
    | Op.Add { Pointwise.Bin.a; b }, [ (_, oid) ]
      when is_i64 (operand a).Tensor_sig.fmt ->
        let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
        let a_sig = operand a and b_sig = operand b in
        let module C = Pointwise.Add.Compute_i64 (Symbolic) (Symbolic) in
        let pixel =
          Expr.Builder.run
            (C.pixel ~a_shape:a_sig.Tensor_sig.shape
               ~b_shape:b_sig.Tensor_sig.shape a_sig b_sig Symbolic.out_vec)
        in
        let st = { Stage_program.Stage_i64.id = oid; sg = out_sig; pixel } in
        (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
    | Op.Sub { Pointwise.Bin.a; b }, [ (_, oid) ]
      when is_i64 (operand a).Tensor_sig.fmt ->
        let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
        let a_sig = operand a and b_sig = operand b in
        let module C = Pointwise.Sub.Compute_i64 (Symbolic) (Symbolic) in
        let pixel =
          Expr.Builder.run
            (C.pixel ~a_shape:a_sig.Tensor_sig.shape
               ~b_shape:b_sig.Tensor_sig.shape a_sig b_sig Symbolic.out_vec)
        in
        let st = { Stage_program.Stage_i64.id = oid; sg = out_sig; pixel } in
        (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
    | Op.Mul { Pointwise.Bin.a; b }, [ (_, oid) ]
      when is_i64 (operand a).Tensor_sig.fmt ->
        let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
        let a_sig = operand a and b_sig = operand b in
        let module C = Pointwise.Mul.Compute_i64 (Symbolic) (Symbolic) in
        let pixel =
          Expr.Builder.run
            (C.pixel ~a_shape:a_sig.Tensor_sig.shape
               ~b_shape:b_sig.Tensor_sig.shape a_sig b_sig Symbolic.out_vec)
        in
        let st = { Stage_program.Stage_i64.id = oid; sg = out_sig; pixel } in
        (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
    (* Mirrors [Eval_symbolic]'s own [Mul_scalar] arm -- output stays the
       ordinary float carrier, so this produces a [Stage.t] pushed onto
       [stages], not a [Stage_i64.t]; see that arm's own comment. *)
    | Op.Mul_scalar { Pointwise.Scalar_bin.x; scalar }, [ (_, oid) ]
      when is_i64 (operand x).Tensor_sig.fmt ->
        let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
        let x_sig = operand x in
        let module C = Pointwise.Mul_scalar.Compute_i64 (Symbolic) (Symbolic) in
        let pixel = Expr.Builder.run (C.pixel ~scalar x_sig Symbolic.out_vec) in
        let st =
          {
            Stage_program.Stage.id = oid;
            sg = out_sig;
            computation = Region_group.Ref.Solo (Region_program.pixel pixel);
          }
        in
        (Tensor_id.Map.add oid out_sig env, st :: stages, stages_i64)
    (* Mirrors [Eval_symbolic]'s own [To_copy] [Float] arm -- output stays
       the ordinary float carrier, so this produces a [Stage.t] pushed onto
       [stages], not a [Stage_i64.t]; see that arm's own comment. [Long]/
       [Bool] targets are untouched, matching [Eval_direct4]'s own scope. *)
    | ( Op.To_copy { Pointwise.To_copy.target = Pointwise.To_copy.Float; x },
        [ (_, oid) ] )
      when is_i64 (operand x).Tensor_sig.fmt ->
        let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
        let x_sig = operand x in
        let module C = Pointwise.To_copy.Compute_i64 (Symbolic) (Symbolic) in
        let pixel = Expr.Builder.run (C.pixel x_sig Symbolic.out_vec) in
        let st =
          {
            Stage_program.Stage.id = oid;
            sg = out_sig;
            computation = Region_group.Ref.Solo (Region_program.pixel pixel);
          }
        in
        (Tensor_id.Map.add oid out_sig env, st :: stages, stages_i64)
    (* [To_copy]'s [Long] target on an F32 or Bool operand: the checked
       Float-to-I64 cast ([Symbolic.float_to_i64], whose bounds check runs at
       evaluation time) as an exact int64 stage, mirroring [Eval_direct]'s
       [Compute_to_long] arm. A Bool operand reads as exact 0./1., so the same
       cast is exact for it. This is the int64-reads-float direction that
       [Kernel.create] now admits, so the stage may read a computed float
       stage (mvitv2's [add.Tensor -> _to_copy(Long)]). Other operand formats
       keep the default arm, unchanged. *)
    | ( Op.To_copy { Pointwise.To_copy.target = Pointwise.To_copy.Long; x },
        [ (_, oid) ] )
      when match (operand x).Tensor_sig.fmt with
           | Payload.Fmt Payload.F32 | Payload.Fmt Payload.Bool -> true
           | _ -> false ->
        let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
        let x_sig = operand x in
        let module C = Pointwise.To_copy.Compute_to_long (Symbolic) (Symbolic)
        in
        let pixel = Expr.Builder.run (C.pixel x_sig Symbolic.out_vec) in
        let st = { Stage_program.Stage_i64.id = oid; sg = out_sig; pixel } in
        (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
    (* Mirrors [Eval_symbolic]'s own multi-output group construction. *)
    | _, _
      when List.length outs > 1 && Region_computation4.is_region_authored op ->
        let group =
          match
            Region_computation4.group ~limits:Kernel.Limits.default ~op
              ~operand:(fun id -> Tensor_id.Map.find_opt id env)
          with
          | Ok group -> group
          | Error error ->
              Err.raise_error ~pp_error:Region_computation.pp_error error
        in
        let env, stages =
          List.fold_left
            (fun (env, stages) (output, oid) ->
              let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
              let st =
                {
                  Stage_program.Stage.id = oid;
                  sg = out_sig;
                  computation =
                    Region_group.Ref.Grouped
                      (group, Region_computation.emitter_of_output output);
                }
              in
              (Tensor_id.Map.add oid out_sig env, st :: stages))
            (env, stages) outs
        in
        (env, stages, stages_i64)
    | _, _ ->
        let env, stages, stages_i64 =
          List.fold_left
            (fun (env, stages, stages_i64) (output, oid) ->
              let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
              let float_pixel () =
                Expr.Builder.run
                  (E.pixel op ~output ~operand ~shape_of ~fill Symbolic.out_vec)
              in
              if is_index_output op output && is_i64 out_sig.Tensor_sig.fmt then
                (* The index output is declared I64; see [Eval_symbolic]'s
                   own arm for why the checked cast is exact. *)
                let pixel = Expr.Value.float_to_i64 (float_pixel ()) in
                let st =
                  { Stage_program.Stage_i64.id = oid; sg = out_sig; pixel }
                in
                (Tensor_id.Map.add oid out_sig env, stages, st :: stages_i64)
              else
                let regional =
                  if Region_computation4.is_region_authored op then
                    Some
                      (Region_computation4.program ~limits:Kernel.Limits.default
                         ~op ~output ~output_shape:out_sig.shape
                         ~operand:(fun id -> Tensor_id.Map.find_opt id env)
                         ~fill:(fun _role value shape -> fill value shape))
                  else None
                in
                let computation =
                  match regional with
                  | Some (Ok program) -> Region_group.Ref.Solo program
                  | Some (Error error) ->
                      Err.raise_error ~pp_error:Region_computation.pp_error
                        error
                  | None ->
                      Region_group.Ref.Solo
                        (Region_program.pixel (float_pixel ()))
                in
                let st =
                  { Stage_program.Stage.id = oid; sg = out_sig; computation }
                in
                (Tensor_id.Map.add oid out_sig env, st :: stages, stages_i64))
            (env, stages, stages_i64) outs
        in
        (env, stages, stages_i64)
  in
  let _env, rev_stages, rev_stages_i64 =
    List.fold_left process_node
      (g.Graph.Graph.tensors, [], [])
      g.Graph.Graph.nodes
  in
  {
    Stage_program.inputs =
      List.map
        (fun id -> (id, Tensor_id.Map.find id g.Graph.Graph.tensors))
        g.Graph.Graph.inputs;
    input_kinds = g.Graph.Graph.input_kinds;
    consts = List.rev !consts;
    stages = List.rev rev_stages;
    stages_i64 = List.rev rev_stages_i64;
    outputs = g.Graph.Graph.outputs;
  }
