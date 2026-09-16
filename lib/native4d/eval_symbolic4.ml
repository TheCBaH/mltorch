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

let is_i64 = function Payload.Fmt Payload.I64 -> true | _ -> false

(* The Native4D twin of [Eval_symbolic]'s own fix, same rationale: closes the
   mixed I64/F32 checked-admission gap for Native4D's Symbolic route (Direct
   already rejects this pair, commit [b96e3bf4]); Symbolic still has no
   I64,I64 [Compute_i64] dispatch of its own, so a matching pair is
   unaffected. *)
let check_mixed_dtype (g : Graph.graph) op =
  let fmt_of r = (Tensor_id.Map.find r g.Graph.Graph.tensors).Tensor_sig.fmt in
  let check_pair mixed_op a b =
    let a_fmt = fmt_of a and b_fmt = fmt_of b in
    if Bool.equal (is_i64 a_fmt) (is_i64 b_fmt) then ()
    else
      Err.or_raise ~pp_error:pp_mixed_dtype
        (Err.fail ~pos:__POS__ { mixed_op; a_fmt; b_fmt })
  in
  match op with
  | Op.Add { Pointwise.Bin.a; b } -> check_pair "add" a b
  | Op.Sub { Pointwise.Bin.a; b } -> check_pair "sub" a b
  | Op.Mul { Pointwise.Bin.a; b } -> check_pair "mul" a b
  | _ -> ()

let first_free_tid (g : Graph.graph) =
  Tensor_id.Map.fold
    (fun k _ acc -> max acc (Tensor_id.to_int k + 1))
    g.Graph.Graph.tensors 0

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
  let process_node env stages (node : Graph.node) =
    let op = node.Graph.Node.op in
    check_mixed_dtype g op;
    let operand r = Tensor_id.Map.find r env in
    let shape_of r = (Tensor_id.Map.find r env).Tensor_sig.shape in
    let outs = List.mapi (fun i oid -> (i, oid)) node.Graph.Node.outputs in
    (* Mirrors [Eval_symbolic]'s own multi-output group construction. *)
    if List.length outs > 1 && Region_computation4.is_region_authored op then
      let group =
        match
          Region_computation4.group ~limits:Kernel.Limits.default ~op
            ~operand:(fun id -> Tensor_id.Map.find_opt id env)
        with
        | Ok group -> group
        | Error error ->
            Err.raise_error ~pp_error:Region_computation.pp_error error
      in
      List.fold_left
        (fun (env, stages) (output, oid) ->
          let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
          let st =
            {
              Stage_program.Stage.id = oid;
              sg = out_sig;
              computation = Region_group.Ref.Grouped (group, output);
            }
          in
          (Tensor_id.Map.add oid out_sig env, st :: stages))
        (env, stages) outs
    else
      List.fold_left
        (fun (env, stages) (output, oid) ->
          let out_sig = Tensor_id.Map.find oid g.Graph.Graph.tensors in
          let regional =
            if Region_computation4.is_region_authored op then
              Some
                (Region_computation4.program ~limits:Kernel.Limits.default ~op
                   ~output ~output_shape:out_sig.shape
                   ~operand:(fun id -> Tensor_id.Map.find_opt id env)
                   ~fill:(fun _role value shape -> fill value shape))
            else None
          in
          let computation =
            match regional with
            | Some (Ok program) -> Region_group.Ref.Solo program
            | Some (Error error) ->
                Err.raise_error ~pp_error:Region_computation.pp_error error
            | None ->
                Region_group.Ref.Solo
                  (Region_program.pixel
                     (Expr.Builder.run
                        (E.pixel op ~output ~operand ~shape_of ~fill
                           Symbolic.out_vec)))
          in
          let st =
            { Stage_program.Stage.id = oid; sg = out_sig; computation }
          in
          (Tensor_id.Map.add oid out_sig env, st :: stages))
        (env, stages) outs
  in
  let _env, rev_stages =
    List.fold_left
      (fun (env, stages) node -> process_node env stages node)
      (g.Graph.Graph.tensors, [])
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
    outputs = g.Graph.Graph.outputs;
  }
