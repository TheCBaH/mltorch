open Ssa_ir

module Roles = struct
  type t = {
    x : Tensor_id.t;
    norm_weight : Tensor_id.t option;
    norm_bias : Tensor_id.t option;
    wq : Tensor_id.t;
    bq : Tensor_id.t option;
    wk : Tensor_id.t;
    bk : Tensor_id.t option;
    wv : Tensor_id.t;
    bv : Tensor_id.t option;
    wo : Tensor_id.t;
    bo : Tensor_id.t option;
    gamma : Tensor_id.t option;
    cos : Tensor_id.t option;
    sin : Tensor_id.t option;
    mask : Tensor_id.t option;
    output : Tensor_id.t;
  }
end

type t = { block : Attention_layer_block.t; roles : Roles.t }

module U = Attention_site.Unbound

let buffers (t : t) (inv : Loop_ir.Loop_bundle.invocation) =
  let positional =
    List.combine inv.Loop_ir.Loop_bundle.program.Loop_ir.Loop_program.buffers
      inv.Loop_ir.Loop_bundle.edges
  in
  let find role tensor =
    match
      List.find_opt (fun (_, edge) -> Tensor_id.equal edge tensor) positional
    with
    | Some (b, _) ->
        Ok (Ssa_id.Buffer.of_int (Tensor_id.to_int b.Loop_ir.Loop_buffer.id), b)
    | None -> Error (U.Role { role; tensor })
  in
  let ( let* ) = Result.bind in
  let one role tensor =
    let* id, _ = find role tensor in
    Ok id
  in
  let opt role = function
    | None -> Ok None
    | Some tensor ->
        let* id = one role tensor in
        Ok (Some id)
  in
  let r = t.roles in
  let* x = one "input" r.Roles.x in
  let* norm_weight = opt "norm weight" r.Roles.norm_weight in
  let* norm_bias = opt "norm bias" r.Roles.norm_bias in
  let* wq = one "query weight" r.Roles.wq in
  let* bq = opt "query bias" r.Roles.bq in
  let* wk = one "key weight" r.Roles.wk in
  let* bk = opt "key bias" r.Roles.bk in
  let* wv = one "value weight" r.Roles.wv in
  let* bv = opt "value bias" r.Roles.bv in
  let* wo = one "output weight" r.Roles.wo in
  let* bo = opt "output bias" r.Roles.bo in
  let* gamma = opt "layer scale" r.Roles.gamma in
  let* cos = opt "cos" r.Roles.cos in
  let* sin = opt "sin" r.Roles.sin in
  let* mask = opt "mask" r.Roles.mask in
  let* y, yb = find "output" r.Roles.output in
  match yb.Loop_ir.Loop_buffer.role with
  | Loop_ir.Loop_buffer.Output ->
      Ok
        {
          Ssa_attention_layer.Buffers.x;
          norm_weight;
          norm_bias;
          wq;
          bq;
          wk;
          bk;
          wv;
          bv;
          wo;
          bo;
          gamma;
          cos;
          sin;
          mask;
          y;
        }
  | _ -> Error (U.Not_an_output r.Roles.output)

let program (t : t) (b : Ssa_attention_layer.Buffers.t) (p : Ssa_program.t) =
  match
    ( p.Ssa_program.entry.Ssa_region.params,
      t.block.Attention_layer_block.precision )
  with
  | _, Attention_block.Precision.Bf16_paired ->
      Error "a bfloat16 attention layer has no body yet"
  | [ effect ], (Attention_block.Precision.F32_fused | F32_ordered)
    when effect.Ssa_value.ty = Ssa_type.Effect ->
      let q =
        Ssa_attention_layer.program t.block b ~buffers:p.Ssa_program.buffers
      in
      Ok
        {
          q with
          Ssa_program.revision = Ssa_id.Revision.succ p.Ssa_program.revision;
          scan_limits = p.Ssa_program.scan_limits;
        }
  | _ -> Error "the program's entry does not take exactly one effect"
