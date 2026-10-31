(** A selected attention layer: its descriptor and the identity of each role's
    tensor in the graph. As for {!Attention_site}, a bundle keeps neither the
    operation nor its operand order, so the roles travel with the descriptor and
    are resolved through the invocation's edges, never by buffer position. *)

(** Each role's tensor; an optional one exactly when the descriptor reads it. *)
module Roles : sig
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

val buffers :
  t ->
  Loop_ir.Loop_bundle.invocation ->
  (Ssa_ir.Ssa_attention_layer.Buffers.t, Attention_site.Unbound.t) result
(** Each role's buffer: the one whose position in the program's buffers holds
    the role's tensor among the invocation's edges. *)

val program :
  t ->
  Ssa_ir.Ssa_attention_layer.Buffers.t ->
  Ssa_ir.Ssa_program.t ->
  (Ssa_ir.Ssa_program.t, string) result
(** The program with its body replaced by the layer's sequence
    ({!Ssa_ir.Ssa_attention_layer}) over the buffers, its buffers kept and the
    sequence's scratch declared after them. A program that does not take and
    return one effect is refused. *)
