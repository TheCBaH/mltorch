(** Which [Attention_layer] nodes are computed as one granted layer
    ({!Attention_layer_block}): the grant {!Sdpa_select} takes (its key block
    and precision) and a descriptor the node's shapes make. Every operand is
    binary32, except that a bfloat16 layer ([Bf16_paired]) reads bfloat16
    weights and needs the matrix engine and the fused attention function's
    limits (an even head dimension at most 64, a block of at most 16 or 32 keys,
    at most twice the work padded); anything else keeps the node off the granted
    path, with the reason recorded. *)

(** Why a node is not selected. Alphabetical. *)
module Reason : sig
  type t =
    | Attention of Sdpa_select.Reason.t
        (** a reason the attention's own selection gives, in the same words: the
            grant, the node, an operand's format, the block, the descriptor *)
    | Not_attention_layer
    | Precision of Attention_block.Precision.t
        (** a precision the layer has no body for (none today) *)

  val pp : t Fmt.t
end

val choose :
  grant:Sdpa_select.Grant.t option ->
  Contraction.Select.Capabilities.t ->
  Graph_ir.graph ->
  Graph_ir.Node_id.t ->
  (Layer_site.t, Reason.t) result
