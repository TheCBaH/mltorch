(** Which [Sdpa] nodes the fused attention function computes. A granted numerics
    ([bf16_attention]: the streaming recurrence of {!Attention_block.eval}, its
    key block, bfloat16 probabilities and paired products), a core with the
    matrix engine, bfloat16 Q, K and V, a binary32 mask, and shapes the function
    takes; anything else keeps the generated path, with the reason recorded. *)

(** The grant. Like the pair rounding it is a flag beside the numerics policy,
    never a silent substitute. *)
module Grant : sig
  type t = { block_keys : int; precision : Attention_block.Precision.t }
  (** The keys per block, part of the arithmetic, and the precision: the
      bfloat16 kernel's, or a binary32 form that is expanded into ordinary
      operations and so needs no engine and none of the kernel's shape limits.
  *)

  val default : t
  (** 32 keys, the bfloat16 kernel's. *)
end

(** Why a node is not selected. Alphabetical. *)
module Reason : sig
  type t =
    | Block_keys of int
        (** the grant's block, neither at most 16 nor 32 keys *)
    | Descriptor of string  (** the descriptor's own refusal *)
    | Engine_absent
    | Feature_extent of int  (** a head dimension that is odd or past 64 *)
    | Mask_not_f32
    | Not_bf16 of string  (** the role whose format is not bfloat16 *)
    | Not_f32 of string  (** the role whose format is not binary32 *)
    | Not_sdpa
    | Outer_axes  (** an operand extent past 1 on the N or T axis *)
    | Padded_work of { padded : int64; actual : int64 }
        (** the fixed blocks multiply more than twice the work *)
    | Ungranted
    | Unknown_node

  val pp : t Fmt.t
end

val choose :
  grant:Grant.t option ->
  Contraction.Select.Capabilities.t ->
  Graph_ir.graph ->
  Graph_ir.Node_id.t ->
  (Attention_site.t, Reason.t) result
