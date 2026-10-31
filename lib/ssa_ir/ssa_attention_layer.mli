(** A granted attention layer ({!Attention_layer_block}) as a sequence of
    existing operations over its buffers and declared scratch:

    {v
    [norm loops, Pre] -> 3 x (Contract ; bias loop) -> [rotation loops on Q, K]
      -> Attend -> Contract ; bias, gamma and residual loop -> [norm loops, Post]
    v}

    The norm loops are {!Ssa_norm}'s; the projections are [Contract]s over the
    dense row-major operands; the rotation reads each pair of half-row features
    before it writes either, so it needs no copy; the attention is one [Attend]
    over the head-strided scratch, which {!Ssa_expand_attend} expands for
    binary32. The arithmetic is binary64 until {!Ssa_precision.to_f32}. A
    bfloat16 layer has no body yet: no store rounds to bfloat16. *)

(** The buffer of each operand the layer reads, and its output. An optional one
    is needed exactly when the descriptor reads it. *)
module Buffers : sig
  type t = {
    x : Ssa_id.Buffer.t;
    norm_weight : Ssa_id.Buffer.t option;
    norm_bias : Ssa_id.Buffer.t option;
    wq : Ssa_id.Buffer.t;
    bq : Ssa_id.Buffer.t option;
    wk : Ssa_id.Buffer.t;
    bk : Ssa_id.Buffer.t option;
    wv : Ssa_id.Buffer.t;
    bv : Ssa_id.Buffer.t option;
    wo : Ssa_id.Buffer.t;
    bo : Ssa_id.Buffer.t option;
    gamma : Ssa_id.Buffer.t option;
    cos : Ssa_id.Buffer.t option;
    sin : Ssa_id.Buffer.t option;
    mask : Ssa_id.Buffer.t option;
    y : Ssa_id.Buffer.t;
  }
end

val scratch :
  Attention_layer_block.t -> first:Ssa_id.Buffer.t -> Ssa_buffer.t list
(** The binary32 scratch the body uses, ids from [first]: the norm's output when
    a norm stands before the attention, Q, K, V, the attention's output, the
    output projection's, and the residual when a norm follows it. *)

val body :
  Ssa_builder.t ->
  Attention_layer_block.t ->
  Buffers.t ->
  scratch:Ssa_buffer.t list ->
  unit
(** The sequence, appended to the builder's current region, over {!scratch}'s
    buffers. [Invalid_argument] for a bfloat16 layer or a missing buffer. *)

val program :
  Attention_layer_block.t ->
  Buffers.t ->
  buffers:Ssa_buffer.t list ->
  Ssa_program.t
(** {!body} as a whole program over [buffers] and the scratch, which it declares
    with ids after the largest of [buffers]. *)
