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
    bfloat16 layer rounds each paired product's and the attention's operands
    into bfloat16 scratch (the norm's output, or X; Q, K and V; the attention's
    output), and its projections are [Paired] [Contract]s over bfloat16 weights.
*)

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
(** The scratch the body uses, ids from [first]: the norm's output when a norm
    stands before the attention (or, bfloat16, X's copy), Q, K, V, the
    attention's output, the output projection's, the residual when a norm
    follows it, and for a bfloat16 layer the bfloat16 copies of Q, K, V and the
    attention's output. *)

val body :
  Ssa_builder.t ->
  Attention_layer_block.t ->
  Buffers.t ->
  scratch:Ssa_buffer.t list ->
  unit
(** The sequence, appended to the builder's current region, over {!scratch}'s
    buffers. [Invalid_argument] for a missing buffer. *)

val program :
  Attention_layer_block.t ->
  Buffers.t ->
  buffers:Ssa_buffer.t list ->
  Ssa_program.t
(** {!body} as a whole program over [buffers] and the scratch, which it declares
    with ids after the largest of [buffers]. *)
