(** The loops of a row normalization ({!Norm_block}) over given buffers: per row
    the sums, the inverse root and the affine output, in the order
    {!Norm_block.eval} states. The sums of an [F32_ordered] descriptor are
    ordered sums, its sum of squares one product a term, so a plan that
    contracts dots fuses exactly that site; an [F32_fused] one carries its sum
    of squares through explicit fused multiply-adds, so every backend computes
    it whatever the plan. The arithmetic is binary64 until
    {!Ssa_precision.to_f32}, as in every lowered program. *)

val body :
  Ssa_builder.t ->
  Norm_block.t ->
  x:Ssa_id.Buffer.t ->
  weight:Ssa_id.Buffer.t option ->
  bias:Ssa_id.Buffer.t option ->
  y:Ssa_id.Buffer.t ->
  unit
(** The loops, appended to the builder's current region. X and Y are addressed
    by the descriptor's strides from offset 0, the weight and the bias densely;
    each present affine operand needs its buffer. [Invalid_argument] for a
    bfloat16 descriptor (no store rounds to bfloat16) or a missing buffer. *)

val program :
  Norm_block.t ->
  x:Ssa_id.Buffer.t ->
  weight:Ssa_id.Buffer.t option ->
  bias:Ssa_id.Buffer.t option ->
  y:Ssa_id.Buffer.t ->
  buffers:Ssa_buffer.t list ->
  Ssa_program.t
(** {!body} as a whole program over [buffers]. *)
