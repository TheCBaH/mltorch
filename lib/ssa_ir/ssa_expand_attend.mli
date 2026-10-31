(** A binary32 attention as ordinary operations: each [Attend] whose precision
    is binary32 ({!Attention_block.Precision}) becomes the loops of
    {!Attention_block.eval}, the two products of each key block as
    {!Ssa_op.Contract}s over dense scratch copies of Q, K and V, so no emitter
    sees an [Attend]. The scratch buffers are declared on the program. A
    bfloat16 attention is left as it is. The arithmetic is binary64 until
    {!Ssa_precision.to_f32}, as in every lowered program. *)

val program : Ssa_program.t -> Ssa_program.t
