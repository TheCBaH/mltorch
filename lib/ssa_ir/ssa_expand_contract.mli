(** Each [Contract] with [Ordered] numerics over binary32 or bfloat16 operands,
    or [Fused_sequential] over binary32, as the loops of {!Contraction.eval}: an
    ordered sum per output cell. The fused form is the same loop; it computes
    the fused sums only under a plan that contracts dots, which the caller must
    ensure. A paired or integer contraction is left as it is. For a backend, or
    a Machine IR route, that realizes no [Contract] of its own. *)

val program : Ssa_program.t -> Ssa_program.t

val expansion :
  ?bases:
    (Ssa_builder.t ->
    Ssa_type.index Ssa_builder.value
    * Ssa_type.index Ssa_builder.value
    * Ssa_type.index Ssa_builder.value) ->
  Contraction.t ->
  a:Ssa_id.Buffer.t ->
  b:Ssa_id.Buffer.t ->
  c:Ssa_id.Buffer.t ->
  buffers:Ssa_buffer.t list ->
  Ssa_program.t
(** The loops of one contraction as a program over [buffers]. [bases], when
    given, makes in the program the index each operand's cells are offset by (A,
    B, C): the expansion of a {!Ssa_view_call}. *)

val expandable : Contraction.t -> bool
(** Whether {!program} expands a contraction of this descriptor. *)
