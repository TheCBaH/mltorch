(** Each [Contract] with [Ordered] numerics over binary32 or bfloat16 operands,
    or [Fused_sequential] over binary32, as the loops of {!Contraction.eval}: an
    ordered sum per output cell. The fused form is the same loop; it computes
    the fused sums only under a plan that contracts dots, which the caller must
    ensure. A paired or integer contraction is left as it is. For a backend, or
    a Machine IR route, that realizes no [Contract] of its own. *)

val program : Ssa_program.t -> Ssa_program.t

(** How an operand's cells are placed: moved by one base, or by one base per row
    of its outer axis (A's and C's rows, B's columns), which replaces the
    descriptor's outer stride. *)
type view =
  | Base of Ssa_type.index Ssa_builder.value
  | Rows of Ssa_type.index Ssa_builder.value array

val expansion :
  ?views:(Ssa_builder.t -> view * view * view) ->
  Contraction.t ->
  a:Ssa_id.Buffer.t ->
  b:Ssa_id.Buffer.t ->
  c:Ssa_id.Buffer.t ->
  buffers:Ssa_buffer.t list ->
  Ssa_program.t
(** The loops of one contraction as a program over [buffers]. [views], when
    given, makes in the program how each operand (A, B, C) is placed: the
    expansion of a {!Ssa_view_call}. An axis some operand names by rows is
    unrolled. *)

val expandable : Contraction.t -> bool
(** Whether {!program} expands a contraction of this descriptor. *)
