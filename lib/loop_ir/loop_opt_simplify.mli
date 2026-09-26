(** Index simplification against the enclosing loops' ranges: a point range
    becomes its constant, an ordered [Max]/[Min]/[Clamp_low] its winning
    operand, a division sheds the multiples of its divisor, and an [Add]/[Scale]
    tree is re-summed when that is cheaper. Every rewritten node and its
    replacement are proven overflow-free first. *)

val run : Loop_program.t -> Loop_program.t

val with_proof :
  (Loop_range.Env.t -> Loop_index.t -> bool) -> Loop_program.t -> Loop_program.t
(** [run] with the overflow proof replaced, for the mutation test:
    [with_proof Loop_range.proven] is [run]. *)
