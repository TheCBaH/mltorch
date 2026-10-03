(** Structured sums ({!Loop_stmt.Reduce_sum}): the reference expansion every
    other consumer reads, a count, and the checks a sum descriptor must pass.

    The expansion is exactly the loop the lowering has always produced for an
    [Expr.Reduction.Sum]: [acc <- seed], then a [For] whose body is a
    [Reduction] mark, the term's statements, and [acc <- acc + term]. A program
    with structured sums and its expansion therefore have the same values,
    failures and mark counts, and a consumer that does not know the node loses
    nothing by being handed the expansion. *)

val expand : Loop_stmt.t -> Loop_stmt.t list
(** The statements a statement stands for, nested sums included: a [Reduce_sum]
    becomes its seed assignment and loop; any other statement is itself with its
    nested blocks expanded. *)

val block : Loop_stmt.t list -> Loop_stmt.t list

val program : Loop_program.t -> Loop_program.t
(** {!expand} over every statement of a program. *)

val recover_block : Loop_stmt.t list -> Loop_stmt.t list
(** {!recover} over one statement list. *)

val recover : Loop_program.t -> Loop_program.t
(** Recovers structured sums from an expanded (typically optimized) program: a
    loop whose body opens with a [Reduction] mark and ends with
    [acc <- acc + e], with [acc] mentioned nowhere else in the loop and seeded
    by a constant assignment as the statement just before it. Anything else is
    left as it was. {!program} of the result is the input program exactly, and
    the sums in it are the ones the planner may reorder. *)

val count : Loop_program.t -> int
(** The structured sums in a program, nested ones included. *)

type error =
  [ `Accumulator_assigned_in_body of Loop_temp.t
  | `Accumulator_read_by_term of Loop_temp.t
  | `Variable_reused of Loop_var.t ]

val pp_error : Format.formatter -> [< error ] -> unit

val check : Loop_program.t -> (unit, [> error ]) Err.t
(** Every sum's accumulator is written only by the sum itself (no statement of
    its body assigns it) and is never read by its term or body, and a sum's
    variable is not the variable of a loop or sum it sits inside. *)
