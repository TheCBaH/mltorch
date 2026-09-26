(** Deterministic printing of a Loop IR program, for diagnostics and expect
    tests. Loop variables, temporaries and arrays are named by position of first
    appearance in the printed text ([i0], [t0], [a0]), never by their allocation
    ids, so two structurally identical programs built by different allocation
    histories print identically (the [Expr.Pp] model). *)

val program : Format.formatter -> Loop_program.t -> unit
val stmts : Format.formatter -> Loop_stmt.t list -> unit
