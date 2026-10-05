(** The executable specification: a virtual-environment interpreter that shares
    no code with the builder, the lowering or any emitter. It verifies a program
    first, executes only the selected branch of an [if], evaluates loop bounds
    once, transfers every carried value of an iteration at once, and leaves on
    the first failure.

    Execution recurses once per region nesting level and {!Ssa_verify} bounds
    that level at [Ssa_verify.max_region_depth], which is what keeps a deep
    supported program inside a JavaScript stack. *)

(** Logical work and primitive counts. Marks are the work a transformation must
    preserve; loads and stores are diagnostics of the program as transformed. *)
module Counters : sig
  type t

  val create : unit -> t
  val mark : t -> Ssa_mark.t -> int
  val loads : t -> int
  val stores : t -> int
end

type failure =
  [ `Coord_out_of_range of Expr.Source.t * Expr.Axis.t * int * int Expr.Coord.t
  | `Gather_index_out_of_range of Expr.Eval.Gather_index_out_of_range.t
  | `I64_division_by_zero
  | `I64_division_overflow
  | `I64_from_float_infinite
  | `I64_from_float_nan
  | `I64_from_float_out_of_range of float
  | `Index_overflow of Expr.Index_overflow.t ]
(** The failures a program can raise, the rows [Expr.Eval] and [Kernel_eval]
    report for the same events. *)

type error = [ failure | `Invalid_program of Ssa_verify.diagnostic ]

val pp_error : Format.formatter -> [< error ] -> unit

val run :
  ?counters:Counters.t ->
  Ssa_program.t ->
  memory:Ssa_memory.t ->
  (unit, error) Err.t
(** Executes the program over [memory], which must hold every declared buffer.
    An access outside a buffer through a flat offset or a store is a defect in
    the program and raises [Invalid_argument]: only a checked operation fails
    with a row. *)
