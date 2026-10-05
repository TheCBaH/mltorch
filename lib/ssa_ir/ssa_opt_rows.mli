(** Register blocking over rows: unroll-and-jam of a scalar loop around a vector
    loop.

    A dense kernel is a scalar loop over rows (output pixels, batch rows) around
    a vector loop over output channels, with a sum over the inner dimension in
    each vector iteration. Each row's sum is a chain of dependent adds, so one
    row runs at the latency of the instruction, and the weights it loads are
    loaded again by the next row. Blocking runs [rows] consecutive rows in one
    vector iteration: the rows' independent chains share one loop and a load
    they share is issued once.

    Only the order of independent iterations changes. Each output cell computes
    the same operations in the same order, so the blocked kernel is bitwise the
    unblocked one under every numerical policy, and the logical work is the
    same. That the rows are independent is checked, not assumed: the body cannot
    fail, touches neither the scan meter nor a scratch object, reads nothing it
    may write, and every store's coordinates are the row index itself or
    independent of it. The rows left over run unblocked.

    Applied to a loop with constant bounds that directly holds one vector loop
    whose body is setup, one ordered sum and its stores. A row loop with a
    remainder loop beside the vector loop is left as it was. *)

val program :
  rows:int -> policy:Ssa_effects.policy -> Ssa_program.t -> Ssa_program.t * int
(** [rows] is the target's block factor ({!Ssa_target.t}[.row_block]); fewer
    than two blocks nothing. The result and how many loops were blocked. *)

(** Why a row loop was left alone. *)
type refusal =
  | Bounds_vary
      (** a bound of the vector loop or the sum varies with the row *)
  | May_fail
  | Meter_or_locals
  | Reads_written_buffer
  | Shape
  | Stores_not_independent
  | Too_few_rows
  | Trips_unknown

val refusal_name : refusal -> string

val analyze :
  rows:int ->
  policy:Ssa_effects.policy ->
  Ssa_program.t ->
  (Ssa_id.Region.t * (int, refusal) result) list
(** Every loop that holds a vector loop in the shape above, by its body region,
    and the block factor it would get or the condition that failed. *)
