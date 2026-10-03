(** Register blocking over rows: unroll-and-jam of a scalar loop around a vector
    loop.

    A dense kernel is a scalar loop over rows (output pixels, batch rows) around
    a vector loop over the output channels, with a sum over the inner dimension
    in each vector iteration. Each row's sum is a chain of dependent
    multiply-adds, so one row runs at the latency of the instruction; and the
    weights it loads are loaded again by the next row. Blocking runs [k]
    consecutive rows in one vector iteration: the rows' independent chains
    overlap and a load they share is issued once.

    Only the order of independent iterations changes: each output cell computes
    the same operations in the same order, so the blocked kernel is bitwise the
    unblocked one. That the rows are independent is not assumed: the blocked
    vector loop goes through {!Loop_vector_check}, which proves the rows' stores
    touch distinct cells and nothing the loop stores is loaded elsewhere; a loop
    that does not pass is left as it was. Applied to a vector loop whose body
    carries an accumulator, directly inside a scalar loop with constant bounds;
    the rows left over run unblocked. *)

val program :
  target:Loop_target.t -> Loop_vector.program -> Loop_vector.program * int
(** The program with every eligible row loop blocked by [target.row_block]
    (fewer rows when the body carries several accumulators), and how many loops
    were blocked. *)
