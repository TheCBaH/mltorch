# Interval allocator (`lib/interval_alloc`)

A standalone library that packs a script of allocations and frees into one pool.
It is the placement half of the tensor arena and knows nothing about tensors,
element kinds or Bigarrays: the client makes one problem per pool. Its units are
bytes, as `Core.Storage_units` types, so a client cannot hand it element counts
or mix sizes with offsets.

## Contract

- **Script.** `Script.validate ~equal events` accepts a list of
  `Alloc {key; size; alignment}` / `Free key` events. A key is allocated once
  and freed at most once, after its allocation; a size is non-negative and an
  alignment a positive power of two, both by type. A block never freed lives to
  the end. Keys are compared only through the `equal` given here (and reused by
  `check`), so the library is generic in the key.
- **Alignment.** A block's offset must be a multiple of its alignment. Only the
  start is aligned: a size is never rounded up, so a later block may begin
  inside the same alignment unit. Offsets are relative to the pool, so an
  aligned offset is physically aligned only if the client's pool base is.
- **Liveness is by event position.** A block allocated at `p` and freed at `q`
  is live over `[p, q)`. A `Free` placed after a later `Alloc` therefore
  conflicts with it: this is what keeps a node's operand and its output apart
  when the operand is released after the node runs.
- **Sizes and offsets are `int64`-backed.** js_of_ocaml's `int` is 32 bits.
  Arithmetic goes through `Core.Storage_units`' checked operations; an overflow
  surfaces as `Live_overflow`, `Pool_overflow` or `Offset_overflow`. A failure
  the code has already ruled out (the end of a block the checker placed inside
  its pool) raises rather than inventing a row.
- **Witness.** `check` accepts a `Solution` only if every block is placed
  exactly once, at a multiple of its alignment (`Misaligned` otherwise, even
  when everything fits), inside the pool, without overlapping any block it is
  live with. `Witness.t` is abstract and only `check` builds
  one; `Solution.Unsafe.make` exists so tests can build a bad claim. Errors are
  data (the keys and byte ranges), not prose.
- **Strategies.** `Greedy_by_area`, `Greedy_by_lifetime`, `Greedy_by_size`,
  `Greedy_by_size_best_fit`. Each sorts blocks by its key (largest first, ties in
  allocation order) and places each at the lowest fitting gap (best-fit: the
  smallest fitting gap), measured from the gap's first aligned offset. Deterministic: the result depends on the event order and
  sizes, never on key names or a clock.
- **Lower bound.** The maximum running sum of live sizes: no placement of that
  script, in that order, can use a smaller pool. It ignores alignment, so
  padding can leave the optimum above it.

## Tests

`test/interval_alloc` runs natively and under `@runtest-js`: hand-built scripts,
`validate` rejections, checker mutations (overlap, unplaced, out-of-pool,
misaligned, overflow), and seeded random scripts, with and without alignment,
checking every strategy against `check`, the bound, determinism and
key-renaming. Dropping the decoder's or the reference search's alignment turns
the aligned suites red.

## Search and budget

`solve_best` runs the four strategies (the smallest pool wins, the first strategy
on a tie), then an order search from the winner; `improve` does the same from a
checked solution, ordering blocks by offset.

- **One iteration is one move plus one decode.** A move relocates a block that
  sets the pool to an earlier position in the order, or (one time in four) swaps
  two positions; it is kept if the pool is no larger. The moves of iteration `i`
  do not depend on the budget, so a larger budget extends the same walk and can
  only match or beat a smaller one.
- **Bounded by iterations, never by a clock.** `Budget` is an iteration count and
  a seed. The constructive portfolio runs first and is not counted, so a valid
  solution exists before the search starts.
- **The bound takes precedence.** `Stop.Lower_bound` whenever the returned pool
  equals the lower bound, at any budget, so "optimal for this script's order"
  never depends on the budget. Otherwise `Budget_exhausted`.
- **Measured quality** (tests): on random scripts of up to ~80 blocks the
  constructive best averages ~1.008 of the bound and the search ~1.005; against
  an exact branch-and-bound oracle on scripts of up to 8 blocks the search
  reaches the optimum in all but a few percent of cases.

## Reference minimum (`Reference`)

A bounded exact search that grades the strategies rather than competing with
them: nothing it finds seeds `solve_best` or `improve`.

- **One ceiling.** `Reference.feasible limits work script ~ceiling` answers
  `Feasible` (a checked witness whose pool is its actual length), `Infeasible`
  (proven), or `Unknown Depth | Unknown States`. A branch cut by a limit makes
  the answer `Unknown`; only an exhausted search is `Infeasible`, and a heuristic
  failure never is.
- **Orientation search.** Two positive-size blocks that are live together must be
  stacked one above the other; a set of such choices is a DAG whose longest paths
  are the lowest offsets it allows. The search keeps those offsets incrementally
  (an undo trail, no recursion), and branches only on a conflicting pair whose
  current ranges still overlap: when none do, the offsets are a placement.
  Complete because a fitting placement that honours the choices so far orients
  the overlapping pair one way or the other, and adding that edge only raises
  offsets towards that placement's, so its branch is never pruned; finite
  because every branch adds an edge no earlier choice implied. With alignment, a
  block's offset is the least aligned offset at or above its predecessors' ends;
  completeness survives because an aligned placement at or above each of those
  ends is at or above their aligned-up maximum. Pruning: a block
  whose end would pass the ceiling, and a raise reaching the new edge's source (a
  cycle). A candidate still goes through `check` before it is returned.
- **Search order** (measured on 200k random scripts, not a correctness
  property): the overlapping pair reaching highest first, the larger block below
  first. Choosing the lowest pair instead cost ~3x the states and left ~4x the
  scripts unresolved at 100k states.
- **Bisection.** `Reference.minimum` checks the incumbent, normalizes its pool to
  the highest occupied end (zero-size blocks at zero), and proves
  `lower <= optimum <= upper`: nothing is asked when the incumbent already meets
  the live bound; otherwise the live bound first, then the midpoint of what is
  open below the incumbent. `Feasible` tightens `upper` to the witness's length,
  `Infeasible` lifts `lower` past the ceiling, `Unknown` stops. Work is
  cumulative across a search's queries, not replenished per ceiling. A feasible
  answer below `lower` or above its ceiling is `Invalid_candidate`, a defect.
  `Status.Optimal_above_live_bound` rests on exhaustive `Infeasible` answers, so
  it is distinct from `Stop.Lower_bound`, which only ever means "at the live
  bound".
- **Tests** grade it against an independent brute-force enumeration of integer
  offsets, including a seven-block script whose optimum is one above the live
  bound, and drive the bisection with a scripted oracle. Such scripts are rare
  without alignment: none in hundreds of thousands of random scripts of up to
  five blocks, and one in about 1.5 million of seven. With alignment they are
  common, since padding is outside the live bound: 79 of 400 small scripts with
  alignments up to 4, all agreeing with the aligned brute force.
