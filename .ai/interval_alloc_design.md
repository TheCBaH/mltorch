# Interval allocator (`lib/interval_alloc`)

A standalone library that packs a script of allocations and frees into one pool.
It is the placement half of the tensor arena and knows nothing about tensors,
element kinds or Bigarrays: the client makes one problem per pool and owns the
units.

## Contract

- **Script.** `Script.validate ~equal events` accepts a list of
  `Alloc {key; size}` / `Free key` events. A key is allocated once and freed at
  most once, after its allocation; a size is non-negative. A block never freed
  lives to the end. Keys are compared only through the `equal` given here (and
  reused by `check`), so the library is generic in the key.
- **Liveness is by event position.** A block allocated at `p` and freed at `q`
  is live over `[p, q)`. A `Free` placed after a later `Alloc` therefore
  conflicts with it: this is what keeps a node's operand and its output apart
  when the operand is released after the node runs.
- **Sizes and offsets are `int64`.** js_of_ocaml's `int` is 32 bits. Sums are
  overflow-checked (`Live_overflow`, `Pool_overflow`, `Offset_overflow`).
- **Witness.** `check` accepts a `Solution` only if every block is placed
  exactly once, at a non-negative offset, inside the pool, without overlapping
  any block it is live with. `Witness.t` is abstract and only `check` builds
  one; `Solution.Unsafe.make` exists so tests can build a bad claim. Errors are
  data (the keys and byte ranges), not prose.
- **Strategies.** `Greedy_by_area`, `Greedy_by_lifetime`, `Greedy_by_size`,
  `Greedy_by_size_best_fit`. Each sorts blocks by its key (largest first, ties in
  allocation order) and places each at the lowest fitting gap (best-fit: the
  smallest fitting gap). Deterministic: the result depends on the event order and
  sizes, never on key names or a clock.
- **Lower bound.** The maximum running sum of live sizes: no placement of that
  script, in that order, can use a smaller pool.

## Tests

`test/interval_alloc` runs natively and under `@runtest-js`: hand-built scripts,
`validate` rejections, checker mutations (overlap, unplaced, out-of-pool,
overflow), and seeded random scripts checking every strategy against `check`,
the bound, determinism and key-renaming.

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
