# Strict vectorization over the Loop IR

Status: Part II of the Wasm/SIMD work. The scalar Wasm backend (see the Wasm
backend design record) is the base; nothing here may change what it computes.
This record fixes the contracts; sections marked *measured* quote the evidence
that decided them.

## Numerical contract (strict mode)

Vectorization changes how many cells are computed at once, never what one cell
computes.

- Working arithmetic is binary64. F32 storage widens on load and narrows on
  store; every `Round_f32` stays an explicit narrow-then-widen. No arithmetic in
  `f32`, no fused multiply-add, no reassociation, no approximate or relaxed
  instruction.
- Lanes are independent output cells. A cell's expression, rounding boundaries
  and reduction order are exactly its scalar program's; a reduction is never
  split across lanes or reordered. Vectorizing the reduction axis is out of
  scope.
- First failure, effects and execution marks keep their scalar order. Anything
  with a failure site, a branch the proofs do not cover, a mark, a local array
  or a scan stays scalar around the vector loop.
- `i64` stays modular and `i32` indices stay in the checked domain; a vector
  loop never computes an index outside the range its scalar loop did.

## Representation (`Loop_vector`)

A *vector program* is a scalar `Loop_program.t` in which some `For` loops are
replaced by vector loops. The scalar program is kept beside it: it is the
fallback for every backend and the reference every check compares with.

A vector loop strip-mines `[lo, hi)` into full vectors of `lanes` consecutive
iterations followed by a scalar remainder:

- `lanes` is the *logical* width: how many cells one vector iteration covers.
  It says nothing about registers. A target maps it to however many physical
  registers it takes (four `f64` lanes are two `f64x2` registers on Wasm).
- The body is straight-line vector statements over typed vector temporaries:
  broadcast of a loop-invariant scalar, contiguous or strided element loads
  (`base + k * stride` for lane `k`, stride 0 is a broadcast), lane-wise
  arithmetic, comparisons into masks, selects, explicit `Round_f32`, and stores.
  There are no physical registers, intrinsics or byte widths in it.
- The remainder is the original scalar loop restricted to the iterations left
  over, so a backend that declines vectors (or a length under one vector) runs
  exactly the scalar program.
- Each vector statement names the scalar statement it came from, so a refusal,
  a failure or a mark count can be traced to its source.

## Legality (`Loop_vector_check`, shared)

A loop is vectorized only when the analysis proves, from the program and not
from emitted text:

1. the body is pure and straight-line: float/int64 assignments and stores of
   pure expressions, no `Fail_if`, `Mark`, `Alloc`, array access, scan
   statement, nested loop or untaken-branch load;
2. every access index is affine in the loop variable (`Loop_linear`) with a
   constant stride, the same cells the scalar iterations touch, so no lane
   reads or writes a cell its scalar iteration would not (memory safety is the
   scalar loop's);
3. no loop-carried dependence: a cell stored in one iteration is not read in
   another (same buffer, different index), and no temporary assigned in the
   body is read before it is assigned in the next iteration;
4. loads and stores of distinct buffers do not overlap. Distinct buffer ids are
   disjoint within one invocation by construction (every operand and the output
   are simultaneously live, so the allocator places them apart); the bundle path
   re-checks the placed byte ranges, and a tensor id is never taken as proof on
   its own. A buffer both read and written in the loop must use the identical
   index.

A refusal is a value with a structured reason, tallied by reason; scalar
retention around a loop is not an unsupported-graph fallback.

## Oracle (`Loop_vector_interp`)

Independent of every backend: a vector loop is expanded into scalar statements
lane by lane (lane `k` of iteration `j` is the scalar body with the loop
variable replaced by `lo + j * lanes + k`) and run by `Loop_interp`. Values,
errors and mark multiplicity must equal the original scalar program's, for
extents 0, 1, `lanes - 1`, `lanes`, `lanes + 1` and non-multiples, offset views
and broadcasts. Malformed vector programs, lane-shape mutations and a dropped
`Round_f32` are rejected or caught by the checks.

## Targets (`Loop_target`)

A target description separates *legality* from *cost*. Legality: which
operations, element types and shapes the target can express (with expansion
into scalar lanes when it cannot). Cost: relative per-operation and per-memory-
access costs, so a profitable plan is chosen without making an unprofitable
legal one illegal. Portable `wasm128` is described first; an installed native
target (the host's 128-bit AArch64 NEON) second.
