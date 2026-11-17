# Band schedule, view call and segment driver

Libraries: band_driver, band_schedule
Status: 2026-11-17 — the driver loops over repeating blocks and runs KxK window stages; schedule interpreter, view-call operation with per-row bases, pointwise driver and fuzz generator implemented; Machine IR lowering through the contraction helper, run on the generic, AArch64 and RV32 interpreters

A chain of row-local kernels runs block by block: the last kernel makes `block`
rows, each earlier kernel runs ahead in chunks of its own, and a produced row
stays in its window until the consumer's last use. `band_schedule` is that
schedule as data: row ranges propagated backward, the window each producer
needs (rounded to a multiple of its chunk, so aligned production never wraps a
call), refusals for chains that are not one spine, and `verify` with four
deliberate mutations. It runs nothing and has no target. A logical row, a ring
row (`Window.ring_row`, the one function between them) and a row count are
distinct types; block rows and chunk rows still share `Count`.

## The view-call operation

`Ssa_op.View_call` is `Contract` on views: the descriptor of `Contraction`
evaluated with each operand's cells read from, or written to, one `index` base
past the buffer's first cell (`Ssa_view_call`). The descriptor's strides are
unchanged. The verifier proves the whole footprint, base plus the last batch,
row, column and step, inside the buffer with `Ssa_range.view_in_bounds`, and
refuses (`Unproven In_bounds`) where it cannot; the call then never fails and
checks nothing. A base alone is not a bound. C aliases neither input.

Where it is handled:

| Place | Policy |
|---|---|
| `Ssa_verify`, `Ssa_cfg_verify`, `Ssa_typing` | operand and footprint checks; bases are `index` |
| `Ssa_interp`, `Ssa_cfg_interp` | `Contraction.eval` through the bases; the CFG executor shares the instruction evaluator |
| `Ssa_expand_view_call` | expansion into the loops of `Ssa_expand_contract` with each cell moved by its base; ordered or fused binary32 and ordered bfloat16, as for `Contract` |
| C, JavaScript, Wasm emitters | typed refusal (`` `Unsupported_view_call ``, `Operation.View_call`); expand first |
| Machine IR | contraction helper calls on offset addresses (`Mir_lower`): one call per view call, one per row of the split axis where bases are per row, the descriptor made single there; the helper is the `Contract` one, so every target that realizes it realizes the view call |
| `Loop_of_ssa` | no Loop form; expand first |

## The driver

`Band_driver.program` turns a schedule of stages (a window of `kh` rows by `kw` columns, each
axis with its own stride, padding and dilation in `Band_schedule.Axis.t`; 1x1 is pointwise) into an
ordinary SSA program: for a pointwise stage one view call per event, its bases the element
offsets of the rows the event reads and writes. The first stage reads and the
last writes whole tensors; each intermediate is a ring buffer of its window's
rows. A read that straddles a window's end is refused (`Read_spans_wrap`): one
base cannot name rows that are not consecutive in the ring. `Band_driver.whole`
is the unbanded reference (one `Contract` per stage over complete tensors); the
tests compare the two bitwise under the SSA interpreter.

The driver emits the blocks that repeat as one loop. Events are grouped into
blocks (up to each event of the last stage); the longest run in which
repetition `k` of a period of blocks equals repetition 0 with every base moved
by `k` times a fixed per-base step becomes `For` over the repetitions, the
bases `base + i * step` (a step of zero stays a constant). Ring bases step by
zero once the window rotation has repeated, logical bases by the rows a period
covers. Blocks before and after the run stay unrolled; `Program.loop` names
the run, and `~looped:false` keeps the unrolled list as the reference. The
verifier proves each body footprint over the induction range, so the loop is as
checked as the unrolled form. Two driver mutations (`Step_dropped`,
`Iterations_short`) are red against the whole chain on the SSA interpreter and
on all three Machine IR routes (`test/band_driver`, `test/machine_view_call`).

A ring base that wraps does not step by a constant. It is emitted as
`rest + slot * ((a*i + b) mod slots)`, a slot being the rows one call's
footprint covers (one row for a window read, the produced chunk for a write
into a ring a multiple of it), the modulus built from existing index
operations as `t + (-slots) * floor_div slots t`. `Ssa_range` recognises that
exact shape (the same value in both terms, a positive literal) and bounds it by
`[0, slots - 1]`, which the interval over the three operations cannot give, so
`view_in_bounds` proves the footprint from the slot count alone; a modulus past
the ring is `Unproven In_bounds`. No new operation exists, so both SSA
interpreters, expansion and the Machine IR lowering (`Index_floor_div` already
lowers) carry it unchanged. A multi-row contiguous read that straddles a wrap
differs in shape from one that does not, so the loop is also searched over the
blocks with every multi-row ring read split per row (the form the wrapping read
already takes); it is used only where it covers more blocks, and the blocks
outside the loop keep the single read. Mutations `Ring_modulus_short` (wrong
slot, red on the whole chain) and `Ring_modulus_long` (verifier refuses).

A window stage (`extent` K > 1) makes `width - K + 1` pixels per row; its
reduction is `K * cin` (a window row's K pixels of channels are contiguous, the
output pixels `cin` apart), its weights `(K * K * cin) x cout`. An event is one
call per output row and window row `t`, A at the ring row (or logical row of
the whole input) `r + t`, B at `t * K * cin * cout`, C at row `r`; the calls
after the first have seed `Tensor` and add into C, so the sum runs window row,
then column, then channel. Each call reads one row, so a window never needs
per-pixel bases, and a ring read cannot wrap. `Band_driver.whole` runs a window
stage as the same calls at logical rows over whole tensors (a `Contract` has no
base, so a window is not a single `Contract`); pointwise stages stay one
`Contract`. The third driver mutation, `Window_offset_dropped` (every window
row reads the first), is red on the SSA interpreter and on the three Machine IR
routes.

## Per-row bases

An operand may be placed by one base per row of its outer axis (`At.Rows`):
A's and C's rows, B's columns. The descriptor's outer stride then does not
apply and the cell is `rows.(i) + batch + inner`; the list must be as long as
the axis (a typing error otherwise) and the verifier proves every base's
footprint. The interpreter decodes `Contraction.eval`'s offsets through a
descriptor with unambiguous strides, so the reference loop order stays the one
`Contraction` owns. The expansion unrolls an axis some operand names by rows.
The driver uses it for a read that straddles its producer's window end: one
base per pixel, each from its own ring row, where a single base would run past
the allocation.

Rows are typed: a logical row (`Row`), a ring row (`Window.ring_row` is the one
function between them) and, for a call's row count, `Span` tagged `block` or
`chunk`, so a block cannot stand for a chunk. The driver names a row's
addressing as `Position.Logical` or `Position.Ring`, and element offsets come
from one function of it.

## Evidence

`test/ssa_fuzz` generates programs of view calls (one base or per-row bases,
constant in-bounds bases, inputs exact in binary32) and runs the CFG executor and
the expansion against the interpreter; the C, JavaScript and Wasm suites run the
expansion through each emitter and show the call itself refused.
`scripts/ssa-fuzz-mutation-check.sh ir` breaks the expansion and the evaluator
four ways and requires each to go red.

`test/machine_view_call` lowers the banded driver chains and hand-built per-row
calls (A and C rows, B columns, a single-base operand beside per-row ones,
batched) through `Mir_lower` and runs them on the generic, AArch64 and RV32
interpreters (the contraction helper bound to its model), bitwise equal to the
SSA interpreter, which equals the whole `Contract` chain. Two lowering
mutations (`View_base_dropped`, `View_outer_dropped`) each turn it red. The per-row
form costs one helper call per row of the axis, since a helper takes one address
per operand.

Real targets (`test/band_driver/band_chains.ml` holds the chains; each target
compares its output bits with the whole chain on the SSA interpreter):

- AArch64, natively on the host CPU (`test/machine_rivet_aarch64`, built only on
  an AArch64 host): the banded program is selected, allocated and published, and
  its Rivet image run in process. The route generates no contraction code, so
  each descriptor is host C (ordered binary32, no contraction) bound to the
  artifact's `mlt_contract_*` symbol by a trampoline. What runs on the CPU is
  the driver: its loops, every base's address arithmetic and the calls. Pointwise
  and 3x3 chains, unrolled and looped, are equal. The driver's three mutations
  and two native-code mutations (`Branch_sense`, `Dropped_lo12`) are red.
- RV32 under qemu (`test/machine_rivet_riscv32_mir`): the pointwise and 3x3
  chains, unrolled and looped, are equal with the contractions as generated RVV
  functions. A window chain adds into its output (seed `Tensor`): the RVV
  function starts each row group from C's cells (a vector load in place of the
  zeroing; refused for an integer product and for C carried by the DMA), and
  the matrix engine's binary32 function loads the tile from C's rows (`vtle32`).
  A `Seed_zeroed` mutation of each is red (qemu, and the VME RTL).

## Open

- An AArch64 contraction realization (so the contraction itself runs as
  generated AArch64, not host C).
- (Settled, unrolled is right.) A chain with fewer than two repetitions of any
  block run (for example `block 3, 8 rows`) stays unrolled: a loop reuses one
  body across repetitions, so with no two blocks of one shape and affine bases
  a loop would be one trip, the same view calls behind an induction value. The
  driver's sweep (blocks 1 to 6, rows 1 to 24, a pointwise and a 3x3 chain)
  has every unrolled case equal in view calls to the loop form and every
  looped case strictly smaller; forcing a one-trip loop on the unrolled cases
  turns that test red. The ring rotation itself no longer needs a period.

### Stride, padding and dilation

A stage with any of them is one view call per output row and window position,
over the output pixels that position reaches; the plain stride-1 window keeps
one call per row over `extent` contiguous columns. A position in the padding is
skipped, not added as a zero (the semantic table's convolution row: a zero
would turn an infinite weight into NaN), so rows outside the producer are never
read, a column's calls cover only the pixels whose input column is inside, and
a pixel's first call (not the first call of the row) seeds with zero while the
rest add into it. A pixel with no position in the input is refused
(`Empty_window`). The consumed ranges are the schedule's clamped ranges.
`Padding_read_edge` (a padded row read from the nearest producer row) and
`Stride_dropped` are red in the driver and view-call suites; a nested binary32
sum that skips padded taps equals `whole`.

### Non-square windows and per-axis parameters

A kernel carries two `Band_schedule.Axis.t` (extent, stride, pad, dilation):
`rows`, the axis the schedule bands over, and `cols`, which only the driver
reads. A plain window (column stride 1, no column padding or dilation) is `kh`
row calls each reducing `kw * cin`; its weights are `(kh * kw * cin) x cout`
with position `ky * kw + kx`. The row axis only moves which rows a call reads,
so a stage is pointwise (one call per event over every pixel) only when both
axes are 1 with no row stride or padding; a 1x1 with a column stride is a
windowed stage. `Axes_swapped` and `Padding_axes_swapped` are red against an
independent nested-sum oracle (the band-driver test library's `oracle`), on the
SSA interpreter and the generic, AArch64 and RV32 Machine IR routes; the chains
cover 1x3, 3x1, 3x5, mixed strides, and a row dilation with a column stride.
