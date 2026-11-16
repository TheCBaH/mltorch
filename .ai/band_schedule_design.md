# Band schedule, view call and segment driver

Libraries: band_driver, band_schedule
Status: 2026-11-16 — schedule interpreter, view-call operation with per-row bases,
pointwise driver and fuzz generator implemented; target realization open

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
| Machine IR | typed refusal (`Refusal.View_call`): the helper route takes whole buffers |
| `Loop_of_ssa` | no Loop form; expand first |

## The driver

`Band_driver.program` turns a schedule of pointwise stages (extent 1, stride 1)
into an ordinary SSA program: one view call per event, its bases the element
offsets of the rows the event reads and writes. The first stage reads and the
last writes whole tensors; each intermediate is a ring buffer of its window's
rows. A read that straddles a window's end is refused (`Read_spans_wrap`): one
base cannot name rows that are not consecutive in the ring. `Band_driver.whole`
is the unbanded reference (one `Contract` per stage over complete tensors); the
tests compare the two bitwise under the SSA interpreter.

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

## Open

- A KxK stage (several rows read per output row, each with its own bases) in
  the driver; stages that are not pointwise.
- Realization on the three machine targets through the `Mir_helper` route.
- The driver as a loop over uniform blocks, not an unrolled event list.
