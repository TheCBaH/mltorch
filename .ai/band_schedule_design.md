# Band schedule, view call and segment driver

Libraries: band_driver, band_schedule
Status: 2026-11-16 — schedule interpreter, view-call operation and a pointwise
driver implemented; per-row bases, target realization and the fuzz generator
open

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

## Open

- A per-row base list form for an operand (a KxK consumer reading rows that
  wrap), with the footprint proved per base, and the driver's use of it.
- Row bases and block rows as typed values end to end (block rows and chunk
  rows as one phantom-role family).
- Realization on the three machine targets through the `Mir_helper` route; the
  generator of the SSA fuzz suite and its mutation script gain the operation.
- Non-pointwise stages and the driver as a loop over uniform blocks, not an
  unrolled event list.
