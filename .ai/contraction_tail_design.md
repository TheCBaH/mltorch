# Contraction with attached pointwise operations

Libraries: contraction_tail
Status: 2026-11-16 — descriptor and the generic realization (block scratch plus
tail loop) implemented; absorbing-engine realization and SSA composition open

A product's tail (bias, residual add, scale, clamp, activation) belongs to its
descriptor, not to a separate kernel. `contraction_tail` extends a
`Contraction.t` with an ordered list of pointwise steps over the result. Each
step reads the running value and, for `Add`/`Mul`, a named side buffer at the
output's own coordinate through element strides (0 broadcasts an axis); each
result is rounded to binary32 where it is defined. An intermediate the outside
needs is a `Tap`, a second output of the descriptor. The numerical requirement
is the core's `numerics`; the tail adds none. An integer product is refused
(its result is not binary32), as are negative strides, NaN or inverted clamp
bounds and a tap named twice. The refusals are variants with named fields.

## Realizations

- `eval_reference`: the unattached form. The whole product (`Contraction.eval`),
  then the steps over every output.
- `eval_scratch`: the generic realization. The output is cut into blocks of at
  most `Block.rows x Block.cols` (edge blocks smaller); each block is a
  sub-contraction into compact scratch (seeded from C when the seed is
  `Tensor`), then a tail loop over the scratch stores to C's offsets and the
  taps. `Block.make` bounds the cells by a division, so the product is never
  formed past the limit and a cell index narrows to `int` safely.

Both give the same bits for every legal block under every numerics: each
output's sum is untouched by blocking and the steps are per-element. The
tests compare them with a third oracle written without the library's tail
code, over blocks that divide and do not divide the output, and show the
comparison red under three deliberate defects (`Fault`: seed from scratch,
side operands read at block-local coordinates, intermediates left unrounded)
and under a source mutation of the seed read.

## Why no new SSA operation yet

A descriptor becomes an operation only when an engine takes it as one opaque
call or its reference order must survive lowering. The generic realization is
the existing `Contract` (or `View_call`) into scratch followed by ordinary
loops, so it needs no operation, verifier or interpreter case. An absorbing
engine would, and is the next step: it brings the operation, its verifier rule
and its interpreter, and is checked against `eval_reference` for the same
bits.
