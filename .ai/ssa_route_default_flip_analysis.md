# Structured route as the default: analysis

The structured route was first withheld as the default for a cycle regression,
then made the default once that was fixed. The text below records both steps.

## What the flip changes (measured)

Seven committed goldens change; each is attributable to the route:

- `coral_plan` plan test: the per-instruction `; value bN [...]` annotation is
  absent (the structured route does not attach the logical value), and an
  unused `[1x32]` scratch buffer of the contraction plans is gone.
- `machine_rivet_aarch64` manifest: two 8-byte bss regions become one 16-byte
  region, following the dropped scratch; the planning subject hash changes.
- `machine_model` census/scan/schedule and `machine_rivet_riscv32_mir` census:
  softmax 534 to 318 and layer_norm 609 to 139 RV32 instructions (sdpa
  aarch64 516 to 127); chain/2 grows 27 to 30 (aarch64) and 36 (x86_64), conv
  +2. The mutation `hole` is caught where it was not, and `flags` mutations are
  caught on fewer kernels (4 to 2 aarch64, 4 to 3 x86_64), a weaker check.
- `ssa_wasm` census: chain memory_move 73 to 72.

## Why it is withheld

Re-running the Coral VME RTL calibration on the structured-route binaries
(`make coral.estimate.calibrate`) shows a cycle regression, not only a shift:

- head split (invocation 0 of 16x64 and 32x128 splits): 173009 to 228305
  cycles (+32%), instret 108328 to 145192; 731737 to 957025 (+31%);
  567288 to 736632 (+30%). The decomposed layer's matching invocation is +23%.
- Faster: softmax over C (-2%), norms (layer norm -5.6%, rms norm -3.7%), and
  the decomposed layer's large invocation (-17%).

A route that is slower on the estimator's own target for the view/permute
copy kernels is a defect to fix in the route, not to absorb by refitting the
estimator; the accuracy limits stay as they are. The estimator fit (committed
rates against Direct-route data) is unchanged and passes under Direct.

## Cause and fix

The index arithmetic was identical in both routes. The difference was one
`check_access` per element: the Loop guard of a load becomes an access check
the converter writes ahead of the load, and the load repeats the same check.
The range analysis cannot prove the floor-division coordinates of a reshape
plus permute in bounds, so the pair survived planning and ran per element.
`Ssa_opt_guards` now drops a `Check_access` whose effect the next effectful
operation consumes when that operation is a load of the same buffer at the same
coordinate (it fails exactly as the load alone does). Re-running the calibration
on the structured route: head splits equal the Direct cycles, no row is slower,
and the norm/softmax gains remain (layer norm, rms norm, softmax and the
decomposed layer's large invocation faster); the only other change in the table
is the dropped scratch buffer in the `ram` lists.

## Outcome

The flip landed: `Ssa_backends` defaults to `Frontend.Structured`; `Direct`
stays selectable as the comparison route. The goldens above were promoted, and
the estimator was recalibrated on structured-route kernels and refit with every
accuracy limit unchanged. Held-out errors under the refit rates: linear 32x96
k48 +1.5% (limit 2%), softmax over C +0.2%, sdpa -9.6%, bare layer -10.2%; the
decomposed layer's out-of-gate error moved from -28.6% to -12.4%. In the
machine-model goldens `flags` drops are caught on fewer kernels (4 to 2
aarch64, 4 to 3 x86_64) because the softmax, sdpa and norm kernels shrank to
a few dozen instructions, while `data` (7) and `order` (6) are unchanged and
`hole` is now caught; the aarch64 sink schedule has no moves left on softmax
and sdpa, which x86_64 still exercises.
