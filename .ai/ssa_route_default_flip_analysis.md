# Structured route as the default: analysis, flip withheld

`Ssa_backends` still defaults to `Frontend.Direct`. Flipping the default to
`Frontend.Structured` was tried and reverted.

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

## Next step

Find what the structured lowering emits for the head-split copy (extra index
arithmetic per element is the first suspect: instret rises with cycles), fix it
in `Ssa_of_loop`/the planner, then repeat the flip: promote the seven goldens,
rerun calibration, refit, check the held-out limits (linear 32x96 k48 was
-7.57% against the Direct-calibrated rates, limit 2%).
