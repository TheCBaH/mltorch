# Loop IR and structured SSA — roles, parity gates, retirement

Libraries: loop_ir
Status: 2026-11-12 — role rule and keep-set measured from source; the owner confirmed the retirement decision (§5) the same day

Two middle-ends sit below `Fusion_plan`: the Loop IR (`lib/loop_ir`) and the
structured SSA IR (`lib/ssa_*`, see the SSA design). This record fixes which one
owns what, which gates must pass before the older one shrinks, and what can then
go. The owner confirmed the retirement decision on 2026-11-12 (§5.2). The role
rule (§1) applies to new work now, because it costs nothing.

Everything below was read from source, `git log` and the Makefile; nothing in this
record was produced by a run unless it says so.

## 1. Role rule

**Loop IR is the scalar, policy-free kernel form: its lowering and its
interpreter, plus the runtime and failure layer the backends share.** SSA owns
everything after it: numeric policy, the vector form, engine recognition and
selection, the segment driver, the emitters and Machine IR. Tiling is decided in
the segment plan, above both IRs.

**Loop IR grows only in the scalar lowering contract.** No numeric policy, no
vector form, no engine operation and no tiling is added to `lib/loop_ir`; a change
that needs one of those lands in SSA only. This ends paying twice for the same
change (§4).

The placed-graph lowering has one implementation, `Loop_lower`. It reaches SSA
through `Loop_lower.lower_structured` and `Ssa_of_loop.convert`
(`lib/ssa_bridge`), before precision planning, vectorization and contraction
recognition. `lower_structured` is the unoptimized form
(`lib/loop_ir/loop_lower.ml`); it never calls `Loop_opt.run`, which would drop the
`Ordered_sum` an expanded accumulator loop needs. Today `Ssa_of_loop` is exercised
by tests only: production SSA lowering is still `Ssa_lower_plan`, direct from the
fusion plan. Wiring the route is open work (§6).

## 2. Keep-set and retirable set

`lib/loop_ir` is ~130 files. By role (line counts approximate):

| Part | Modules | Fate |
|---|---|---|
| Runtime and failure layer | `loop_failure`, `loop_js_failure`, `loop_wasm_failure`, `loop_index`, `loop_numerics`, `loop_target`, `loop_buffer`, `loop_program`, `loop_plan`, `loop_c_runtime`, `loop_js_runtime`, `loop_wasm_runtime`, `loop_wasm_ctx`, `loop_wasm_link`, `loop_c_base` (~3.3k lines) | keep; the SSA emitters use it |
| Bundle layer | `loop_bundle`, `loop_bundle_{c,js,wasm}`, `loop_bundle_oracle`, `Loop_bundle_exec` (~1.6k) | keep; `Loop_bundle.build_with` is what the SSA backends build on |
| Scalar lowering contract | `loop_lower*`, `loop_stmt`, `loop_expr`, `loop_bool`, `loop_carrier`, `loop_mark`, `loop_stored`, `loop_array`, `loop_var`, `loop_temp`, the interpreter `loop_interp`, `loop_check` | keep, by design |
| Optimizer | `loop_opt*` (~1.4k) | retirable |
| Vectorizer | `loop_vector*`, `loop_vectorize`, `loop_block`, `loop_c_vector`, `loop_wasm_vector` (~2k) | retirable |
| Emitter bodies | `loop_c`, `loop_js`, `loop_wasm` | retirable after the shared symbols move out |

The keep-set is roughly two-thirds of `loop_ir` by lines. Consumers outside the
library, by what they reach into:

- `ssa_c`, `ssa_js`, `ssa_wasm`: the runtime layer, plus a handful of
  emitter-module symbols — `Loop_c.source`, `Loop_js.function_name`,
  `Loop_wasm.func` and `Loop_wasm.lower_with`.
- `ssa_backends`: the bundle layer, a precision field of `Loop_c`/`Loop_wasm`, and
  `Loop_vectorize.tally`.
- `ssa_bridge`: the scalar Loop form (`Loop_stmt`, `Loop_expr`, `Loop_bool`,
  `Loop_carrier`, `Loop_interp`, `Loop_lower`) — the converters and the
  differential harness.
- `ssa_ir`, `ssa_lower`: nothing; they are independent of `loop_ir`.

**Precondition for separating the emitters from the runtime:** move
`Loop_c.source`, `Loop_js.function_name`, `Loop_wasm.func`/`lower_with`, the
precision field and `Loop_vectorize.tally` into the shared layer. After that the
emitter bodies, optimizer and vectorizer depend on the runtime and nothing depends
on them except the Loop-owned backends.

`Loop_opt` is needed only by the Loop-owned backends (the interpreter-driven
executors, the Loop C/JS/Wasm paths, the Model Explorer "generated JS" export,
which exposes the pass list). Every pass has an SSA counterpart that runs after
the conversion: unit-loop, fold and simplify in `Ssa_opt.simplify`/`guards`,
guards in `Ssa_opt.guards`, CSE in `simplify` and `share`, hoisting in `hoist`,
collapse in `Ssa_opt_collapse`. Retiring Loop's emitters retires its production
consumers, so the optimizer goes with them.

## 3. Parity gates

Retirement is conditional on four gates. Each names the make target that proves it
(every target below exists in the Makefile) and the gap where one is missing.

| Gate | Question | Proof today | Gap |
|---|---|---|---|
| Capability | Does SSA run everything the Loop path runs? | `c.pt2.ssa.runtest`, `wasm.pt2.ssa.runtest`, `loop_js.bundle.pt2.ssa.runtest`; Native4D: `c.pt2.native4d.ssa.runtest`, `wasm.pt2.native4d.ssa.runtest`, `loop_js.bundle.pt2.native4d.ssa.runtest` (all bitwise against the reference, pipelines `exact` and `planned`) | per-node JS path and Region ops through SSA JS, §5.3 |
| Numerical | Bitwise at the binary64 reference; within tolerance at relaxed binary32 | strict: the `--reference --strict --shadow` runs above; relaxed: `c.pt2.ssa.perf`, `wasm.pt2.ssa.perf` (`--shadow-numeric`, 1e-4 absolute and relative); `fp32.bench`, `fp32.bench.wasm` | the contract should be stated as "SSA against the binary64 reference", not "SSA against Loop's relaxed output" |
| Failure behavior | Same failure row, site and limit | shared `Loop_failure` and `Loop_js_failure`; unit tests in `test/ssa_bridge` | **no whole-model or per-kernel failure-parity target.** `Ssa_of_loop` refuses a guard outside the recognised `Fail_if` pair, so the converter route cannot yet carry failures in general |
| Performance | SSA planned is no slower than the Loop default, within a bound | `c.pt2.bench`, `wasm.pt2.bench` (Loop only: neither passes `--ssa`) | **no SSA-against-Loop timing target.** Add `--ssa=planned` to the benches, record both columns, and define the bound (for example at most 1.05x the Loop warm run per model) |

The int64 and Bool evaluator work, and a Region-authored operation such as the
LSTM, were paid in neither middle-end: they were changes to `expr`, `kernel` and
`native`, and both lowerings consume `Expr.Value.t`. What is paid twice is a
numeric policy, a dtype or store format, or a new IR primitive (§4).

## 4. What was paid twice (sampled from history)

Lines changed, tests included; read by topology, because author dates do not
follow dependency order.

| Change | Loop | SSA |
|---|---|---|
| binary32 `Fp32_ordered` numerics policy | 490 lines, 11 files | 237, 9 |
| bfloat16 store encoding | 175, 26 | 206, 16 |
| int64 and Bool evaluator work (eight commits) | 0 | 0 |
| Region-authored op (LSTM on the scan primitive) | 0 | 0 |

Part of the apparent duplication is sharing that already works: the binary32
transcendentals landed mostly in the shared runtime text, so the SSA side paid
little. The saving from retiring the Loop emitters is therefore the Loop-only
tail — the bundle oracle, per-emitter runtime text, vector expansion, `loop_block`
— and the duplicated validation, where each gate runs twice. The "about one
policy or dtype change a week" frequency is an impression from the log, not a
measured rate.

## 5. Decision

### 5.1 Retirement

Freeze the Loop emitters, optimizer and vectorizer now: no new files matching
`loop_vector*`, `loop_opt*`, `loop_block*` or `loop_{c,js,wasm}_vector*`, and no
behavior change in them beyond keeping them passing. Retire them once all four
gates in §3 have passed twice in a row in CI for every model in `PT2_MODELS_CRAM`
on both architectures. The scalar form, `loop_interp` and the shared runtime stay
until they are renamed and moved (a `loop_runtime` library shaped like `ssa_c`).

Retirable then: `loop_opt*` and the Model Explorer pass-selection surface (which
needs its own decision: keep the export on the interpreter path or drop the pass
selector), `loop_vector*`, `loop_vectorize`, `loop_block`, `loop_c_vector`,
`loop_wasm_vector`, the emitter bodies of `loop_c`/`loop_js`/`loop_wasm`, and the
Loop-only gates they carry: `c.pt2.runtest`, `wasm.pt2.runtest`,
`loop_js.bundle.pt2.runtest` and the per-node tier `loop_js.node.pt2.runtest`.

### 5.2 Are the Loop emitters an independent oracle? Decided: no, retire them

- **Keep them, frozen.** An independently written C/JS/Wasm emitter to check
  SSA's. Cost: every model gate stays twice, the Loop-only tail stays maintained,
  and the optimizer and vectorizer stay as code that must keep passing.
- **Retire them after the gates pass (chosen, confirmed by the owner 2026-11-12).** The oracle role is
  already carried by evaluators that use no Loop emission: `Eval_direct` and
  `Eval_direct4` (the `--reference` every strict SSA gate compares to bitwise),
  `Kernel_eval` and `Loop_interp`. And the two emitter families are not
  independent where it matters: both use `Loop_c_runtime`, `Loop_js_runtime`,
  `Loop_wasm_runtime`, `Loop_failure` and `Loop_numerics`, so a diff between
  emitters cannot catch a runtime bug.

Revisit only if a code-generation bug appears that `Eval_direct` equality missed
and Loop emission would have caught.

### 5.3 Capability unknowns

- **Per-node generated JS** (`loop_js_pt2 --nodes --shadow`, run by
  `loop_js.node.pt2.runtest`): `ssa_js_pt2` has no per-node mode, so this is a
  missing implementation, not a typed refusal. Decide whether it is needed once
  the bundle path is at parity; its job is to localise a bad node, which a
  per-invocation `--shadow` could do.
- **Region-authored ops through SSA JavaScript:** unknown. `ssa_lower_region`
  exists and C and Wasm have region bundle tests under `test/ssa_c` and
  `test/ssa_wasm`; there is none under `test/ssa_js`, and the SSA JS gate runs
  `JS_PT2_MODEL`, a model with no Region ops. One manual run of `ssa_js_pt2` on
  `fastvit_sa12` with `--shadow --reference --strict` settles it. If the JS emitter
  refuses, its payload is an untyped `Unsupported_operation of string`, which
  should become a variant so the gap is countable.
- **Vectors and binary32 in JavaScript:** a typed refusal on both sides (parity).

## 6. Follow-up work this record implies

1. Move the shared symbols named in §2 out of the emitter modules.
2. Wire `Loop_lower.lower_structured` + `Ssa_of_loop` behind a parity gate and
   cover the failure and binding classes; `Ssa_lower_plan` cannot be dropped
   before that.
3. Run `ssa_js_pt2` on `fastvit_sa12`; add the `test/ssa_js` region bundle test;
   type the JS refusal.
4. Add the failure-parity and SSA-against-Loop bench targets.
5. ~~A check that the frozen file patterns in §5.1 gain no new files~~ Done:
   `make check.loop-freeze` (part of `make precommit` and CI) fails on a file
   matching the patterns that is not in `scripts/loop-freeze-files.txt`, and on a
   listed file that no longer exists, so the list only goes down.
