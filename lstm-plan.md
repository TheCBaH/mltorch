# `lstm.input` implementation plan

Status: planned, expanded 2026-09-05. No LSTM implementation or support claim is made
by this document. It continues [todo-ops.md](todo-ops.md),
[ops-progress.md](ops-progress.md), and [lstm.md](lstm.md).

## 1. Decision and corrections to the investigation

Implement a first-class, three-output `Lstm` operation for stacked inference,
with optional biases, one or two directions, and either input layout. Accept
valid dropout probabilities during inference, where dropout is inactive.
Sequencer2D is a regression fixture, not the limit of the accepted domain.
Author the computation as a Region
program with an explicit ordered **vector-state scan**. Add the corresponding
Native4D operation in the same vertical slice. Land and test the scan machinery
before adding LSTM arithmetic.

The initial investigation correctly identifies that `sum`/`max_reduce` cannot
express an arbitrary recurrence by changing their bounds. Four details change
the implementation plan:

1. LSTM carries two **vectors**, each of hidden width `K`. Every new hidden
   component reads the entire previous hidden vector through `W_hh`; two
   scalar accumulators per output channel are insufficient.
2. Region-authored operations already have one numeric definition shared by
   Native Direct, Native Symbolic, Native4D, and Kernel execution. See
   [region_compute_design.md](../.ai/region_compute_design.md),
   `lib/native/region_computation.ml`, and `region_execution.ml`. LSTM should
   extend this path; it does not need a second handwritten Pixel algorithm or
   an obligatory new member of `SEMANTICS`.
3. `Aten_shape` right-aligns rank-3 tensors onto `H/W/C`. For batch-first
   input, sequence length is on **W**, not the physical `Axis.T`. All accepted
   operands and results have `N=T=D=1`. The recurrence does not establish an
   intrinsic Native4D axis boundary. Rejecting all LSTMs there would leave a
   missing counterpart, contrary to the breadth contract.
4. Static unrolling is possible in principle: symbolic time can select among
   unrolled states, and existing Region vector locals can share intermediate
   states. The objection is expression growth, specialization cost, and lack
   of an explicit recurrence contract, not mathematical impossibility. Choose
   a compact scan whose AST size does not grow with sequence length or width.

Also correct the schema description: `output` contains the **last layer's**
hidden states at every timestep, not hidden states from every layer.

## 2. Accepted domain and tensor contract

Support `aten.lstm.input` with positive `num_layers=Q`, either value of
`has_biases`, `bidirectional`, and `batch_first`, finite `dropout` in `[0,1]`,
and `train=false`. Write `R=2` for bidirectional and `R=1` for forward-only;
`R` is a direction count, never the Native frame's `Axis.D`.
Require positive, static batch `B`, sequence length `L`, input width `I`, and
hidden width `K`; rank-3 input; two rank-3 initial states; `Q*R*P` parameters,
where `P=4` with biases and `P=2` without;
and matching F32 operands at the ATen boundary. Initial states and parameters
may be ordinary runtime operands; do not require zeros or captured constants.
Native storage formats remain governed by the existing tensor contract.

Reject packed `lstm.data`, unbatched rank-2 input, projections, training,
unsupported dtypes, empty dimensions, nonpositive layer counts, invalid dropout
probabilities, and malformed parameter/state lists with typed errors. Validate
raw ATen ranks before right-alignment erases that information. Do not impose
an arbitrary one-layer ceiling: use the ordinary checked size/resource limits.

`RNN.cpp:apply_layer_stack` applies dropout only when `dropout_p != 0`,
`train`, and the layer is not the last. Thus all accepted probabilities are
numerically inactive in this inference domain. Validate them, then normalize
them out of the Native compute payload; preserve the imported call's flags in
provenance. This is an explicit inference equivalence, not omitted stochastic
behavior. The finite `[0,1]` validation follows the public RNN configuration
contract in `torch/nn/modules/rnn.py`; the low-level ATen fallback can bypass
that validation when dropout is inactive, so test and document this deliberate
invalid-input boundary. Training and RNG state remain separate future work.

| Tensor | ATen dimensions | Native non-unit frame positions |
| --- | --- | --- |
| `input`, batch-first | `[B,L,I]` | `H=B, W=L, C=I` |
| `input`, time-first | `[L,B,I]` | `H=L, W=B, C=I` |
| `h_0`, `c_0` | `[Q*R,B,K]` | `H=Q*R, W=B, C=K` |
| each `weight_ih`, layer `q` | `[4K,I_q]` | `W=4K, C=I_q` |
| each `weight_hh` | `[4K,K]` | `W=4K, C=K` |
| each bias, when present | `[4K]` | `C=4K` |
| `output`, batch-first | `[B,L,R*K]` | `H=B, W=L, C=R*K` |
| `output`, time-first | `[L,B,R*K]` | `H=L, W=B, C=R*K` |
| `h_n`, `c_n` | `[Q*R,B,K]` | `H=Q*R, W=B, C=K` |

Here `I_0=I`, and `I_q=R*K` for every later layer. Initial and final states
always use `[layer*R+direction,batch,hidden]`, regardless of `batch_first`.

Use an ordered collection of layer records, each with one forward parameter
record and a reverse record exactly when bidirectional. Each direction owns
`weight_ih`, `weight_hh`, and an optional **pair** `(bias_ih,bias_hh)`; one
missing bias within a pair is invalid. Decode parameters layer by layer,
forward before reverse, and within each direction in the order
`weight_ih, weight_hh[, bias_ih, bias_hh]`. `RNN.cpp:gather_params`, `pair_vec`,
and `apply_layer_stack` establish this ordering. An absent pair omits the bias
additions; it need not allocate zero tensors or introduce floating-point `+0`.

The Native payload holds these records, a batch-first/time-first layout tag,
and `input`, `h_0`, `c_0`. Derive `Q`, `R`, and bias presence from validated
records, checking them against the imported flags. Require a uniform direction
count and bias policy across layers. Keep output ordinals
`0=output`, `1=h_n`, `2=c_n` everywhere.

`Graph_shape` must validate every operand, all unused frame axes, state widths
and batch sizes, every layer/direction, first-versus-later input widths, and
checked products such as `4*K`, `Q*R`, `Q*R*P`, and total trace size.
Importers share these semantic checks after their own decoding.
Preserve all three Native outputs even when the corpus discards two; implement
live state outputs now, since they are projections of the same recurrence.
Only actual use analysis may route an output to `Discard`; names containing
`unused` are not proof of deadness.

The checked-in corpus was re-read for this plan: 36 nodes, with 28 instances
of `(B,L,I,K)=(16,16,384,96)` and 8 of `(32,32,192,48)`. All have `Q=1`,
`R=2`, biases, batch-first layout, `dropout=0`, and `train=false`.
Synthetic ATen fixtures must cover the expanded domain. The existing support
record still stops Sequencer2D at `lstm.input`.
Equal corpus batch/sequence extents make a `B != L` fixture essential.

## 3. Foundational primitive: a bounded vector scan

Add a generic scan descriptor to `Expr`, separate from the associative
`Reduction.kind`. A proposed logical interface is:

```text
scan(width, steps,
     init(lane),
     update(step, lane, previous_at)) -> trace

trace[0, lane] = init(lane)
trace[s+1, lane] = update(s, lane, trace[s, :])
```

`width` and `steps` are checked static extents. `step` and `lane` are bound
position indices. `previous_at(index)` reads any component of the previous
vector. All lanes at a step observe the same completed previous snapshot;
there are no reads of partially updated state. Evaluation uses separate old
and next buffers. Initialization has no access to the previous-state binder.
Zero steps returns the initial vector at the primitive level, even though the
initial LSTM domain rejects empty sequences.

The trace has `(steps+1)*width` entries, indexed as `row*width+lane`.
For LSTM use width `2K`, packing hidden state in `[0,K)` and cell state in
`[K,2K)`. This expresses two coupled vectors with one generic vector primitive.
The step body uses `k = lane mod K`; both branches of any strict selection
must use valid coordinates. Compute the new cell expression from the previous
snapshot and reuse it in the new hidden expression; never read a just-written
cell slot from `previous_at`.

Represent the descriptor as a closed, inspectable AST with builder-minted
step/lane/previous-state binders. An existing `Local_at` leaf may represent
previous-state reads if the checker binds its local ID only inside the update;
it must not be treated as a free Region local or tensor source. Add a scalar
`Scan_at(descriptor, row, lane)` projection for symbolic specialization and
reference evaluation. Numeric operations remain ordinary `Expr` operations.

Required semantic bundle:

- Define initialization, increasing step order, simultaneous vector update,
  bounds, invalid reads, checked allocation sizes, and finite resource limits.
  A generic scan does not implicitly round each state write; explicit
  `Round_f32` nodes in the authored body define such rounding when required.
- Extend `expr_repr`, the public `expr_api`, smart builders, interpreter,
  printer, comparison/hash, folding, and capture-safe rewriting. Follow the
  existing binder identity and lexical printing conventions. Substitution
  must freshen every scan binder and preserve references to outer reductions
  or locals without capture.
- Extend well-formedness to reject escaping previous-state references, bad
  scope, illegal dimensions and trace indices, and malformed descriptors.
  Statically validate what can be proven and retain checked runtime reads for
  computed indices. Include all initializer/update tensor sources in traversal
  and conservative dependency summaries.
- Mark scan order as semantically significant. No tree reduction, reordering,
  or reassociation is legal. Structural equality can compare matching scan
  descriptors; it must not claim algebraic equivalence of different recurrences.

Test this primitive independently with a noncommutative scalar recurrence and
a coupled vector update such as `next=[previous[1]+1, previous[0]*2]`.
The latter detects accidental in-place updates and per-lane scalar folding.

## 4. Region integration and production execution

Add a scan-backed local constructor alongside existing scalar/vector locals.
It owns a descriptor and a flat trace reader. This requires an explicit local
RHS/constructor extension: today's `Region_local.value : Expr.Value.t` plus
independent `Vector` element loop does not implement scan sharing by itself.
Do not simply fill a vector by repeatedly evaluating `Scan_at`.

`Region_program.Builder.scan` allocates the trace local and supplies a
`read(row,lane)` callback to subsequent locals/the emitter. `Region_execution`
runs the descriptor **once per region key**, stores its trace, and makes reads
constant-time. Factor the numeric scan interpreter so scalar `Expr.Eval`
projection and Region trace materialization obey one recurrence contract.
`Region_eval`/`value_at` may run a fresh scan for one requested point; production
tensor materialization must use the shared trace path.

Extend `Region_program.check`, local dependency order, source/binder folds,
partition invariance, resource accounting, printing, specialization, rewriting,
execution, and tracing. `specialize_pixel` substitutes a trace-local read with
`Scan_at`; it must preserve the compact descriptor instead of unrolling it.
Update every consumer of `Region_local.shape/value` for the new constructor.
Keep existing scalar and vector execution unchanged in behavior.

Use one Region program per output ordinal, as Stage Program and Kernel already
do. Multi-output Region programs and cross-stage scan caches are deferred:
computing separate scans for the three outputs is an explicit constant-factor
cost, not a reason to redesign Kernel values in this slice.

- For `output`, partition with `Whole(W,C)` in batch-first layout and
  `Whole(H,C)` in time-first layout. Batch remains invariant. Each batch key
  runs `Q*R` scans, declared in layer order, then emits the last layer's
  timesteps and concatenated channels.
- For `h_n`/`c_n`, partition with `Whole(H,C)`, leaving batch `W` invariant.
  Run the same `Q*R` scans once per batch and emit the final state of every
  layer/direction. Do not partition separately by layer and recompute all
  preceding layers for each emitted state; that would add quadratic work in Q.

For layer zero, the scan input reader loads the original input using the layout
tag. For every later layer, it reads the previous layer's completed hidden
traces in original time order, concatenating forward then reverse features.
Both directions of layer `q-1` must finish before either scan of layer `q`
runs. This is a dependency between ordinary earlier Region locals and later
scan bodies, not a new tensor edge or a decomposition of `Graph_ir.Lstm`.
The scan checker and executor must support these captured trace-local reads.

Each scan body has size independent of `L` and `K`; the authored program grows
linearly with `Q*R`. Production execution retains references to earlier traces.
Scalar specialization must respect size/depth limits when substituting those
dependencies; it must not recursively duplicate a layer's entire computation
without a bound. Test two- and three-layer specialization and typed limit
outcomes separately from the efficient materialization path.

Expected work per output program is
`O(B*R*L*K*((I+K)+(Q-1)*(R*K+K)))`, up to a documented constant factor
for gates and packed state. Initial trace storage per batch key is
`O(Q*R*(L+1)*2K)`; retaining traces for all layers makes every final state
available. Lifetime-based release of intermediate traces can be optimized later.
Count scan starts, steps, state elements, and loads in tests. Increasing emitted
channels must not multiply scan starts; increasing `L` must not grow the AST.
Reject resource-limit violations explicitly, without a slow scalar fallback.

## 5. LSTM arithmetic and output indexing

Let `q` be layer, `s` scan step, and `d` direction. Read the layer's input at
original time `t=s` for forward and `t=L-1-s` for reverse. Initialize from
`h_0[q*R+d,b,:]` and `c_0[q*R+d,b,:]` independently for each layer/direction.
Use the layout tag only to address the external sequence tensors; internal
trace indexing and the state layout are the same for both layouts.

For gate row `r` use the ATen order `i,f,g,o`:

```text
a[r] = (sum_{j<K} W_hh[q,d,r,j] * previous_h[j] [+ b_hh[q,d,r]])
     + (sum_{j<I_q} W_ih[q,d,r,j] * layer_input(q,b,t,j) [+ b_ih[q,d,r]])
i = sigmoid(a[k]);       f = sigmoid(a[K+k])
g = tanh(a[2K+k]);        o = sigmoid(a[3K+k])
next_c[k] = f * previous_c[k] + i * g
next_h[k] = o * tanh(next_c[k])
```

Square brackets above mean the additions exist only when biases are present.
`layer_input(0,b,t,j)` reads `input[b,t,j]` or `input[t,b,j]` according to the
layout. For `q>0`, set `d'=j/K`, `k'=j mod K` and read layer `q-1`'s hidden
trace at row `t+1` for `d'=0`, or `L-t` for `d'=1`, lane `k'`. This restores
original time order before passing either direction's output to the next layer.
With `R=1`, instantiate only the forward reader; there is no reverse operand
to read, even in an unselected strict expression branch.

For output coordinate `(b,t,c)` or `(t,b,c)`, set `d=c/K`, `k=c mod K`.
Read layer `Q-1`'s forward trace row `t+1` or reverse trace row `L-t`,
hidden lane `k`. For state output leading index `a`, set `q=a/R`, `d=a mod R`.
For `h_n[a,b,k]` read that layer/direction's row `L`, lane `k`; for `c_n`
read row `L`, lane `K+k`.
The reverse final hidden state corresponds to original time zero, not the
reverse half of `output[b,L-1,:]`. Pin this distinction with asymmetric data.

The vendored reference is `modules/pytorch/aten/src/ATen/native/RNN.cpp`,
especially `LSTMCell`, the bidirectional layer, and the `lstm` overload.
Inspect the actual linked CPU backend during the oracle milestone: backend
selection and float32 intermediate materializations affect numerical agreement.
Use explicit rounding where the selected Native arithmetic contract requires
it, and preserve graph-output F32 rounding. Do not silently assume OCaml float
accumulation matches ATen bit-for-bit or change a shared tolerance to pass LSTM.

Sigmoid and tanh can be expressed with the existing exponential basis. Use a
stable tanh expression based on `exp(-2*abs(x))` and sign selection rather than
an overflowing `exp(2*x)` quotient. Freeze the formula and any intermediate
rounds only after tiny and longer-sequence ATen comparisons; include saturated
gates, mixed signs, and nonzero initial states. Native/Native4D and Direct/
Symbolic agreement should be exact under the same authored program; ATen
agreement uses the repository's existing numerical tolerance unless a measured,
explicitly documented operation-specific contract is needed.

## 6. Wiring and implementation sequence

Each milestone has an exit condition; do not mark the operation landed when
only its importer or Direct evaluator works.

### M0 — binding and contract evidence

- [ ] Add `op "lstm" ~overload:"input"` to `bin/aten_ops_gen.ml` and verify
  Tensor-list arguments, the three-tensor return, generated dispatch, and real
  static linking. Inspect generated artifacts rather than editing them.
- [ ] Run a tiny, unequal `B/L/I/K` fixture through the real ATen binding;
  capture all three results. Check the actual archive's RNN closure, given the
  prior `conv3d` stub issue. Fix an in-scope closure omission or record the exact
  missing symbol/kernel and oracle alternative; a generated binding alone is
  insufficient evidence.
- [ ] Establish oracle cases for two/three layers, both bias settings, both
  direction counts, and both layouts. Confirm `dropout=0`, `0.5`, and `1`
  agree during inference, including a multilayer fixture where dropout would
  otherwise affect an intermediate layer. Record any backend-specific issue.
- [ ] Record accepted/rejected configurations and rounding observations in a
  tracked `.ai/lstm_design.md`; fold the scan contract into the tracked Expr/
  Region design documents so implementation does not depend on `_ai_` files.

### M1 — scan language and reference interpreter

- [ ] Implement the descriptor, projection, scope rules, and full semantic
  bundle in `lib/expr_internal/*` and `lib/expr/{expr,expr_api}.ml`/interfaces.
- [ ] Add `test/expr/scan_test.ml`: zero/one/multiple steps, coupled lanes,
  symbolic projections, nested reductions, capture-safe substitution, escaped
  binders, bad bounds, resource errors, structural comparison, and printing.
- [ ] Extend `Ground_eval`/`Ground_expr` handling for scan projections. Small
  concrete grounding may expand under explicit limits; larger cases must return
  the established bounded/unverifiable outcome, never assert or falsely prove.

### M2 — shared Region scans

- [ ] Extend `region_local`, `region_program`, `region_execution`, `region_eval`,
  and `region_trace`, plus dependent error/interface and rewrite consumers.
- [ ] Add `test/native/region_scan_test.ml`: materialized trace versus scalar
  reference; partition coverage; no duplicate outputs; scan counts; invariance
  violations; specialization; constant per-scan AST size; bounded memory
  accounting. Include a scan that reads an earlier trace, dependency-order
  rejection, and a three-scan chain that catches accidental recursive replay.
- [ ] Exercise Stage Program, Kernel adaptation/evaluation, source resolution,
  and Model Explorer printing with a tiny scan program before LSTM is involved.
  Teach fusion/cost analysis to handle a scan conservatively or reject an
  unsupported fusion with its typed boundary. Loop IR/code emitters remain
  outside scope because those backends are still proposed generally.

### M3 — first-class Native operation

- [ ] Add `lib/native/ops/lstm.ml`: payload/parameter records, JSON/printing,
  shape checks, walk configuration, and one operation-owned Region computation.
  Define the expanded payload up front: a nonempty layer collection, uniform
  direction count and bias policy, and explicit sequence layout.
- [ ] Register `Lstm` in `graph_ir.ml/.mli`, registry/dataflow/output arity,
  `graph_builder.ml/.mli`, and `graph_shape.ml`. All operands and three outputs
  must survive codec round trips, cloning/remapping, and graph validation.
- [ ] Extend `Region_computation.is_region_authored/program` for all three
  output ordinals and their distinct expected shapes. Route graph Direct and
  Symbolic evaluation through it; handle `eval_op`'s Region boundary explicitly.
- [ ] Audit Const-SSA dispatch, output transfer, transformations, and operation
  walks. Classify each output conservatively as continuous; do not add recurrent
  algebraic rewrites. Test constant-operand graphs or report an explicit
  existing limitation, rather than leaving an accidental registry hole.
- [ ] Add tiny graph fixtures with live `output`, `h_n`, and `c_n`, separately
  and together, plus a fixture with discarded state outputs.

### M3b — stacked layers and configuration coverage

This is required implementation work before the operation is considered landed,
not an optional extension after a single-layer landing.

- [ ] Author the layer/direction loop with `I_0=I`, `I_q=R*K` for `q>0`;
  initialize each scan from its own `q*R+d` state and use completed preceding
  traces as inputs. Check inter-layer rounding against ATen; adding layers
  must not erase a materialization boundary's numeric effect.
- [ ] Support optional bias pairs and both direction counts at construction
  time, without fabricated reverse operands or unnecessary zero-bias tensors.
- [ ] Centralize external input/output coordinate mapping for both layouts.
  Keep hidden/cell state indexing independent of `batch_first`.
- [ ] Build state emitters over every declared layer/direction trace using
  bounded, safe selection; all candidate reads must have valid row/lane indices.
  Assert `B*Q*R` scan starts per materialized output program and linear work
  growth with layers. Layer-state emission must not rerun prefixes of the stack.
- [ ] Keep the entire computation owned by one `Lstm` graph node. Static
  construction of one scan per layer/direction is permitted; static unrolling
  of every timestep and graph-level decomposition are not the chosen design.

### M4 — both imports and ATen verification

- [ ] Add `op_bridge_recurrent.ml` and `native_interp_lower_recurrent.ml`,
  registered with their respective dispatchers and target census. Decode
  `hx`/`params` tensor lists and every option; invoke the same Native validator.
  Check the exact `Q*R*(has_biases ? 4 : 2)` parameter count and each layer's
  input width. Validate finite `[0,1]` dropout before inference normalization.
- [ ] Update error variants/interfaces and bind all serialized result names in
  ordinal order. Compare the graphs produced by both imports on one fixture.
- [ ] Add `test/native_bridge/lstm_test.ml` and
  `test/native_interp/lstm_test.ml`. Require comparison of all three outputs:
  the current verification helper permits leading-only fixed-tuple comparisons,
  so an explicit output-count assertion is necessary to avoid a false green.
- [ ] Cover malformed flags/lists/ranks/shapes/dtypes, including missing or
  extra parameters, a missing second bias, invalid layer counts, invalid
  dropout, later-layer input-width mistakes, state leading dimension unequal
  to `Q*R`, reverse-only mistakes, and projected hidden/cell width mismatch.
  Positive tests must show that every newly supported flag value is admitted
  identically by both imports, including nonzero inference dropout.

### M5 — Native4D counterpart

- [ ] Add the first-class `Native4d.Op.Lstm` counterpart, reusing the Native
  payload if it remains axis/shape-free (the `Sdpa` precedent). A `4` suffix or
  duplicate `Ops4` record is unnecessary in that case.
- [ ] Wire `op`, `builder`, `graph_shape4`, `eval_op4`, `domain`, `lower_engine`,
  output transfer, and `region_computation4.native_op`. Reuse the same Region
  program and arithmetic. Map all three output IDs and operand roles directly.
- [ ] Add codec/domain/lowering/computation and map-verification fixtures, with
  all three outputs live, multiple layers, both layouts/bias settings/direction
  counts, and `B != L`. Time-first sequences use H for time and W for batch;
  neither layout requires T or D. Test the global `T=D=1` requirement and
  the operation's stricter accepted layout, including required `N=1`.
  Do not create a recurrent decomposition or unconditional dialect rejection.

### M6 — end-to-end evidence and ledger update

- [ ] Register a bounded Native operation walk and its coverage expectations.
  Exercise Direct, grounded Symbolic, Native4D, and Kernel for the same cases.
- [ ] Test both corpus shape families, including bounded numerical samples or
  full single-op runs with measured cost, in addition to cheap CI fixtures.
  Include `L=1`, `K=1`, unequal dimensions, dense off-diagonal recurrent
  weights, direction-specific weights, nonzero states, saturation, and a
  longer recurrence that reveals error accumulation. Run the configuration
  matrix below with tiny tensors so wider coverage remains inexpensive.
- [ ] Run an isolated LSTM graph through the Kernel/explorer path and record
  all three output stages. Keep production scan sharing under cost assertions.
- [ ] Regenerate `make pt2.json-model-support`; record Sequencer2D's actual next
  frontier independently for Native, Native4D, and Kernel. Clearing 36 nodes
  does not guarantee the full graph succeeds. Preserve the current scoreboard
  until the sweep demonstrates a change.
- [ ] Update `todo-ops.md` and `ops-progress.md` with the complete boundary /
  both-importers / Native / Native4D / Kernel / representative-graph matrix.
  Replace provisional claims in `lstm.md` with the landed tracked design link.

## 7. Expanded-domain acceptance matrix

Use deterministic F32 fixtures with unequal `B/L/I/K`, dense weights, distinct
per-layer/direction parameters, and nonzero initial states. Assert shapes,
output count, parameter routing, and values for all three outputs.

| Coverage | Required evidence |
| --- | --- |
| Core combinations | Tiny exhaustive `Q in {1,2,3}` × biases on/off × `R in {1,2}` × batch-first/time-first: 24 configurations, real ATen comparison plus both imports, Native Direct/Symbolic, Native4D, and Kernel. |
| Inference dropout | For a two-layer case in each bias/direction/layout combination, compare `p=0,0.5,1`; require equal Native computations/results after normalization and ATen agreement. Neither RNG state nor random-mask operands should be introduced. |
| Layout equivalence | Transposing input/output sequence axes gives the matching alternate-layout result; `h_n`/`c_n` retain identical dimensions and values. |
| Layer handoff | Compare a stacked call with separately evaluated one-layer calls using the corresponding parameter/state slices. Each upper layer receives the full lower-layer sequence; final states concatenate in layer-then-direction order. This is a test oracle, not implementation decomposition. |
| Bidirectional ordering | Asymmetric timesteps and weights distinguish forward/reverse concatenation and reverse final time zero, including at upper layers. |
| Bias absence | Exact two-parameter groups per direction; compare to ATen without biases. Supplement with zero-bias numerical comparisons without claiming that inserting `+0` is bitwise identity. |
| Limits and malformed input | Invalid `Q`, parameter counts, later-layer widths, state dimensions, dropout including NaN/infinity, and checked aggregate sizes fail with typed diagnostics. |
| Cost | Per output program: `B*Q*R` scan starts; `L` steps per scan; AST independent of `L/K` and linear in `Q*R`; no per-output-state replay of earlier layers. |

The core combination count is 24; dropout and numerical edge fixtures are
additional checks. Exhaustive tiny combinations belong in ordinary CI, while
corpus-size numerical/cost runs can use the established larger-fixture path.

## 8. Verification commands and completion criteria

Use focused checks while implementing, then the repository-wide checks required
by the affected surfaces. Start with non-promoting tests so failures remain
visible; review any golden changes before promotion.

```sh
opam exec -- dune build
NO_COLOR=1 opam exec -- dune runtest test/expr test/native
NO_COLOR=1 opam exec -- dune runtest test/native_bridge test/native_interp test/native4d
NO_COLOR=1 opam exec -- dune runtest
make check.file-size
make check.whitespace
make pt2.json-model-support
```

Also run the established JavaScript/backend checks affected by new Expr nodes
when their toolchains are available; audit exhaustive AST matches in those
consumers during M1. Honor the tracked file-size limits: existing OCaml files
at most 1000 lines, new files at most 750 unless already excepted. Split new
recurrent/scan modules instead of enlarging the existing compute dispatchers.

Completion means a real ATen oracle or documented binding-closure exception,
matching import domains, all three correct outputs, bounded shared recurrence
execution, Native4D mapping verification, and demonstrated Kernel construction
and reference execution across the expanded configuration matrix. Multiple
layers, optional biases, both directions/layouts, and nonzero inference dropout
are required for completion. Deferred optimizations are cross-output sharing,
trace lifetime analysis, gate/input-projection caching, and generated loop
backends. Deferred semantic extensions are training with stochastic dropout,
projections, packed sequences, unbatched input, and additional dtypes. None
should be implied by the inference support claim.
