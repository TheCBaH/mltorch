# `lstm.input`: scoping notes (investigated 2026-09-05, not landed)

Follow-up: [lstm-plan.md](lstm-plan.md) is the implementation plan. It corrects
the provisional conclusions below: state must be vector-valued; the current
Region path is the computation entry point; rank-3 sequence data occupies
H/W/C and can have a Native4D counterpart; and static unrolling is possible
but undesirable for compactness and execution cost. `output` contains the
last layer's sequence of hidden states. The original investigation is retained
below as history; no implementation has landed.

Scope expanded at the user's request: the plan now requires multiple layers,
optional biases, forward-only or bidirectional execution, both sequence
layouts, and valid nonzero dropout during inference. The corpus's single
configuration is a regression case, not a support restriction. Training,
projections, packed/unbatched input, and additional dtypes remain deferred.

Working notes from the session that landed `cumsum.default` and investigated
`lstm.input` alongside it. Not a tracked design doc (`_ai_/` is gitignored);
fold anything durable into `.ai/native_compute_design.md` or a new
`.ai/native_kernel_dsl_design.md` section if this work is picked up for real.
See [`todo-ops.md`](todo-ops.md)'s own status-header and "Matrix / reduction"
notes for the short version of this finding.

## ATen schema

```
lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases,
           int num_layers, float dropout, bool train, bool bidirectional,
           bool batch_first) -> (Tensor, Tensor, Tensor)
```

Three outputs: `output` (all-layer, all-timestep hidden states), `h_n`
(final hidden state per layer/direction), `c_n` (final cell state per
layer/direction).

## Corpus evidence: Sequencer2D (`sequencer2d_s`)

36 occurrences, all ONE shape/config family (checked by decoding
`modules/devcontainer.pytorch-image-models/models/sequencer2d_s/models/model.json`
directly):

- `num_layers=1`, `bidirectional=True`, `batch_first=True`, `has_biases=True`,
  `dropout=0.0`, `train=False`.
- `params` has exactly 8 tensors: `weight_ih_l0`, `weight_hh_l0`,
  `bias_ih_l0`, `bias_hh_l0`, and the four `_reverse` twins (one layer, two
  directions).
- Two concrete shape instances across the 36 occurrences:
  - `input [16,16,384]` (batch_first: `[batch=16, seq=16, input=384]`),
    `hx` each `[2,16,96]` (`[num_directions*num_layers, batch, hidden]`).
  - `input [32,32,192]`, `hx` each `[2,32,48]`.
  - (`weight_ih_l0` is `[4*hidden, input]`, `weight_hh_l0` is
    `[4*hidden, hidden]`, biases are `[4*hidden]` — the usual 4-gate LSTM
    layout, same for the `_reverse` set.)
- Only `output` (the first return) is consumed downstream in every occurrence
  — the graph names the other two `lstm_unused_1`/`lstm_unused_2` and never
  reads them (dead outputs, same shape `Discard`-routing precedent as
  `max_pool2d_with_indices`'s dead argmax edge already has).

So the corpus-scoping half of this op is easy: single-layer, bidirectional,
batch-first, biased, value-output-only. That is NOT what blocks landing it.

## The real blocker: no ordered-scan primitive

Native's `Compute` functor is written once per op, generic over
`Semantics.SEMANTICS`, and instantiated at both `Direct` (concrete floats)
and `Symbolic` (an `Expr.Value.t` AST, later interpreted/printed/proved) —
see `.ai/native_compute_design.md` and `lib/native/semantics.ml`. The ONLY
ordered-reduction primitives it exposes are:

```ocaml
val sum : lo:position index -> hi:delta index -> (position index -> t) -> t
val max_reduce : lo:position index -> hi:delta index -> (position index -> t) -> t
```

Both fold independently-computable terms (`body : position index -> t`, no
running-accumulator argument) with a FIXED, associative combine (`+` or
`max`). `Expr.Reduction.kind = Max | Sum` mirrors this at the symbolic level
(`lib/expr/expr_api.ml`) — there is no third kind, and no way for a reduction
body to see the fold's own running state.

An LSTM step is fundamentally not this shape:

```
i_t, f_t, g_t, o_t = split4(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
i_t, f_t, o_t = sigmoid(i_t), sigmoid(f_t), sigmoid(o_t); g_t = tanh(g_t)
c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
```

`h_t`/`c_t` are a NON-associative, nonlinear function of `h_{t-1}`/`c_{t-1}`,
not a wider accumulation of independently-computable terms. `sigmoid`/`tanh`
are not new primitives (both are already expressible via `exp`, the same way
`Softmax`/`Sigmoid` are), but the recurrence itself has no representation.

### Why this is NOT the same shape as `cumsum.default`

`cumsum[i] = sum(input[0..i])` LOOKS similar ("output at position `i` depends
on earlier positions") but is actually a plain associative `sum` whose upper
bound is derived from the output's own coordinate instead of a fixed extent —
already precedented by `Adaptive_axis.Compute.bin` (`lib/native/ops/pool.ml`,
adaptive pooling's per-output bin edges are also coordinate-derived). See
`Reduce.Cumsum.Compute.pixel` (`lib/native/ops/reduce.ml`) for the three-line
implementation. No new primitive was needed. This was checked by hand before
concluding `lstm.input` needed one — do not re-derive it.

### Why the recursion can't just be unrolled in OCaml

`T` (sequence length) IS a concrete, known integer at the point `Compute` runs
(Native shapes are static). The temptation is to unroll the `T` steps as
plain OCaml `let h_0 = ... in let h_1 = f h_0 in ...`, producing an
`Expr.Value.t` DAG with no new AST node. This does not work because of
`Compute`'s own contract: `pixel` must build ONE expression generic over a
SYMBOLIC output coordinate (an `Index_var`, standing for "any" position), not
one expression per concrete pixel — see `.ai/native_compute_design.md`'s
`Symbolic` walkthrough and `Conv.Conv2d.Compute (Symbolic)`'s own usage
example. The OUTPUT's own time coordinate is symbolic, so "which unrolled
`h_t` does this output read" is itself the thing being computed — you are
back to needing an ordered reduction/scan parameterized by a symbolic index,
not a fixed compile-time unroll.

## What landing this for real would need

A new ordered-SCAN primitive, threaded through the same places `Sum`/`Max`
already are:

1. `Expr.Reduction` (or a new sibling type): a fold whose body is a function
   of BOTH the index and the running accumulator(s) — LSTM needs two
   (`h`, `c`), not one.
2. `Semantics.SEMANTICS`: a new `scan`-shaped primitive both `Direct` and
   `Symbolic` implement (`Direct`: an actual loop; `Symbolic`: a new
   `Expr.Value.t` constructor).
3. Printer, structural-equality/hash (`Expr.Value.compare`/`hash`), and
   well-formedness checks for the new constructor.
4. Per `.ai/native_kernel_dsl_design.md`'s "Reductions and intrinsics"
   section, at minimum an "opaque intrinsic" bundle (interpreter, printer,
   footprint, well-formedness) — Loop IR lowering and a proof-comparison rule
   can lag if the op stays out of the generated-kernel/fusion path for now,
   the same partial-implementation precedent `Max_pool` already has.

This is foundational DSL work, not a vertical op slice — closer in shape to
the original `Kernel_adapt`/`Kernel_eval` work in
`.ai/native_kernel_dsl_design.md` than to any single op landing in
`ops-progress.md`. No other current backlog row needs it (`einsum.default`,
`max.dim`, `tile.default` are all ordinary associative reductions or
gathers).

## Recommendation

Do not attempt `lstm.input` as a normal op-landing session. If picked up:

1. Design the scan primitive first, as its own `.ai/` doc section (probably
   an addendum to `.ai/native_kernel_dsl_design.md`, since it already has the
   "Reductions and intrinsics" framing this fits into).
2. Land the primitive with a TRIVIAL consumer first (e.g. a hand-written test
   op, or nothing at all beyond the DSL plumbing) before building `Lstm` on
   top of it, so a DSL bug and an LSTM-semantics bug can't be confused.
3. Only then build `Lstm`'s own `Compute`, scoped to the corpus's single
   observed config (`num_layers=1`, both directions, biased, batch_first) —
   widen later with evidence, per this file's own "no corpus evidence, no
   representation" discipline used elsewhere (`To_copy`, `cumsum`'s own
   `dtype` restriction).
4. Native4D: almost certainly reject unconditionally from day one. A real
   two-state recurrence over a `T` axis with a per-step nonlinear gate has no
   obvious 4-axis-dialect-compatible shape, the same family `Unfold`/
   multi-batch `Batched_matmul`/`Conv3d` already occupy.
