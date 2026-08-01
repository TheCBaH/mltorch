# Native inference framework — master design

Phases 0–3 implemented (`lib/native/`); Phases 3b–5 (footprint/dependency
analysis, codegen/fusion, graph driver) are design-only. Pure-OCaml inference
engine with no libtorch/ATen dependency, living beside the existing C++-bridge
path (`lib/aten` + `lib/interp`).

The engine rests on four pillars, each detailed in its own doc:

1. **A 6-axis NHWC coordinate model** — every tensor is logically 6D over the
   fixed axes `N T D H W C` (C innermost / channels-last). → `native_tensor_design.md`.
2. **GADT payloads** — `float32 | float16 | bfloat16 | int64 | int32 | int16 |
   int8`, where the quantized `int16`/`int8` *must* carry quantization data and the
   reals *must not*, enforced by the type system (`float16`/`bfloat16` are reals
   with a software codec — no Bigarray kind). Quant is per-tensor **and** optionally
   per-channel along C. → `native_tensor_design.md`.
3. **Operations as semantics functors** — an op's algorithm is "given an output
   coordinate, produce that one output scalar by reading inputs." It is written
   once against an abstract `SEMANTICS` signature and instantiated two ways:
   `Direct` (returns a `float`, computes now) and `Symbolic` (returns an `expr`,
   builds a formula over the inputs). → `native_compute_design.md`.
4. **An external iterator / schedule** — the loop nest over the output coordinate
   space is decoupled from the algorithm. Naive, tiled, parallel and vectorized
   schedules call the same per-pixel functor; the symbolic path lowers the same
   algorithm to a fused loop nest. → `native_compute_design.md`.

The symbolic IR is specified in depth in `native_symbolic_language.md`: it makes
**indices first-class** (affine vs data-dependent), models multi-stage
computation as a **producer DAG** (so softmax's reused reductions and gather's
data-dependent reads are expressible), and adds a **`Footprint` interpretation**
that derives the input→output dependency of an output tile — the analysis that
validates tiling/fusion decisions.

## Why this shape

The split of *algorithm* (what one output pixel computes) from *schedule* (the
order/way pixels are visited) is the Halide idea, expressed in OCaml's module
system: the algorithm is a `functor (S : SEMANTICS) -> ...`, the semantics is the
"backend," and the schedule is an ordinary driver loop. Choosing the semantics
at instantiation is what lets a single op definition serve both concrete
execution and symbolic codegen — no duplicated op logic, and the symbolic and
direct results are provably the same expression by construction.

Codegen/fusion is the symbolic mode's purpose, and it falls out naturally:
because each op's pixel is a pure expression in terms of `load`s of its inputs,
**producer→consumer fusion is expression substitution** — replace a `Load` of a
producer with that producer's own pixel expression, and a chain like
`relu(add(conv(x), y))` collapses into one loop body over the symbolic IR.

## Proposed module layout (`lib/native/`)

The build order is bottom-up: each module compiles and its expect test is green
before the next starts.

| Module | Role |
|---|---|
| `axis.ml` | `axis = N\|T\|D\|H\|W\|C` and its order |
| `dim.ml` | scalar dimensional types `extent/index/count/offset` (≥0) + signed `delta` |
| `vec6.ml` | role-tagged 6-vector → `shape`/`coord`/`deltas`, `numel`, dense `offset`, `in_bounds` |
| `quant.ml` | quant params (per-tensor / per-channel), `quantize` / `dequantize` |
| `half.ml` | `Half`/`Bf16` software float codecs (no Bigarray half kind) |
| `payload.ml` | format GADT (`F32\|F16\|BF16\|I64\|I32\|I16\|I8`) + quant-indexed `payload` |
| `tensor.ml` | tensor record (`shape`+`payload`, dense), `packed`, `read`, `materialize` |
| `symint.ml` | `SymInt` + `ShapeEnv`; `size = Static \| Sym` (dynamic shapes) |
| `semantics.ml` | the `SEMANTICS`/`INDEX` signatures (the abstract numeric/index domains) |
| `direct.ml` | `Direct : SEMANTICS with type t = float` |
| `symbolic.ml` | `Tensor_sig` + the `expr` IR + `Symbolic : SEMANTICS with type t = expr` |
| `footprint.ml` | the `Footprint` interpretation → dependency/tiling verdicts |
| `schedule.ml` | iterators (naive / tiled / parallel / vectorized) + `evaluate` |
| `ops/*.ml` | one functor per op: `pointwise`, `conv`, `matmul`, `pool`, `reduce`, … |
| `codegen.ml` | lower (`expr` + schedule) → fused loop nest (later phase) |

Two cross-cutting concerns thread through the substrate: the **`Dim` scalar
discipline** (extent/index/count/offset are distinct non-negative types, `delta`
the one signed one — `native_tensor_design.md` §1a) and **dynamic shapes**
(`SymInt` axis sizes, §4) — both designed so they cost nothing in `Direct`
execution and only become live in the symbolic/footprint/codegen path.

No libtorch, no `ctypes`; storage is `Bigarray.Array1`. The library builds and
tests with `wrapped false` like the rest of the repo and needs no C++ archive.

## Relationship to the ATen path

`lib/aten`/`lib/interp` stay as-is — the reference. The native engine is an
independent backend with the same job (run a `.pt2` `ExportedProgram`), so the
natural acceptance test is **agreement with the ATen interp** on the same model
and image (top-1 class, then element-wise `allclose`). A later
`Native_interp.run` mirrors `Interp.run`: seed an env from `signature.input_specs`,
walk `graph.nodes`, dispatch each `target` to the matching op functor instead of
the bound `O.*` op.

## Phased roadmap

- **Phase 0 — substrate.** `axis` + `quant` + `payload` + `tensor`. Quant
  round-trip and 6D indexing/broadcast tests. No ops.
- **Phase 1 — direct pointwise.** `SEMANTICS` sig, `Direct`, naive `schedule`,
  `relu`/`add`/affine; one bounded reduction. End-to-end on a tiny tensor.
- **Phase 2 — direct heavy ops.** `conv2d`, `matmul`/`addmm`, `max_pool`,
  batchnorm-as-affine — all `Direct`. Golden-checked element-wise vs the ATen path.
- **Phase 3 — symbolic.** First-class index IR + stage DAG + `Symbolic`
  (`native_symbolic_language.md`). Per op, assert `eval (Symbolic.pixel c) ==
  Direct.pixel c` (verification falls out for free).
- **Phase 3b — footprint/dependency analysis.** The `Footprint` interpretation;
  derive each op's input→output dependency for a symbolic tile and classify it
  (local / stencil-halo / reduction-coupled / data-dependent). Validate against
  the brute-force read set on small shapes.
- **Phase 4 — codegen/fusion.** Pointwise fusion via expr substitution (legality
  checked by the footprint analysis), dequant∘quant elision across fused ops,
  tiled/parallel/vectorized schedules. Tiling legality is validated over `Sym`
  extents (one proof for the whole admissible-shape family).
- **Phase 5 — graph driver + dynamic shapes.** The first implemented slice is
  `lib/native_interp`: a pure-OCaml static importer that lowers a root PT2 graph
  to one `Graph_ir.graph`, preserves source SSA metadata and captured payload
  targets in `Pt2_native_graph`, and covers ResNet-18's convolution, inference
  batch-norm, ReLU, max-pool-with-indices, add, mean, permute, view, and addmm
  nodes, plus the MobileNet set: clamp, clone, hardtanh, mul and div. Non-trivial
  PT2 mappings are represented as structural native `Group`s:
  their relayout/decomposition stays grouped under the corresponding PT2 node,
  while one-to-one mappings remain flat. `Native_interp.run` materialises
  static float32 PT2 storage (including
  strides) into native tensors, binds constants through the sidecar, and calls
  `Eval_direct`; `native_graph print`/`eval` and `native_graph_cram.t` verify
  lowering of the real archive without the ATen bridge. Dynamic shape guards
  remain next.

  **Compile-time scalars.** MobileNet-v3's hardswish is serialised as
  `add.Tensor(x, 3)` / `div.Tensor(x, 6)` — a bare `as_int` sitting in a
  Tensor-typed slot. The ATen path materialises those with `full_like`
  (`Interp_decode.tensor_or_scalar_arg`); the native path instead routes them to
  scalar-parameter ops (`Add_scalar`, `Div_scalar`), so no edge needs a payload
  and the symbolic form gets a `const` leaf rather than a load — which is what a
  later fusion pass wants. `alpha` on add/sub is rejected rather than ignored:
  nothing in this zoo serialises a non-default one, and silently dropping it
  would compute `self + other` while claiming `self + alpha * other`.

  All three MobileNets (v2, v3_small, v3_large) run end-to-end natively and
  agree with `Interp.run` on the reference image (cosine similarity 1,
  relative L2 ~1e-6); ResNet-18 likewise. See `native_inference_verify.md`.

## Non-goals

Autograd / training, non-CPU devices, dynamic-*rank* tensors (rank is always the
fixed 6D frame — dynamic axis *sizes* via `SymInt` **are** supported, §4), sparse
tensors. Performance is explicitly *deferred* to the schedule/codegen phases —
Phase 1–2 optimize for obvious correctness.

## Open questions (Phases 3b–5)

- **Footprint/dependency analysis** (`native_symbolic_language.md` §3): not yet
  implemented; needed before tiling/fusion decisions can be validated.
- **Codegen/fusion** (Phase 4): pointwise fusion via expr substitution, dequant∘quant
  elision, tiled/vectorized schedules. Depends on footprint analysis.
- **Graph driver** (Phase 5): `Native_interp` dispatching `.pt2` nodes to native ops;
  resnet18 comparison vs `Interp.run`.
- **Quantized integer path**: default is dequantize→float→requantize; true
  integer-accumulator conv (int32 acc) is a Phase-4 option.
- **Per-pixel scalar granularity** recomputes shared window loads; only the
  schedule/codegen phase (CSE + fusion over the symbolic IR) recovers that. Fine
  for a correctness-first engine.
