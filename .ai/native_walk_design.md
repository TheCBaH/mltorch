# Native Op Walk Verification — Design

## Goal

Systematically verify functional equivalence between native ops (`lib/native/`) and ATen C++ kernels
by random-walking over the configuration space of each op and asserting `native == ATen` at every
step. Individual spec tests check a few hand-picked configs; this framework provides broad coverage
over the hyperparameter manifold, and aims to cover the **whole curated aten↔native bridge**.

## Two backends, one machinery

The walk runs against **two backends** that share the same engine:

- **ATen backend** (`lib/native_walk`, libtorch-linked): subject is an `Aten_spec.Op_spec.t`, the
  oracle is `native == ATen` (`Aten_spec_run.run`). Walk modules are **generated** (below).
- **Native backend** (`lib/native_op_walk`, pure OCaml): subject is a one-node native graph + its
  inputs, the oracle is **Direct == Symbolic** — the engine's two evaluation paths (`Eval_direct`
  vs `Stage_program.ground (Eval_symbolic.run g)`) must agree. No libtorch; runs under
  `dune runtest test/native`. Walk modules are **hand-written**, one per native op.

### `walk_core` — the shared, backend-neutral foundation (`lib/walk_core/`)

- `Float32`, `Pcg` (moved here; `aten_spec` re-exports them so `lib/native` can use the PRNG
  without depending on `aten_spec`).
- `Walk`: the `axis` type, `pick`/`field_axis`, the `module type Op_space`/`Op`, and the generic
  `run : (module Op with type subject = 'a) -> verify:(…) -> …` loop. `verify` is passed in per
  backend (run evaluator + compare), so `walk_core` links neither backend.
- `Window_math`: pure conv/pool cascade arithmetic.
- **The walk "language"** — `Limits`, `Shape`, and range/compound axis combinators (next section).

### The walk language: global Limits + compound axes

Numeric dimensions are **domains, not lists**, bounded by one shared config:

- `Limits.t` — global caps (max kernel/stride/pad/dilation/channels/extent/batch/numel) + `default`
  + `module type S`. Each op's `Walk` is a **functor over `Limits.S`**, so one shared instance
  (`lib/native_op_walk/walk_limits.ml`) bounds every op's tensor sizes.
- `Shape.t` — a neutral 6-axis (N T D H W C) tensor shape carried as **one** compound config entry;
  `valid`/`resize` clamp to the per-axis cap and the total-numel budget, so a shape is valid by
  construction. `lib/native/walk_bridge.ml` maps it to the engine's `Vec6` (direct field map).
- Axis combinators: `int_axis ~lo ~hi` (draw any int in range), `hw_axis ~lo ~hi` (an H/W pair as
  one entry setting two fields — kernel/stride/pad/dilation), `shape_axis` (mutate one axis of the
  shape within limits). `field_axis` (enumerated) stays for non-numeric axes (padding mode, dim
  subsets, bool). This collapses conv's ~27 scalar axes to ~7 compound ones.

A native op's walk properties (`cfg`/`axes`/`cascade`) live **inside the op module** (its `Walk`
functor in `lib/native/ops/*.ml`) — they are NOT shared with the ATen backend, whose constraints can
differ. Only the machinery above is shared. The graph `build` (which needs `Graph_builder`, above
the op modules in the dep graph) and the Direct-vs-Symbolic `verify` live in `lib/native_op_walk`.

> The rest of this document describes the **ATen backend**'s generated walk modules. The native
> backend reuses the same `walk_core` engine + language; Phase C will move the ATen backend onto the
> same Limits/compound language (retiring the `Recipe_*` discrete lists).

## Architecture

The walk modules are **generated at build time** from the same curated op `selection` that drives
the ATen bindings, augmented with hand-crafted meta for the shapes the yaml can't express:

```
native_functions.yaml ─┐
                        ├─> bin/aten_ops_gen.ml  (target "aten_op_walk.ml")
walk_meta.ml (curated) ─┘        │  Aten_walk_gen.file
                                 ▼
                lib/aten_op_walk/aten_op_walk.ml   ← generated walk modules + all_walks + needs_meta
                                 │  delegates logic to ↓ and spec-assembly to Op_<name>.spec
   lib/aten_walk_recipes/  (pure, aten_spec only)
     Walk        : axis type, `module type Op`, pick, tensor_spec
     Window_math : output_dim, spatial cascade helpers
     Recipe_conv / Recipe_pool / Recipe_reduce / Recipe_default : cfg + axes + cascade + derivations + pp
                                 │
   lib/native_walk/op_walk.ml :  run : (module Walk.Op) -> ...   (the only ATen-C++-linked piece)
```

This mirrors the existing pure/C++ split: `aten_op_spec` (pure, generated) vs `aten_spec_run`
(C++). The recipes and the generated `aten_op_walk` stay **off** the ATen C++ build; only
`native_walk.run` links it (via `aten_spec_run`).

### Engine — a module, not a bag of functions

`Walk.Op` (in `lib/aten_walk_recipes/walk.ml`) is the required signature: `type cfg`, `target`,
`initial`, `axes`, `cascade`, `build`, `pp`. `Op_walk.run (module M) ~ppf ~pcg ~steps` consumes one.
Step 0 is the cascaded `initial`; each step picks an axis, mutates it **freely**, re-cascades, then
calls `Aten_spec_run.run`. Stops on the first mismatch so the failing config is the last line.

### Coverage tiers (set by the generator, per op)

| Tier | Condition | Walk emitted |
|---|---|---|
| **Meta** | has a `walk_meta` entry | full family recipe (conv/pool/reduce) |
| **Default** | exactly one tensor arg, and every other arg has a **concrete renderable default** | `Recipe_default`: vary the lone tensor's dim sizes (rank fixed), scalars at defaults — provably valid |
| **Needs-meta** | ≥2 tensor args, a tensor-list, or any arg that would be filled by None / has no default | no module; target collected into `needs_meta : string list` |

The Default boundary is conservative on purpose: an optional arg that nominally defaults to `None`
is *not* auto-filled, because per-arg `None` can still be an invalid *combination* (e.g.
`clamp(min=None, max=None)` is rejected by ATen with a hard `c10::Error` that aborts the process).
Such cross-arg constraints belong in `walk_meta`. The generator partitions the entire `selection`,
so every spec'd op either walks or is listed in `needs_meta` — the gap is visible, never silent.

The same rule decides `layer_norm.default` before anyone chooses: it has three tensor arguments,
two of them optional, so `default_tensor`'s "exactly one tensor argument" test fails and
`fill_non_tensor` would refuse the optionals anyway. It lands in `needs_meta` the moment the binding
is added, and a `walk_meta` entry plus a recipe is the only walk it can ever have — there is no
vacuous default-tier walk to displace. Any op with an optional tensor argument is automatically
`needs_meta` for this reason.

That conservatism is also what keeps a whole class of vacuous walk from existing, and it is worth
recording because op6-impl's F7 predicted the opposite. `slice.Tensor` has a default for *every*
non-tensor argument (`dim=0, start=None, end=None, step=1`), and those defaults are jointly the
IDENTITY slice — so a Default-tier walk would vary only the input shape, hold every parameter at
the identity, and report `matched` for any implementation that returns its input. It does not get
one: `fill_non_tensor` returns `None` for an optional type before it ever looks at the default, so
`start`/`end` disqualify the op and it lands in `needs_meta` like `pad.default`. The `walk_meta`
entry is therefore what gives it a walk at all, not what displaces a misleading one. Do not weaken
`is_opt` to "fill it if it has a default" — the identity-configuration hazard is real even though
this particular op avoids it.

## Cascade replaces filtering

Cross-correlated parameters are kept valid by a deterministic `cascade : cfg -> cfg` (per recipe),
applied after every mutation — not by filtering candidate values. Examples:
- **Conv** (`Recipe_conv`): round `in_channels`/`out_channels` up to multiples of `groups`; grow each
  spatial input until `output_dim ≥ 1`.
- **Pool** (`Recipe_pool`): clamp `pad ≤ kernel/2`; grow inputs for `output_dim ≥ 1` (dilation 1).
- **Reduce** / **Default**: identity (any dim subset / any single-tensor shape is valid).
- **Norm** (`Recipe_norm`, shared by `rms_norm.default`, `layer_norm.default` and
  `native_layer_norm.default`): identity, for the same structural reason as Linear. The normalized
  extents appear in four places — the input's trailing axes, `normalized_shape`, the weight and the
  bias — and one candidate list supplies all of them, so they cannot disagree. Its `weight` and
  `bias` axes walk all four **graph shapes**, because `Graph_ir` carries both operands as
  **options** and each absent case is a different path rather than a ones/zeros tensor; "bias but no
  weight" is the state no exported model produces and the one a paired encoding gets wrong, so the
  walk's step count and seed are chosen to *reach* all four rather than assumed to. Note what an
  *uncorrelated* recipe would do here: generate specs ATen rejects, which degrade to `skipped` — a
  suite of skips that reads as coverage and is none.

  The `bias` and `cudnn_enable` axes are **omitted** for a target that has no such argument, not
  defaulted to a one-element candidate list. A single-valued axis is not free: the walk still spends
  a step on it and still names it as the mutated field, which is a step that changed nothing dressed
  up as one that did. `pp` prints the whole config including the fields a given target does not use,
  because one recipe serves three targets and a renderer that hid them would have to know which
  target it was printing for.

  `layer_norm`'s `cudnn_enable` is walked at both values against the same expected result. ATen's
  own composite names it `bool /* cudnn_enable, deprecated */` and drops it, so what the axis pins
  is that the CPU path really does ignore it — the *decoding* of the argument is pinned by the
  bridge dispatch fixtures instead. `native_layer_norm` has no such argument (none survives export)
  and its `eps` is required with no schema default, which is why the recipe's `eps_or_default`
  supplies the functional overload's own 1e-05 rather than inventing a third number.
- **Linear** (`Recipe_linear`): identity, and deliberately so. `in_features` occurs once, in both the
  input's trailing extent and the weight's second dimension, so the correlation other recipes repair
  after the fact is structural here and there is nothing left to cascade. Its `leading` axis carries
  the whole leading shape as one candidate value, which is how a step varies the input's **rank** —
  the pass-through of leading axes is the property a rank-2-only walk can never see.
- **View** (`Recipe_view`, `view.default` / `_unsafe_view.default`): a rank-4 shape plus a reshape
  `target` drawn as one correlated axis, never picked independently — a valid target is a
  factorization of the source's element count, and mutating it on its own would usually be invalid
  and degrade to a skip. Unlike `Recipe_binary`'s `pattern` (valid for any base shape, so its
  `cascade` is identity), a reshape target's concrete VALUE depends on the current shape, so
  `cascade` here is non-trivial: it recomputes `target` from `pattern` and the current `n`/`c`/`h`/`w`
  after every mutation, whichever axis a step touched (a shape axis or `pattern` itself), and repairs
  `c` to even first for the one pattern (`Split_c`) that needs it. `pattern` covers a `-1` form
  (`Minus_one_c`) and both rank-changing directions (`Split_c` grows rank 4→5, `Merge_hw`/`Flatten`
  shrink it). Two `walk_meta` entries share the recipe: `_unsafe_view.default` is the row this group
  needs, and `view.default` is added alongside it as independent evidence that commit 1's shared
  numel resolver still accepts what it used to (op3-impl.md commit 5) — the native `Reshape_nwalk`
  already covers the compute kernel and needs no change, so it is not evidence for either exact
  target.
- **Pad** (`Recipe_pad`, `pad.default`): shape plus the whole `(pad list, mode, value)` triple as ONE
  correlated axis. Two independent reasons, and either alone is fatal to an uncorrelated recipe.
  First, mode and value are not independent — ATen refuses a value on a non-constant mode — so a
  separate `value` axis would spend most draws on configurations ATen rejects, which degrade to
  skips that read as coverage. Second, reflect's amounts are bounded by the extent they mirror, and
  ATen accepts two pairs only at rank 3 or 4. The second correlation needs no `cascade`: `pads`
  *derives* the reflect amounts from the shape currently drawn, so a step on any shape axis cannot
  leave a stale amount behind, and `cascade` stays the identity. Rank is fixed at 4 deliberately —
  a rank axis would have to correlate with the pair count as well, and the rank-relative pad-list
  decoding it would exercise is covered against every rank ATen accepts by the bridge and importer
  fixtures.

  **What a random walk over one configuration axis does and does not cover, stated because the
  golden makes it look otherwise.** The configuration is one axis of five and `field_axis` redraws
  uniformly including the value already held, so twenty steps reach six of the seven configurations
  and five steps reach one. That is not a gap: the configuration space is swept exhaustively, and
  against the same ATen oracle, by `test/native_bridge/shape_ops_test.ml`. What the walk adds is shape and
  tensor-value variation *under* each configuration. Both halves are needed, and neither is the
  other — the same division `.ai/native4d_design.md` draws between hand values and the
  direct-versus-symbolic table.
- **Slice** (`Recipe_slice`, `slice.Tensor`): rank, dim and the bound PATTERN as one correlated
  axis, with the concrete bounds derived from the drawn extent. Two correlations stacked: a dim is
  only valid for a rank (so they travel as one candidate, as `Recipe_transpose` does) and a
  `(start, end)` pair is only valid for an extent (so `bounds` computes it from the shape currently
  drawn, which is why `cascade` is the identity — there is no stored value for a repair to fix).
  Patterns cover the identity, both ends, both negative spellings, both clamps, and two strides,
  and every one of them is non-empty for any extent ≥ 2 — an empty slice is legal in ATen, has no
  Native form, and would report as a `skipped` that reads as coverage.
- **Transpose** (`Recipe_transpose`, `transpose.int`): rank and the dim pair drawn as ONE correlated
  axis, never independently — a dim pair like `(2,3)` is only valid for rank ≥ 4, so mutating rank and
  dims separately would generate invalid combinations ATen rejects. Candidates cover positive,
  negative, equal, leading and trailing pairs across ranks 2, 3 and 4; `cascade` is identity since
  every candidate is already a whole valid triple.

  The **native** side gets its first `Permute` walk in the same commit (`lib/native/ops/permute.ml`'s
  `Permute.Walk`, assembled by `Permute_nwalk`): a full 6-axis shape plus a permutation drawn from a
  *fixed* candidate list (not generated combinatorially — a full 6-axis bijection has 720 elements,
  most meaningless for a rank-4-visible tensor), **weighted away from identity** — none of the
  candidates is the identity permutation, which is what the coverage-sweep header
  (`test/native/native_walk_test.ml:9-18`) warns a lazy axis produces. At least one candidate is a
  3-cycle: a two-element swap is its own inverse, so it cannot distinguish a correctly-directed
  mapping from a reversed one — the "source/destination mapping reversed" mutation `transpose.int`'s
  own evidence proves vacuous (op3-impl.md commit 6) becomes real only once a non-involutive
  permutation exists to test it against.
- **Binary** (`Recipe_binary`, `sub.Tensor`): the first **two-tensor** recipe (op3-impl.md F5 —
  `Aten_walk_gen.default_tensor` admits an op only with exactly one tensor argument, so a two-operand
  op has no Default tier and would sit in `needs_meta` forever without one). One shape plus a
  broadcast `pattern` — `Equal`, or one operand with a single axis forced to extent 1 — drawn as one
  correlated axis over whole valid `(lhs, rhs)` shape *pairs*, never two independently mutated shape
  fields: mutating each operand's shape on its own axis would generate genuine broadcast mismatches
  ATen rejects, degrading to skips that read as coverage and are none — the same failure mode the
  `RmsNorm` bullet above warns about, for the same reason. `cascade` is identity: every `pattern` is
  valid for every base shape. The native side's `Sub_nwalk` reuses `Pointwise.Bin.Walk`, which walks
  **one** shape for both operands (no broadcast) — deliberately narrower than the ATen recipe, since
  broadcasting both operands independently in the native walk too would move goldens for `add`/`mul`
  (which share `Bin.Walk`) to prove a property neither of those rows claims. Broadcast evidence for
  the native side is the ATen recipe (an independent oracle) plus the hand-written fixtures in
  test/native_bridge/dispatch_test.ml (op3-impl.md Part IV #4).

- **Sdpa** (`Recipe_sdpa`, `scaled_dot_product_attention.default`, op8-impl.md commit 4): the first
  **three-tensor-plus-optional-mask** recipe, and the first where an uncorrelated axis would degrade
  to something worse than a `skipped` line (see "the oracle-identity trap" below). One correlated
  `(batch, heads, sq, sk, e)` tuple derives Q/K/V's shapes — there is no `ev` field, deliberately: a
  distinct value feature dimension moves ATen's CPU dispatch off the flash kernel entirely (F4), so
  the recipe must be unable to express it, not merely unlikely to draw it. `mask_kind` is a closed
  variant (`No_mask | M2 of {q;k} | M4 of {d;h;q;k}`) where each field says "this axis is 1 rather
  than its real extent", so every representable mask is flash-admissible *by construction* — there is
  no way to build the `[Wq,Wk]`-shaped-but-wrong mask `check_attn_mask_shape` would reject. `cascade`
  is identity, matching `Recipe_binary`'s reasoning: every field is already a whole valid tuple.

  **The oracle-identity trap.** Every other recipe in this file risks generating a spec ATen itself
  refuses, which degrades to `skipped` — visible, and merely wastes a walk step. `Sdpa`'s risk is
  different in kind: `check_attn_mask_shape` and the other flash gates fail *silently*
  (`sdp_utils_cpp.cpp`'s `priority_order_cpp` falls through to `SDPBackend::math` with no error), so
  an inadmissible config still produces an answer — just from a different kernel — and the golden
  would read `matched` while quietly comparing against the wrong oracle. A `skipped` line is not the
  signal to watch for here; the recipe's types making an off-oracle point unrepresentable is the
  *primary* defence, and it must be independently checked, not trusted by construction alone. The
  secondary check is a scratch `extern "C"` wrapper around `_fused_sdp_choice` (never committed — see
  op8-impl.md commit 0 step 2 for why: the op's `int` return is outside `Aten_c_type.map_returns`'
  admitted shapes, so the generator cannot reach it, and the return ABI is not extended for a
  one-time spike), run over the recipe's **full** axis product before its golden is fixed. **Anyone
  who widens this recipe must rebuild that probe and re-run it** — nothing in CI re-checks the
  backend choice, only that the walk's answer agrees with ATen's, which is exactly the failure mode
  described above.

`output_dim in pad kernel dilation stride = (in + 2*pad - ((kernel-1)*dilation + 1)) / stride + 1`
(`Window_math.output_dim`). Cascade is pure (no PCG), so trajectories stay reproducible from the
seed. Because mutation always picks freely and cascade guarantees validity, there is no "advance the
PCG on a rejected candidate" special case anymore.

## Spec assembly via the generated bindings

Each generated `build` constructs its `Op_spec.t` through the generated `Aten_op_spec.Op_<name>.spec`
(`{ target; args = to_args r }`, added by `aten_spec_gen`). The record field names are checked
against the binding at build time, so a walk's argument assembly **cannot drift** from the op it
calls — a missing/renamed field is a compile error in the generated file.

## PCG Threading

A single `Pcg.t` threads through everything:
1. `Walk.pick pcg axes` → next axis (1 `Pcg.next`)
2. `axis.mutate pcg cfg` → pick a candidate value (1 `Pcg.next`); cascade then repairs dependents
3. `build pcg cfg` → fill tensor elements with `Pcg.uniform` (numel steps per tensor)

Tensor specs use `Tensor_spec.Values`, so `Aten_spec_run`'s internal PCG is never advanced — our
external PCG has exclusive control over tensor content.

## Adding coverage for a new op

1. If the op fits an existing family (conv/pool/reduce), add a `walk_meta` entry: name the recipe,
   the `initial` config, the per-axis candidate sets, and the `build` body that fills the generated
   `Op_<name>` record. It moves from `needs_meta` to a Meta walk automatically.
2. If it's a new family, add a recipe module under `lib/aten_walk_recipes/` (cfg + `axes` +
   `cascade` + shape derivations + `pp`), then point a `walk_meta` entry at it.
3. Single-tensor value-only ops need nothing — they're picked up by the Default tier.
4. `dune runtest test/native_walk_test.ml` + `dune promote` to capture the new trajectories; check
   the `bridge coverage` block — every step should read `matched` or `skipped` (a `mismatched`/error
   line is a real native-vs-ATen divergence to investigate, not promote away).

## A configuration no walk_meta recipe can reach

`add.Tensor`/`sub.Tensor`/`mul.Tensor`/`div.Tensor` each have a Meta-tier walk (`Recipe_binary`,
plus `Walk.tensor_spec_nonzero` for `div`'s divisor) covering their genuine tensor-tensor form. None
of those recipes — and no recipe ever could — cover the *scalar-in-a-Tensor-slot* configuration the
PT2 exporter also produces for these targets (`add.Tensor(x, 3)`, serialized with a bare `Int`/
`Float` where the schema's second argument is `Tensor`). A `walk_meta` recipe still has to produce a
real `Op_<name>.t` that `Aten_spec_run` dispatches through actual libtorch, and libtorch's
`add.Tensor` takes two real tensors — there is no ATen call that reproduces "a Tensor argument
holding a bare scalar", because that is a PT2 serialization choice, not an ATen calling convention.
The walk generator synthesizes real tensor arguments for every `Tensor`-typed schema parameter, so
this configuration is unreachable by construction, not merely unlikely to be drawn — the same
distinction §"Cascade replaces filtering" draws between "improbable" and "impossible".

This is not a coverage gap, for the same reason the `pad.default` bullet above says its own
one-axis-of-five configuration space isn't one: the space is swept exhaustively against real ATen
elsewhere. Here that's `test/native_bridge/scalar_verify_test.ml`'s `verify_print` tests (`"verify: add.Tensor with
a serialized Int/Float scalar"`, and the `sub`/`mul`/`div` equivalents), which go through
`Interp_verify.dispatch` — real ATen materializing the scalar with `full_like` on one side,
`Op_bridge` routing it into a Native scalar op parameter on the other. Both int and float literals
are covered for all four targets. Don't add a `walk_meta` entry chasing this configuration; extend
the `verify_print` fixtures instead.

## Files

- `lib/aten_walk_recipes/` — pure recipes + `Walk.Op` signature (`walk.ml`, `window_math.ml`,
  `recipe_{conv,conv_padding,pool,reduce,linear,norm,adaptive,bounds,unbind,default}.ml`)
- `lib/aten_gen/walk_meta.ml` — hand-crafted per-op meta (conv2d, conv2d.padding, linear, rms_norm,
  max_pool2d, mean.dim, …)
- `lib/aten_gen/aten_walk_gen.ml` — the generator (tier partition + emit)
- `lib/aten_op_walk/` — generated `aten_op_walk.ml` (walk modules, `all_walks`, `needs_meta`)
- `lib/native_walk/op_walk.{ml,mli}` — the runner (`run : (module Walk.Op) -> ...`)
- `test/native_walk_test.ml` — per-op focused walks + the `bridge coverage` sweep
