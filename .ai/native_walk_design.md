# Native Op Walk Verification — Design

## Goal

Systematically verify functional equivalence between native ops (`lib/native/`) and ATen C++ kernels
by random-walking over the configuration space of each op and asserting `native == ATen` at every
step. Individual spec tests check a few hand-picked configs; this framework provides broad coverage
over the hyperparameter manifold, and aims to cover the **whole curated aten↔native bridge**.

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

## Cascade replaces filtering

Cross-correlated parameters are kept valid by a deterministic `cascade : cfg -> cfg` (per recipe),
applied after every mutation — not by filtering candidate values. Examples:
- **Conv** (`Recipe_conv`): round `in_channels`/`out_channels` up to multiples of `groups`; grow each
  spatial input until `output_dim ≥ 1`.
- **Pool** (`Recipe_pool`): clamp `pad ≤ kernel/2`; grow inputs for `output_dim ≥ 1` (dilation 1).
- **Reduce** / **Default**: identity (any dim subset / any single-tensor shape is valid).

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

## Files

- `lib/aten_walk_recipes/` — pure recipes + `Walk.Op` signature (`walk.ml`, `window_math.ml`,
  `recipe_{conv,pool,reduce,default}.ml`)
- `lib/aten_gen/walk_meta.ml` — hand-crafted per-op meta (conv2d, max_pool2d, mean.dim)
- `lib/aten_gen/aten_walk_gen.ml` — the generator (tier partition + emit)
- `lib/aten_op_walk/` — generated `aten_op_walk.ml` (walk modules, `all_walks`, `needs_meta`)
- `lib/native_walk/op_walk.{ml,mli}` — the runner (`run : (module Walk.Op) -> ...`)
- `test/native_walk_test.ml` — per-op focused walks + the `bridge coverage` sweep
