# `aten.max.dim`: a paired value/index reduction

`aten.max.dim(self, dim, keepdim=False)` returns `(values, indices)`: the max
over one named axis, and the position along that axis where it occurred.
`values` has the same shape/keepdim rule as `Amax`; `indices` shares that
shape and reports the winning position as an f32 ordinal, the same convention
`Value_of_index`/`max_pool2d_index` use elsewhere in the engine.

## Why this is not `Amax` plus a dropped output

`Amax` folds with `Max_op.Float_max` (`Float.max`). A second reduction that
independently re-scans for "which position holds the max" (the two-pass
`select`/`lt` derivation `AdaptiveMaxPool2dWithIndices.index_pixel` uses for
its own bounded 2D window) can disagree with a `Float_max`-folded value on a
NaN, because `Float_max` and the tie/NaN predicate a second pass would use are
different comparators applied to the same data. `Intrinsic.Max_pool`'s own doc
comment names this defect directly: value and index must advance under ONE
predicate, or they fall out of step.

So `MaxDim`'s two outputs both fold with `Max_op.pool_better` (ties keep the
incumbent — first index wins; a NaN retriggers, so the last NaN in iteration
order wins) — the SAME predicate `max_pool2d`/`max_pool2d_index` already use
for the fixed-window case. This is a deliberate choice of "the engine's
existing paired-fold convention" over replicating ATen's own `max.dim` CPU
kernel exactly: that kernel breaks out of its scan at the FIRST NaN
(`aten/src/ATen/native/cpu/TensorCompareKernel.cpp`'s `max_kernel_impl`),
which is a third, distinct NaN convention neither `Float_max` nor
`Max_op.Pool_max` implements. `Reduce.Amax` already diverges from ATen's own
`amax` kernel on NaN edge cases (documented in `Max_op.mli`), so accepting an
analogous, already-precedented divergence here — rather than inventing a
fourth bespoke convention — keeps the number of NaN policies in the engine
at two, not three.

## The reduction is axis-generic, not window-shaped

Unlike `max_pool2d`, the reduced axis here can be any extent, not a small
fixed kernel window — so it needs the same generic `[lo, hi)` reduction shape
`max_reduce`/`sum` already have, not `Intrinsic.Max_pool`'s opaque
fixed-geometry node. `Expr_repr.reduction_kind` gains two tags,
`Argmax_value`/`Argmax_index`, alongside the existing `Max`/`Sum`: same
`{kind; var; lo; hi; body}` record, `body` still the per-position comparison
key, differing only in the FOLD (`pool_better`, not `Float_max`/`(+.)`) and in
which half of that fold each kind reports (the winning `body` value, or the
winning `var` position carried into the value domain).

Because the record shape is unchanged, every construct-generic pass that
already treats `Reduce {kind; var; lo; hi; body}` structurally —
`Fold`/`Rewrite`/`Check`/`Value`'s compare-hash — needed **no** changes: none
of them branch on `kind` for anything but forwarding it (`Stdlib.compare`,
`Hashtbl.hash`) or printing it (`Reduction.kind_name`). Only the actual
evaluators needed new fold logic, because they are the ones that interpret
`kind` as an algorithm:

- `Expr.Eval.value` (both the direct-recursion and the depth-cutoff/JS
  branches of the `#if defined JS_BACKEND` split in `eval.ml`)
- `Eval_js_machine.run`'s explicit reuse-stack machine, and its three
  historical `js/probe/` candidate siblings (`eval_candidates.ml`,
  `eval_trampoline_delayed.ml`, `eval_machine_reuse.ml`, `eval_hybrid.ml`) —
  kept in sync as peer implementations of the same denotation, per their own
  header comments
- `Native_transform.Ground_eval`'s `Value.Reduce` arm, which builds the same
  paired fold out of `Ground_expr`'s already-generic `max`/`select`/
  `pool_better` primitives — the same primitives its own `max_pool` arm
  already used for the fixed-window case, just looped over `[lo, hi)` instead
  of a 2D window.

`Semantics.SEMANTICS` gains `max_dim`/`max_dim_index`, the `lo`/`hi`/`f` shape
`max_reduce` has; `Direct` implements each as its own single fused loop over
`Max_op.pool_better` (mirroring `max_pool2d_index`'s existing shape), and
`Symbolic` builds `Expr.Builder.reduction ~kind:Argmax_value`/`Argmax_index`.
The two must be called with the same `lo`/`hi`/`f` at a given call site — two
independent calls to a deterministic fold over identical inputs agree by
construction, the same reasoning that already lets `max_pool2d`/
`max_pool2d_index` be two separate `SEMANTICS` primitives rather than one
call returning a pair.

## The Native op

`Reduce.MaxDim` (`lib/native/ops/reduce.ml`) is a fixed two-output op,
`{ axis : Axis.t; keepdim : bool }` — a single `Axis.t` rather than
`Dims_keepdim`'s `Axis.t list`, since ATen's schema names exactly one `dim`
and an index is only meaningful for a single reduced axis. Its shape math
still goes through `Dims_keepdim` (wrapped as a one-element `dims`), so it
cannot drift from `Amax`/`Mean` on `keepdim` packing. Wired exactly like
`Max_pool2d_with_indices`: `Graph_shape` returns the same shape twice,
`Eval_op` selects `value_pixel`/`index_pixel` by `~output`, `Output_transfer`
classifies output 0 `Continuous` and output 1 `Discontinuous` (an argmax).

## PT2 import: liveness-aware, not assumed-dead

`max_pool2d_with_indices.default`/`adaptive_max_pool2d.default` always route
their index to `Discard`: no corpus occurrence reads it, and ATen provides no
value-only overload to import into directly. `max.dim`'s index does not get
the same treatment, because unpacking both outputs
(`values, indices = x.max(dim)`) is ordinary PyTorch, not a corpus-observed
constant — so `Native_interp`'s importer (which has whole-graph liveness via
`ctx.reads`, unlike the isolated-node `Op_bridge`) retains or discards EACH
output independently, the same `discard_if_dead` pattern `lstm.input`'s own
three-output arm already uses. `Op_bridge` has no such context (and ATen's
own index is int64 where Native is f32, not comparable to an oracle anyway),
so it always discards the index and exposes only the values — matching
`max_pool2d_with_indices.default`'s own bridge arm.

## Native4D: deferred, not rejected in principle

Reducing over one axis with a second (index) output is not, by itself,
outside Native4D's domain: `Max_pool2d_with_indices`/
`Adaptive_max_pool2d_with_indices` already prove the dialect can carry a
paired value/index output. What `MaxDim` needs beyond that is the routine
`Ops4` axis-renaming payload (`Axis.t` → `Axis4.t`) `Max_keepdims` and its
siblings already have, specialised to one axis plus a second output — undone
because no corpus model exercises it (the one real occurrence reduces axis
`D`, which `Domain.check_dims` would refuse regardless of whether the op is
otherwise supported). `Domain.check_node`/`Lower_engine`'s dispatch both
reject `Max_dim` explicitly, with a comment recording this as a scope
decision, not a discovered limitation — revisit if a model needs it.

See `native_add_op.md`/`native4d_add_op.md`/`expr_construct_migration.md` for
the general site lists this change followed, and `native_multi_output_design.md`
for the multi-output/dead-output policy background.
