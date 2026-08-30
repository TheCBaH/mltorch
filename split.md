# Source file split plan

## Goal

Keep authored source and test files small enough to review, navigate, and test
independently.  This plan concerns the main repository only; it does not apply
to submodules, generated files, downloaded model fixtures, build output, or
third-party sources.

## Size policy

Line count is the primary measure.  Byte/character count is a secondary signal
for unusually dense files.

| Size | Policy |
| --- | --- |
| At most 750 LOC | Preferred maximum for a new authored file. |
| 751--1,000 LOC | Do not grow without a short justification in the change. |
| 1,001--1,500 LOC | A split is expected; additions need a concrete decomposition plan. |
| More than 1,500 LOC | Exception only.  Record why the file cannot reasonably be split. |

Review a file at the same time when it exceeds roughly 60 KB, even if it is
under the line limit.  Generated sources must be marked as generated and are
not subject to this policy.

Snapshot tests have one narrow exception: a single command's required expected
output may exceed 1,500 lines.  Such tests must still be one command or model
per file; unrelated snapshots must not be combined merely for convenience.

## Baseline

The initial scan covered 588 authored source and test files (141,733 lines).
There are 24 files over 1,000 lines and 11 over 1,500 lines.  Production code
accounts for 71,338 lines; test code accounts for 70,395 lines.

The largest files are:

| File | Lines | Split direction |
| --- | ---: | --- |
| `test/native_graph_cram.t` | 5,997 | One snapshot file per model/command. |
| `test/native_transform_cram.t` | 4,750 | One snapshot file per model/transform mode. |
| `test/native_bridge_test.ml` | 4,556 | Tests and helpers grouped by bridge operation family. |
| `lib/native_interp/native_interp.ml` | 3,012 | **Done.** See below. |
| `test/native/compute_test.ml` | 2,593 | Tests grouped by operation family, with shared fixtures extracted. |
| `test/native_transform_fold_cram.t` | 2,530 | One folded-transform snapshot file per model. |
| `lib/native_aten_bridge/op_bridge.ml` | 2,056 | **Done.** See below. |
| `lib/expr/expr.ml` | 1,694 | **Done**, via a sibling library. See below. |
| `test/native/graph_json_test.ml` | 1,632 | Parsing, serialization, and error-case groups. |
| `test/native/permute_passes_test.ml` | 1,592 | Tests by pass or permutation family. |
| `lib/model_explorer_export/me_limits.ml` | 1,545 | **Done.** See below. |

## Implementation order

### 1. Split Cram snapshots without changing behavior

Split the three large Cram files by the command/model boundaries already
present in their transcripts.  Preserve each command and its expected output
verbatim; only move blocks into separately named `.t` files.  Confirm that the
Dune Cram configuration discovers or explicitly lists every new file, then run
the relevant Cram suite.

This is the lowest-risk first step and makes individual snapshot failures
readable.  It improves file size, though not total snapshot line count.

### 2. Split oversized test modules by domain

For each OCaml test module above 1,500 lines:

1. Identify shared builders, fixtures, and assertion helpers; move only those
   to a narrowly named helper module.
2. Partition tests by externally visible feature or operation family, rather
   than by arbitrary line ranges.
3. Keep each resulting test module below 750 lines where practical; retain
   test names and coverage unchanged.
4. Run the focused test executable before and after each module split, followed
   by the affected suite.

Start with `native_bridge_test.ml` and `compute_test.ml`, since their test
groups can be separated with less production-code coupling.

### 3. Decompose production modules along existing boundaries

Do one production module per change so each extraction is easy to review and
bisect.

* `lib/expr/expr.ml`: **done**, superseding the earlier "exception, do not
  split" verdict below the line. That verdict held for splitting *within* the
  `expr` library; it did not anticipate moving the implementation to a
  *separate* library. `lib/expr_internal/expr_repr.ml` isolates the mutually
  recursive `value`/`bool_expr`/`reduction` knot as raw, unexported type
  definitions in one small file; every other module (`bool.ml`, `value.ml`,
  `reduction.ml`, `builder.ml`, `rewrite.ml`, `fold.ml`, `check.ml`, `eval.ml`,
  `pp.ml`, plus the already-acyclic `axis.ml`/`role.ml`/`coord.ml`/`source.ml`/
  `max_op.ml`/`reduce_var.ml`/`index.ml`/`intrinsic.ml`/`checked.ml`) declares
  its own public slice as a **manifest re-declaration** of the matching
  `Expr_repr` type (`type t = Expr_repr.value = | Const of float | ...`)
  rather than an alias — since the recursive tie lives only inside
  `Expr_repr`, every other file is ordinary non-recursive OCaml and can be its
  own compilation unit, including `Rewrite`, which still reconstructs raw
  constructors directly (no smart-constructor folding) since a
  structure-preserving rewrite must not silently change syntax. `lib/expr`
  itself is now a 27-line façade (`expr.ml`) plus a 4-line `expr.mli`
  (`include Expr_api.S`), with `Expr_api` a `private_modules` signature
  fragment inside the `expr` library re-exporting `expr_internal`'s modules
  under the `Expr.*` namespace. Every `expr_internal` file is comfortably
  under 300 lines. See `.ai/native_expr_refactoring_design.md`'s August 2026
  update for the full mechanism. The original rejected-layout analysis is
  preserved below for the record; it remains correct for same-library splits,
  it was just not the whole answer.
* `lib/native_interp/native_interp.ml`: **done.** Split into
  `native_interp_error.ml` (error payload types and printers, 499 lines),
  `native_interp_decode.ml` (argument decoding and shape/config helpers
  internal to `lower`, 912 lines), `native_interp_lower_context.ml` (the
  immutable dispatch context, 30 lines), `native_interp_lower_compute.ml`
  (conv/norm/attention/pointwise/pool/reduce/linalg dispatch, 610 lines),
  `native_interp_lower_shape.ml` (shape/view dispatch, 384 lines),
  `native_interp_lower.ml` (`lower`/`lower_archive` orchestration, 227 lines),
  `native_interp_tensor.ml` (`tensor_of_pt2`, 107 lines), and `native_interp_exec.ml`
  (`run` and the transform/verify/evaluate pipeline, 346 lines).
  `native_interp.ml` itself is now a 93-line facade of manifest type/value
  aliases, so `native_interp.mli` did not need to change at all — the public
  `Native_interp.*` surface is byte-for-byte the same as before the split.
  Compilation order is forced one way (error → decode → lower → exec →
  facade) because `lower`'s dispatch depends on the decode helpers and
  `run`/`transform` depend on `lower`/`lower_archive`, and OCaml compilation
  units cannot be mutually recursive across files the way `expr.ml`'s
  in-file `module rec` can — hence the facade running last rather than
  first.

  The dispatch split uses one explicit immutable context containing only the
  graph, escape token, and lazy read set. The mutable provenance accumulators
  remain in the orchestration module, because no operator arm uses them.
* `lib/native_aten_bridge/op_bridge.ml`: **done.** Unlike `native_interp.ml`'s
  `lower`, `dispatch`'s ~1,360-line match has no shared mutable state across
  arms -- each arm is a self-contained expression over `aten_env`/`node` --
  so it split cleanly by operation family rather than needing one big
  function left whole. Extracted `op_bridge_error.ml` (error payload types
  and `pp_error`, 196 lines) and `op_bridge_decode.ml` (argument decoding,
  the six relayout permutations, and conv2d/pool2d param helpers shared by
  every family, 500 lines), then eight family dispatch modules, each a
  `dispatch : aten_env:... -> Node.t -> ... option` covering a disjoint set
  of `node.target` strings: `op_bridge_pointwise.ml` (230 lines: add/sub/mul/
  div/pow/clamp/relu/sigmoid/silu/gelu/hardsigmoid/hardswish/hardtanh/sqrt),
  `op_bridge_linalg.ml` (88: bmm/addmm/linear), `op_bridge_conv.ml` (175:
  conv2d.default/.padding, convolution.default), `op_bridge_pool.ml` (91:
  max_pool2d.default/_with_indices, adaptive_avg_pool2d), `op_bridge_reduce.ml`
  (76: amax, mean.dim, linalg_vector_norm), `op_bridge_norm.ml` (271:
  batch_norm, group_norm, layer_norm/native_layer_norm, rms_norm),
  `op_bridge_attention.ml` (115: scaled_dot_product_attention), and
  `op_bridge_shape.ml` (415: clone/permute/transpose/pad/slice/cat/stack/
  select/unsqueeze/unbind/split_with_sizes/upsample_bilinear2d/view).
  `op_bridge.ml` itself is now a 43-line facade: `dispatch` is
  `List.find_map (fun f -> f ~aten_env node) dispatchers` over the eight
  family functions, which is equivalent to the original single match because
  every `node.target` string is handled by exactly one family (order among
  the eight cannot matter). The library is `(wrapped false)` like
  `native_interp`, and has no `.mli` of its own -- checked every consumer
  (`grep -rn "Op_bridge\."`) before splitting and found only `dispatch`,
  `pp_error`, and `error` used outside comments, so no facade aliasing was
  needed for anything else; the new family/decode/error module names are
  simply not part of `Op_bridge`'s surface.
* `lib/model_explorer_export/me_limits.ml`: **done.** `(wrapped false)` like
  `op_bridge`/`native_interp`, but unlike either it has its own `.mli`
  declaring `Limits.t`/`Diagnostic.t` as `private` records and
  `Wire_limits.t = private Limits.t`. Split along the file's own section
  comments into six modules with a strict one-directional dependency chain:
  `me_limits_error.ml` (31 lines: `Invalid`/`live_error`/`error`/`pp_error`),
  `me_limits_over_limit.ml` (164: `Scope`/`Field`/`Over_limit`/`check`/
  `check64` -- standalone, used by other validators rather than by the rest
  of `Me_limits`), `me_limits_hard.ml` (410: the checked-int64-arithmetic
  helpers, `Hard_scalars`, `request_live_bytes`/`response_live_bytes`, and
  `Hard`), `me_limits_profile.ml` (680: the `Limits` module -- fields,
  `create`/`validate`/`derive`, and the `trusted`/`untrusted`/`small`/`large`
  profiles), `me_limits_wire.ml` (68: `Wire_limits`), and
  `me_limits_diagnostic.ml` (233: `Diagnostic`). `me_limits.ml` itself is now
  a 35-line facade of manifest module/type/value aliases (`module Limits =
  Me_limits_profile.Limits`, etc.), and `me_limits.mli` did not need to
  change: unlike `native_interp.mli`'s GADTs, none of these aliased types are
  GADTs or need a restated concrete declaration, so a bare `module X =
  Y.X`/`type t = Y.t` alias was enough in every case -- OCaml's module-alias
  checking against an ascribed signature accepts a plain public record
  satisfying a `private` specification. No internal file carries its own
  `.mli`, so the `private` restriction in `me_limits.mli` binds only the
  library's external callers, exactly as intended -- the six new files
  construct and pattern-match each other's records freely.

Avoid a purely mechanical file split: each new module needs a single clear
responsibility and explicit dependency direction.  Do not introduce cyclic
dependencies simply to retain the former file layout.

### 4. Prevent regression

**Done.** `scripts/check-file-size.sh`, wired into CI as the `file size check`
step in `.github/workflows/build.yml` (before the devcontainer step, since it
needs only git). Two checks, deliberately different in mechanism:

* **Tree check (1000-line cap):** every tracked `.ml`/`.mli` file in the
  current checkout, full stop -- no diff against a base ref.  This sidesteps
  the shallow-checkout problem entirely (the default `actions/checkout`
  fetch-depth of 1 has no history to diff against) and matches the
  Definition of done below ("no non-exempt file exceeds 1,500 lines" /
  "files over 1,000 lines have ... an approved exception") literally: it is a
  property of the tree, not of any one commit.
* **New-file check (750-line cap):** files with git status `A` in
  `git diff-tree HEAD^ HEAD`, i.e. added by the tip commit.  This needs one
  parent commit, so the checkout step's `fetch-depth` moved from the default
  1 to 2 -- not a full-history fetch, just enough for `HEAD^` to resolve.
  Checking only the tip commit (not the merge-base with `main`) means a
  multi-commit PR could in principle add a large file in an earlier commit
  and shrink it before the last one; accepted as an intentional simplicity
  tradeoff, since the tree check still catches the file if it stays over
  1000 lines, and squash-merge workflows collapse a PR to one commit anyway.

Both checks share `scripts/file-size-exceptions.txt`
(`path|review-by|reason`).  It currently grandfathers every pre-existing
file over 1000 lines that Phases 2-3 didn't reach (`test/native/verify_test.ml`,
`test/native/graph_symbolic_test.ml`, `lib/native/transform/map_verify.ml`,
`test/native/graph_test.ml`, `lib/model_explorer_export/me_export.ml`,
`lib/model_explorer_export/me_session.ml`, `bin/native_graph.ml`,
`lib/aten_gen/walk_meta.ml`, `lib/native/ops/pointwise.ml`,
`lib/native/ops/conv.ml`) plus the two already-documented exceptions
(`lib/expr/expr.ml`) -- these were
found during Phase 4 by a fresh `wc -l` sweep and were **not** part of the
original baseline table above, which only tracked the top ~11 files.  The
check therefore ships required (not warn-only) from the start: every currently
non-exempt file is already under the cap, so there is nothing left for a
warn-only period to catch.

Cram (`.t`) snapshots are out of scope for this script: their exception is
"one command/model per file", not a line count, and isn't mechanically
checkable the same way.

Run locally with `make check.file-size`.

### 5. Clear the grandfathered exception list

Phase 4's `wc -l` sweep found 11 files over 1000 lines that Phases 2-3 never
reached (see Phase 4 above) were grandfathered into `scripts/file-size-exceptions.txt`
purely so the new CI check would not fail on pre-existing debt. This section
tracks paying that debt down, one file per commit, largest first, same
process as Phase 2:

* `test/native/verify_test.ml` (1,380 lines): **done.** Split into
  `verify_fixtures.ml` (95 lines) plus ten by-concern files, all under 400
  lines. See the commit for the exact helper-sharing analysis — several
  identifiers looked shared under a naive grep and were not (`both` and
  `cell` both collide with ordinary English words in the file's prose
  comments, `payloads` collided with a string literal and a `%s` format
  specifier); confirming each candidate's *real* call sites, not just grep
  hit counts, is the load-bearing step for this kind of narrative test file.
* `test/native/graph_symbolic_test.ml` (1,335 lines): **done.** Split into
  `graph_symbolic_fixtures.ml` (77 lines) plus nine by-operation-family
  files, all under 300 lines. Simpler than `verify_test.ml`: no narrative
  prose to confuse grep, but two helpers (`conv_axis`/`conv_params`) that
  looked conv-only by their names turned out to be shared with the
  unrelated "Symbolic lowering: stages are well-scoped" test too, caught
  only by the build actually failing on the first `dune build` after moving
  them — a reminder that grep hit-counting narrows the search, it does not
  replace building.
* `lib/native/transform/map_verify.ml` (1,238 lines): **done.** First
  production (non-test) module in this wave. Split into
  `map_verify_types.ml` (563: the Member/Budget/.../Policy vocabulary,
  including the `Member.t` GADT — public only via `Member.Erased`, so the
  facade's `module Member = Map_verify_types.Member` alias narrows correctly
  under ascription with no manifest re-declaration needed) and
  `map_verify_check.ml` (538: `boundary_of` through `check_cluster`, the
  actual verification algorithm). `map_verify.ml` is now a 178-line facade:
  aliases for the types module plus `Make_pair`/`run`/`step`, which stay
  here since they are real entry-point logic, not mechanical re-export.
  `map_verify.mli` unchanged. Built clean after exactly one fix (an
  `open Graph_ir` the types file needed that the original single file
  didn't need to state).
* `test/native/graph_test.ml` (1,226 lines): **done.** Same family split as
  `graph_symbolic_test.ml`, same op-family boundaries where they overlap
  (pointwise/activation/combine/shape/norm/pool/conv/pad_slice), plus a
  `basic` family for the four tests that are about graph/id construction
  rather than one op. Named `graph_direct_*` (not `graph_<family>_test.ml`)
  because a differently-purposed `graph_fixtures.ml` already exists in this
  directory (shared transform-pass fixtures, unrelated). Split into
  `graph_direct_fixtures.ml` (75 lines) plus nine family files, all under
  250 lines.
* `lib/model_explorer_export/me_export.ml` (1,128 lines): **done.** A
  different shape from the earlier production splits: `me_export.ml` has
  its own `.mli`, so (unlike `op_bridge`/`me_limits`/`map_verify`, none of
  which had one on any sibling file) a plain sibling `open Me_export` from
  inside the library would only see the five-item public surface, not the
  internal `shape`/`staged`/`capability`/`wrap`/`passes` helpers every
  other function needs. The fix was to move those OUT of the file carrying
  the `.mli`: `me_export_types.ml` (212 lines: format detection, the error
  vocabulary, `encode_bounded`, `staged`/`shape`) and `me_export_shape.ml`
  (625: `unlowered_shape` and `lowered_shape` — the latter alone was ~570
  of the original 1,128 lines, one long monadic pipeline of sequential
  `let*` stages with no internal mutable state, unlike
  `native_interp_lower.ml`'s `lower`, so it needed no further split once
  moved to its own file). `me_export.ml` itself keeps the real public
  entry points (`Options`/`load`/`session`/`detail`/`handle`) plus aliases
  for what moved out; `me_export.mli` unchanged.
* `lib/model_explorer_export/me_session.ml` (1,114 lines): **done.** The
  easy case in this batch: no `private`/GADT anywhere in `me_session.mli`,
  so the split was mechanical. `me_session_panes.ml` (193: the ten small
  pane/comparison/flow wire types), `me_session_capability.ml` (267:
  `Capability` and `View`, which depends on `Capability.graph_stage`), and
  `me_session_document.ml` (663: the `Session` document type). Facade is
  23 lines (plain aliases plus `Runtime`, three lines, left in place since
  it isn't worth its own file). `me_session.mli` unchanged. Built clean
  after adding the two expected `open`s to `me_session_document.ml`.
* `bin/native_graph.ml` (1,093 lines): **done.** First `bin/` split: no
  `.mli` to worry about, but `bin/dune`'s `(modules native_graph)` needed
  the new module names added explicitly (unlike `lib/native`'s
  `include_subdirs unqualified`, this executable stanza does not
  auto-discover sibling files). Split into `native_graph_common.ml` (237
  lines: everything used by more than one subcommand),
  `native_graph_args.ml` (150: every Cmdliner argument term), and one file
  per subcommand (`_print`, `_eval`, `_transform`, `_to4d`, `_const_ssa`,
  `_visualize`). `native_graph.ml` itself is now 44 lines: the `Cmd.group`
  and entry point. `verdicts_by_edge` looked transform-only by its
  position in the file but is also called from `to4d`, caught by the
  first `dune build` after the split — same lesson as
  `graph_symbolic_test.ml`'s `conv_axis`/`conv_params`.
* `lib/aten_gen/walk_meta.ml` (1,089 lines): **done.** ~35 independent,
  self-contained `let <op> = { ... }` entries of one record type, no
  cross-references between them — the cleanest case in this batch
  mechanically, except that most entries carry a descriptive comment
  glued directly to their own `let` (no blank line before it), so a naive
  "next entry's line minus one" range would have put ~27 of the 35
  comments in the wrong (preceding) file. Computed real per-entry start
  lines by walking backward from each `let` to the nearest blank line
  first, then verified the full 1,089-line count reconciled before
  extracting anything. Split into `walk_meta_entry.ml` (the `t` record)
  plus eight op-family files; `walk_meta.ml` is now a 57-line registry
  (`entries`/`find`) plus a `type t = Walk_meta_entry.t` alias, needed
  because `aten_walk_gen.ml` references `Walk_meta.t` directly and there
  is no `.mli` here to make that automatic. `lib/aten_gen/dune`'s explicit
  `(modules ...)` list needed the new files added by hand.
* `lib/native/ops/pointwise.ml` (1,068 lines): **done.** No `.mli`, but a
  new kind of external-surface constraint anyway: `lib/native4d/graph_shape4.ml`
  references `Pointwise.Add`, `Pointwise.Bin`, `Pointwise.broadcast_coord`
  from OUTSIDE the library, which only works because this file's own
  filename-to-module mapping names it `Pointwise` and every op is a
  submodule nested inside — splitting without a facade would have quietly
  renamed that surface to `Pointwise_unary.Clamp`/`Pointwise_binary.Add`/
  etc, each a distinct top-level module, breaking every external reference
  at the type level with no missing-file error to point at it. Checked
  every `Pointwise.<name>` reference across lib/test/bin before writing
  the facade. Split into `pointwise_unary.ml` (222: Clamp/Clone/Sqrt),
  `pointwise_activation.ml` (507: Hardsigmoid/Hardswish/Hardtanh/Relu/
  Sigmoid/Silu/Gelu), and `pointwise_binary.ml` (352: broadcasting +
  Binary/Scalar_binary functors + Add/Sub/Mul/Div/Pow). Facade is 37 lines.
* `lib/native/ops/conv.ml` (1,060 lines): **done.** The simplest split in
  this batch: three modules (Conv2d, Conv2d_padding, Convolution), one
  dependency edge (padding and convolution both depend on Conv2d for its
  `axis_window`/`params`/`Compute`/`Walk`, not on each other), no
  attached-comment surprises, same `Conv.<name>` external-surface check as
  `pointwise.ml`. Split into `conv_conv2d.ml` (387), `conv_padding.ml`
  (333), `conv_convolution.ml` (351); facade is 10 lines. This closes out
  the grandfathered-exception list from Phase 4's `wc -l` sweep.

## Definition of done

* The three combined Cram snapshots are split at command/model boundaries.
* No non-exempt file exceeds 1,500 lines.
* Files over 1,000 lines have a recorded, scheduled split or an approved
  exception.
* CI enforces the no-growth rule for oversized files.
* Focused and full affected test suites pass after each split.
