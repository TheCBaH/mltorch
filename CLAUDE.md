# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Principles

1. **Think before coding** — state assumptions explicitly, surface tradeoffs, halt if something is unclear.
2. **Simplicity first** — write minimal code solving only the stated problem; no speculative abstractions.
3. **Surgical changes** — modify only essential code; match existing style.
4. **Goal-driven** — turn vague requests into verifiable criteria before starting.

## Conventions

- **Record types get their own module, named `t`** (e.g. `Node.t`, `Graph.t`),
  matching the existing `Module.t` style (`Tensor.t`, `Vec6.t`). This guarantees
  field-label uniqueness by construction (`Node.outputs` vs `Graph.outputs`).
  **Do not silence warning 30** (duplicate labels) — give the types distinct
  namespaces instead. For mutually recursive records, use `module rec`; if a
  variant only references the others, parametrise it so it can stay outside the
  recursive group (see `lib/native/graph_ir.ml`: `'g gop`).
- **Result/Option handling**: the error framework is `Err`, the single module of
  the `err_trace` library vendored at `vendored/err_trace` — depend on it as
  `err_trace`; name and public_name are the same, unlike most of this repo's
  other libraries. `lib/core` keeps only `Core.Pretty`, the Fmt glue; the
  dependency runs one way and `Err` must never depend on Fmt.
  Don't hand-roll `match ... Ok/Error`/`Some/None` purely to print — compose
  through `Fmt.result`/`Fmt.option`/`Core.Pretty` instead. Don't cross an
  `Err.t` into an exception boundary with an open-coded
  `match ... | Error e -> failwith (...)` — use `Err.or_raise ~pp_error`, which
  raises `Err.Exn.E` carrying the whole wrapper. Because `Err` registers a
  `Printexc` printer, a boundary that emits **outward** (wire response, browser,
  third-party log) must match `Err.Exn.E` and print `Err.Exn.pp_kind` — the
  default rendering includes the detection stack. Bridge an option with
  `Err.of_option`, or `Err.map_none` when the payload is expensive, raising, or
  effectful (`of_option`'s is eager, and so is `Err.guard`'s — which is why the
  hot per-node limit checks in `lib/model_explorer_export` stay explicit).
  The `Error.t` wrapper is **abstract**: rebuilding it is a type error, proved by
  `test/native/error_opacity.t`. Use `Err.map_error` to widen a row —
  it preserves the original detection origin, and `graph_view.ml:352` was a live
  instance of the bug that follows from not doing so. Add `~pos:__POS__` at
  subsystem seams, not per row: it is the only provenance that exists on the
  JavaScript backends. A deliberate drop of the wrapper (into Cmdliner's
  `(_, string) result`, into Jsont) is fine, but give it a **named** helper —
  `to_cli`, `Pattern.of_err`, `Me_request.or_jsont` — built on `Err.export`,
  which marks `Err.Action.Export` and then unwraps, so the marking cannot be
  forgotten and the crossing is distinguishable from the defect. Inbound is
  `Err.import`, which records `Import` and no `Detect` (a third party detected
  it, not you); `Err.payload` is the unmarked unwrap, for tests and rendering.
  **Escape a deeply recursive walk with `Err.Escape`**, never a private
  exception: `with_escape` establishes one frame, `throw`/`throw_error`/
  `or_throw` exit it, and the token is threaded so the reach is visible in the
  types. Carrying a bare row instead of the `Error.t` is the defect it removes —
  `Native_interp` had it, and reported every malformed graph as detected at the
  catch. A token is invariant in its payload, so a callee at a narrower row gets
  `Err.Escape.map` of its caller's, not one of its own (`Expr.Eval.eval_index`).
  **`Err.Accum` accumulates every failure, so it must not run above a ceiling
  that has not yet passed** — `Me_session`/`Me_flow` check aggregates before the
  walks linear in them, and that ordering is the bound. Where the work is a
  compile-time constant and no check gates another it is right, and
  `Err.Accum.fold_errors` collapses the batch back into the module's own domain
  so the list never reaches a published signature
  (`Me_limits.Limits.check_against`).
  Two-list traversals go through `Err.List.map2`/`iter2 ~unequal_lengths` so the
  arity check cannot drift from the pairing it guards — unless the mismatch is a
  module invariant, which stays `invalid_arg` (`Native_interp.add_env`).
  **Payloads carry data, not prose.** A closed value set is a variant, not a
  string (`Me_limits.Field`, `Native_interp.arg_kind`,
  `Me_flow.Transition.Kind_tag`); a multi-field payload is a named record with
  named fields, in its own module when labels would collide
  (`Me_limits.Over_limit`, `Shape_error`'s `channels_mismatch`). A single-tag,
  single-id payload — `` `Unknown_state of string `` — is already precise; leave
  it. A third-party message (Jsont's) may stay a string, but name the tag for its
  source (`` `Jsont ``, `` `Model_json_decode ``) so it does not read as a case
  the module declined to classify.
  Configuration and monitors belong to the **host**, never to `lib/`:
  `lib/err_host` reads `MLTORCH_ERROR_*` once from an executable's entry point.
  A test that changes the policy uses `Err.Config.with_config`, which restores it
  even when the body raises. `Config.fast` is **not** the production preset its
  name suggests — it empties the action set and drops the whole event trail;
  `Config.deterministic` is the one that keeps boundaries without stacks.
  See `.ai/error_handling_design.md` and `.ai/printer_conventions.md`.
- **Never assume a 63-bit `int` in the JS-reachable libraries** — `lib/native`,
  `lib/walk_core`, `lib/core`, `lib/native4d`, `lib/expr`, and now `lib/pt2`,
  `lib/native_graph` and `lib/native_interp`, which js_of_ocaml reaches since the probe
  began reading real `.pt2` models. js_of_ocaml's `int` is **32 bits**
  (`Sys.int_size = 32`), so a value that can reach 2^31 must be `int32`/`int64` **or be
  bounds-checked at every operation**, and a literal like `0xFFFFFFFF`
  silently truncates to `-1` there — which turns a mask into a no-op.
  **Narrow to `int` only after bounding the value**, and bound *aggregates*
  separately: a product or sum of individually-in-range factors can still overflow.
  Seven defects have turned on this — `Pcg.next`'s `[0, 2^32)` contract and
  `Half.f32_bits`' mask (both live), `Pt2_pickle.int_of`, `Pt2_tensor.numel` /
  `contiguous_strides` and `Native_interp`'s `storage_index` (all latent, now
  bounded), `Opickle.Src.uint4`'s `land 0xFFFF_FFFF` (live; fixed upstream in
  the vendored submodule), and `Reshape.output_shape`/`Aten_shape.resolve_view_size`
  comparing unchecked `Vec6.numel` results (live at the graph-construction
  boundary: two *different* source/target products can wrap to the same 32-bit
  `int`, so the equality that is supposed to reject a numel-changing reshape
  target passes instead — fixed by `Vec6.numel_bounded`, which divides each
  per-axis factor into the ceiling before multiplying, the same shape as
  `Kernel.Bounds.signature`'s existing fold). **A check on a wrapped result is
  not a bound** — three of those validated only the folded value, which a
  mid-fold wrap sails straight past.
  **Two commands catch a regression, and they catch different things.**
  `make jsoo.runtest` diffs the backends, so it sees only divergence;
  `make jsoo.inline-runtest` runs the expect suites under node against their
  committed goldens, so it sees a wrong answer that both backends agree on.
  See `.ai/js_backends_design.md`.

## Exploration & Planning — start in `.ai/`

The `.ai/` directory is the canonical design record for this repo. **Before exploring
unfamiliar code or planning any non-trivial change, list `.ai/` and read the docs
relevant to the area you're touching** — they capture the intended design, rationale,
and tradeoffs that the source alone doesn't show, and the place to look for an existing
plan before writing a new one.

The set of docs grows and changes over time, so discover them at the start of a task
rather than relying on a fixed list. Filenames are descriptive (`*_design.md` for
designs, area prefixes like `aten_*`, `native_*`, `schema_*`); grep `.ai/` by topic when
you're unsure which file applies.

**Only the design record is tracked.** `*_plan.md` and `*_census.*` are working files —
stage checklists, per-site inventories, progress markers — and are gitignored. Write
them freely, never `git add -f` them, and expect them to be missing in a fresh clone:
that is why some tracked docs cite plans that are not in the repo. When a plan or census
yields a rule that outlives it, fold that rule into a tracked doc (or this file) in the
same change — the plan is scaffolding, the design record is the deliverable.

When a design changes materially, update the corresponding `.ai/*.md` doc — or add a new
one — in the same change so this record stays the source of truth.

## Handling review feedback

Reviews arrive as a list of findings. Two rules do most of the work.

**Check every finding against the source before acting on it.** A finding can be right,
stale, or *understated*, and each needs a different response. Confirm with `file:line`,
and say plainly when one does not hold — accepting a wrong finding is as costly as
ignoring a right one. Worked examples from the symbolic-verifier reviews:

**If a finding says a check is vacuous, prove the check can fail.** Revert the fix, watch
the test go red, restore it. A test that has never failed is not evidence, and this is
the one class of defect a passing suite cannot surface.

**Verify the edit landed — a green build is not proof that it did.** Prefer the `Edit`
tool, which fails when its pattern does not match. A scripted `str.replace` silently does
nothing on a miss, and a formatter reflowing the target between writing the patch and
running it is enough to cause one.

Commit fixes as `fixup!` commits naming the commit that introduced the defect — one per
defect, not one per review — so `git rebase --autosquash` lands each where it belongs and
the history records what was wrong where. Group by defect rather than by file: a fix
spanning three files is one commit, and three unrelated fixes inside one file are three.

**Never amend a commit that already exists.** A second `fixup!` onto the same target is
the right move even when the first `fixup!` was itself wrong; both squash together.

Carry the `.ai/*.md` delta in the same commit as the fix it documents, exactly as for any
other change.

## Commands

```sh
make build      # build
make runtest    # run all tests
make format     # format — mandatory before every commit

# Single cram test or promote (no make target)
dune runtest test/model_cram.t
dune promote test/model_cram.t
```

**Run `make format` before every commit.** Formatting is enforced; unformatted diffs are noise.

### JavaScript backends (gated on node + the JS toolchain)

```sh
make js.build                # build everything: jsoo + melange
make jsoo.runtest            # build the probe natively + via js_of_ocaml, run both, diff
make jsoo.inline-runtest     # the expect suites under node (@runtest-js)
make melange.runtest         # same for the pure half (walk_core + core)
make js.runtest              # the three above
make melange.build.scaffold  # shim + fmt + jsont_base only — the diagnostic floor

# Gated on downloaded weights, so outside js.runtest — CI runs it in the jsoo job,
# which fetches this one model under its own cache key (never build.yml's).
make jsoo.pt2.download       # = pt2.download for the tier-2/3 model
make jsoo.pt2.runtest        # open a real .pt2, lower it, run inference, diff
```

`make jsoo.pt2.run` is **tier 3 and manual** — every sample of the same model through
`Native_interp` under node, checked against the `results.json` in the release zip, with
`--strict` so a ranking mismatch exits 1. It answers "is the answer right", which the
tier-2 diff cannot: that one compares the two backends to each other and would pass if
both were wrong. ~7.5 min (ten inferences at ~45s), so it is in no CI job and no
`runtest` alias. The flow is shared with the native runner through `lib/infer_report`,
which must never gain `interp`/`aten`/`ctypes`/`unix` — the clock is passed in as `~now`
for exactly that reason.

Not part of `make runtest` (they need node); CI runs jsoo and melange as two parallel
jobs. Melange is behind `--profile melange`, so plain `dune build` never compiles it.
Needs a devcontainer rebuild for `js_of_ocaml`/`melange`/`node`.

**The two backends have different scopes, deliberately.** jsoo reaches the whole PT2
path — so it needs `modules/pytorch` for `schema.yaml`, and its CI job checks submodules
out. Melange stops at `walk_core` + `core` (no Bigarray in any release), reaches no
submodule, and its job asserts that. See `.ai/js_backends_design.md`.

### Working in a git worktree

If you work in a `.claude/worktrees/<name>` checkout, **read `.ai/worktree_setup.md`
first** — `make` from the repo root won't see worktree edits, dune needs `--root .`
there, and submodules/base-ref have gotchas. Quick form:

```sh
git submodule update --init --recursive vendored/opickle   # once per worktree
opam exec -- dune build   --root . lib/native
opam exec -- dune runtest --root . test/native
```

### .pt2 / interpreter (gated on real model data)

```sh
make pt2.download-cram   # fetch the 5 models in PT2_MODELS_CRAM (see the Makefile)
make pt2.runtest         # run pt2_load_cram.t + interp_*_cram.t against them
make inference           # timed smoke run over every model in PT2_MODELS_ALL
```

These need real release weights (`data/pt2/`, gitignored) and are not part of
`make runtest`; CI runs them as separate steps after the main build.

## Architecture

```
modules/pytorch/torch/_export/serde/schema.yaml   ← git submodule (excluded from dune scan)
        │  (shell copy rule in data/dune)
        ▼
bin/schema_gen.ml                                  ← reads YAML, emits OCaml source
        ▼
lib/generated/pytorch_types.ml                     ← build-tree artifact (compiled library)
test/schema_pytorch.ml                             ← build-tree artifact (used via #use in cram)
        ▼
test/*_cram.t                                      ← cram tests decode real model.json files
```

### Libraries

| Path | Dune name | Role |
|---|---|---|
| `lib/pytorch_schema/` | `pytorch_schema` | Schema meta-parser + code generator |
| `lib/schema_runtime/` | `schema_runtime` | Runtime: `String_map = Map.Make(String)` |
| `lib/generated/` | `pytorch_types` | Generated decoder library |
| `bin/schema_gen.ml` | `schema_gen` (exe) | CLI: reads YAML, calls generator |
| `lib/expr/` | `expr` | The symbolic expression language: typed indices, values, reductions, intrinsics. Depends only on `core`+`fmt`, so it owns no tensor, storage or graph type — `native` consumes it, never the reverse |
| `lib/pt2/` | `pt2` | Libtorch-free `.pt2` reader: ZIP (via `zipc`), pickle (via vendored `opickle`), model.json/weights-config decoding |
| `lib/pt2_aten/` | `pt2_aten` | Bridges `pt2`'s raw strided tensors to runnable `Aten_tensor.t` (via `of_storage`), kept separate so `pt2`'s own tests need no C++ build |
| `lib/interp/` | `interp` | Walks an `ExportedProgram` graph and dispatches each node to the bound ATen ops |

Key modules in `pytorch_schema`: `Pytorch_schema` (YAML→type map), `Schema_codegen` (type map→OCaml source), `Type_expr`/`Type_expr_lexer`/`Type_expr_parser` (type-string parser), `Scc` (Tarjan SCC for recursive type detection).

`lib/interp`'s per-node dispatch (`interp_dispatch.ml`) is generated by `bin/aten_ops_gen.ml` from `lib/aten_gen/aten_decode_gen.ml`, the dual of the ATen binding generator: it reads the same `native_functions.yaml` entries that produce a binding and emits the matching decode-and-call arm, so dispatch can't drift from the bindings it calls. Same dep-tracking caveat as `lib/aten/dune` — rebuild manually when the yaml changes.

### Tests

- **Inline expect tests** (`test/scc_test.ml`, `codegen_test.ml`, `type_expr_test.ml`) — `ppx_expect`; promote with `dune promote`.
- **Cram tests** — isolated decode scenarios (`parse_cram.t`, `default_cram.t`), full model parsing (`model_cram.t`, `models_cram.t`), weights config (`weights_config_cram.t`).
- **pt2/interp tests** — `pt2_test.ml` is hermetic (in-memory zip + hand-encoded pickle); `pt2_load_cram.t` and `test/interp_*_cram.t` are gated on `PT2_DATA` (set by `make pt2.runtest`) and run against real downloaded model weights.

Cram tests run `ocaml schema_runtime.cma script.ml 2>/dev/null`. Both `schema_runtime.cma` and `.schema_runtime.objs/byte/schema_runtime.cmi` must be copied into the cram sandbox via dune rules. Source-tree files (e.g., `model_test_utils.ml`) go in `(deps ...)` directly — no copy rule needed.

## Key Constraints

- `modules/pytorch/` and `modules/devcontainer.pytorch-image-models/` are git submodules excluded from dune's scan. Reach them via shell rules using `%{project_root}/../../`. No automatic dep-tracking — `touch data/dune` to force a rebuild when `schema.yaml` changes.
- `yamlt` is vendored under `vendored/ocaml-yamlt/` (not on opam); `opickle` (pickle decoding) is vendored under `vendored/opickle/` the same way.
- `data/pt2/<model>/` (downloaded release weights/images/results) is gitignored and not part of the default build — fetch via `make pt2.download-cram` or `make pt2.download-all` before running the gated tests above.
