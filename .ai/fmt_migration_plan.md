# Fmt migration plan

## Goal

Move the project's human-facing pretty-printing from ad hoc `Format` usage to
`Fmt`-style composition, while preserving the existing public printer shape:

```ocaml
val pp : Format.formatter -> t -> unit
```

The migration target is not "replace every mention of `Format`". The target is:

- value printers
- error printers
- report printers used by tests and CLIs
- the small helper functions repeatedly built around those printers

The main benefit is compositionality: nested printers should compose through
`Fmt.list`, `Fmt.option`, `Fmt.result`, `Fmt.any`, `Fmt.brackets`, and
`Fmt.pf`, rather than by open-coding separators, `Array.iteri`, or repeated
`Format.fprintf` fragments.

## Scope and non-goals

### In scope

- `pp_*` functions printing values, errors, shapes, graphs, reports, summaries
- repeated stringification paths using `Format.asprintf "%a" ...`
- repeated expect-test printer helpers (`pp_result`, captured formatter output)
- project-wide printer helper modules

### Out of scope for the first waves

- straight-line `Printf.printf` status lines with no compositional printer value
- source-code emitters whose primary job is generating OCaml/C text rather than
  rendering runtime values

Examples of the second category are:

- `lib/pytorch_schema/schema_codegen.ml`
- `lib/aten_gen/aten_emit.ml`
- `lib/aten_gen/aten_config_gen.ml`

Those modules should still be audited at the end, but they should not block the
main migration of runtime/value printers.

## Current state

Stage 0/1 (infrastructure) are done: `Core.Pretty` lives in
`lib/core/core.ml`/`.mli` (`to_string`, `option_or`, `result`, `error_kind`,
`core_result`, `capture_to_string`), and `lib/core/dune` depends on `fmt`.

Stage 2 (`lib/native`) and Stage 4 (`lib/pt2`, `lib/interp`) are **partial**,
not complete — an earlier note in this doc claimed otherwise; correcting that
here. What's actually migrated:

- Stage 2: `lib/native/{expr,payload,shape_error,symint,vec6}.ml`
- Stage 4: `lib/pt2/{pt2_archive,pt2_dtype,pt2_pickle,pt2_tensor,pt2_zip}.ml`,
  `lib/interp/{interp,interp_decode}.ml`

Raw `Format` printers still remain in `lib/native/{tensor,eval_direct,
graph_builder,ops/*}` and other Stage 4 CLI/report entry points
(`bin/native_graph.ml` had several migrated as part of a Stage-3-adjacent
fix, but not audited end-to-end for Stage 4 purposes).

Stage 3 (`lib/native_aten_bridge/*`, `lib/native_op_walk/*`,
`lib/aten_walk_recipes/*`, `lib/aten_native_verify/*`,
`lib/aten_spec_run/aten_spec_run.ml`) is now done — see "Stage 3 commits"
below.

There is also one architectural constraint that Stage 0/1 already resolved:

- `lib/core/dune` depends on `fmt`; the shared-helper story went with
  extending `lib/core` directly rather than a companion library.

### A pitfall found doing Stage 3: `Fmt.brackets` and `Fmt.float` are not drop-in

Two silent-but-real behavior changes surfaced migrating bracketed-list and
float printers:

- `Fmt.brackets pp` wraps its content in `box ~indent:1` — a **breakable**
  box. A bare `Format.fprintf ppf "[%a]" (Format.pp_print_list ~pp_sep:...)
  ...` (no box at all) never wraps regardless of line width; `Fmt.brackets`
  can, once the rendered width crosses the formatter's margin. Prefer
  `Fmt.pf ppf "[%a]" (Fmt.list ~sep:... ...) x` (no box) when the original
  was unboxed and must stay a single line — this bit `aten_spec_run.ml`'s
  `pp_op_call` (wrapped cram goldens) and `op_bridge.ml`'s `pp_int_list`.
- `Fmt.float` prints with `%g` (6 significant digits). `Format.pp_print_float`
  round-trips full precision (`string_of_float`-equivalent). These are not
  interchangeable — use `Format.pp_print_float` (calling it from an
  otherwise-`Fmt`-composed printer is fine; this migration targets ad hoc
  *composition*, not every use of a `Format` primitive) wherever the
  original value must survive exactly, e.g. scalar args round-tripped into
  a cram golden.

Also: `Fmt.comma` inserts a *breakable* `",@ "`, not a literal `", "` — safe
inside a box that's expected to wrap, wrong for a separator that must never
break. Use `Fmt.any ","`/`Fmt.any ", "` for a separator that must stay
literal.

### Stage 3 commits

- `native: preserve error backtraces through Build/map_error` — not a Stage 3
  printer migration, but the correctness bug (see "Result-crossing rules"
  below) found while working the same files.
- `core: add Core.or_raise, dedupe native_op_walk build failwiths`
- `native bridge: migrate reports and errors to Fmt helpers` (op_bridge.ml,
  tensor_bridge.ml, native_verify.ml, recipe_bounds.ml,
  aten_native_verify/{interp_verify,verify}.ml)
- `native bridge: avoid Fmt.brackets' breakable box in pp_int_list`
- `aten_spec_run: migrate value printers to Fmt`
- `native graph/tests: migrate remaining Option/Result printers`
  (bin/native_graph.ml, test/native_bridge_test.ml, plus a real bug fix in
  `pp_lens_printer` — see below)

One correctness fix rode along with the printer migration:
`bin/native_graph.ml`'s `pp_lens_printer` used `Result.value ~default` to
silently treat `` `Unknown_destination_tensor``/`` `Unknown_destination_node``
(an id outside the destination graph entirely — an invariant failure) the
same as "recorded but empty" (a folded constant, legitimately absent). Now
surfaced as `"provenance error: ..."` instead of silently rendering
"derived"; covered by a new expect test in `test/native/lens_test.ml`.

## Result-crossing rules (not just printing)

The same audit surfaced a sibling family of anti-patterns around
`Core.result` handling that this doc didn't originally scope (it covers
printing only) but that belong in the same convention. The project-wide audit
and prevention plan now lives in `.ai/result_option_migration_plan.md`; the
rules below remain the printer migration's local summary:

> **Found while implementing (result/option migration, Stage 1):** the bug this
> section warns about had a **live instance** in the tree —
> `lib/native/transform/graph_view.ml:352` rebuilt `D.validate_sig`'s
> `Core.result` by hand, so `Error.make` captured a fresh callstack at the
> re-raise and the dialect's detection site was lost. Fixed as a `fixup!` onto
> `6ce482f`, with a test that reverting the fix alone turns red. Writing the
> rule down was not enough to prevent it; see `.ai/result_option_census.md` for
> the complete wrapper-drop register and the deferred checker that would.

- **Never hand-rebuild an `Error` by unwrapping `e.Core.Error.kind`** when
  `Core.map_error` does exactly that while preserving the original detection
  backtrace:
  ```ocaml
  (* wrong: discards the backtrace, captures a new one at the wrong site *)
  | Error e -> Core.fail (`Build e.Core.Error.kind)
  (* right *)
  Graph_builder.build ... |> Core.map_error (fun e -> `Build e)
  ```
  `Stdlib.Result.map_error` is **not** a substitute for `Core.map_error` on a
  `Core.result` — it would map the whole `Error.t` wrapper, not just
  `.kind`, producing the wrong error shape.
- **Never open-code `match r with Ok x -> x | Error e -> failwith (...)`** at
  a boundary that can't consume `Core.result` (a fixed non-result signature,
  e.g. the walk framework's `build : pcg -> c -> subject * pcg`). Use
  `Core.or_raise : (Format.formatter -> 'e -> unit) -> ('a, 'e) result -> 'a`
  instead — one call site, payload-only message (no backtrace dump; this is
  meant to read like a normal exception, not a diagnostic). `Core.or_raise`
  is deliberately scoped to `Core.result` only; a handful of one-off
  `(string, string) result` boundaries (`lib/pt2_spec_gen/pt2_spec_gen.ml`,
  a few `test/*.ml` decode/encode helpers) are left as plain
  `match ... | Error e -> failwith e` — generalizing the helper for a
  handful of call sites with no `Core.Error.t` wrapper would be speculative
  abstraction.
- **Bridge an option into the framework with `Core.of_option`**, not
  `Stdlib.Option.to_result`: the latter yields a bare `Stdlib.result` with no
  `Core.Error.t`, so a bridged absence carries no backtrace. Its payload is
  built eagerly, before the option is inspected — keep an explicit match where
  building it raises, has effects, or costs something on the success path. On
  the ~20 graph-lookup sites it replaced, the cost is below measurement noise
  (measured; see `.ai/result_option_census.md`).
- **A deliberate drop of the wrapper needs a NAME.** Crossing out of the
  framework is legitimate — Cmdliner wants `(_, string) result`, and the
  pattern monad has its own `failure` type — but an anonymous
  `| Error e -> Error (… e.Core.Error.kind …)` is textually indistinguishable
  from the defect above. Two such boundaries are now named, commented helpers:
  `to_cli` in `bin/native_graph.ml` and `Pattern.of_core` in
  `lib/native/transform/pattern.ml`. `test/native/dce_test.ml:147` is a third,
  left inline behind the comment that already explained it.
- **`Option.value ~default:e` evaluates `e` eagerly.** Fine for a cheap total
  constant; for a raising, expensive, or effectful fallback keep explicit
  branching. The same caveat is why `Core.of_option`'s payload argument is
  documented as eager.

## Stable patterns found

### 1. Delegating error printers

Repeated in:

- `lib/interp/interp.ml`
- `lib/interp/interp_decode.ml`
- `lib/native_aten_bridge/op_bridge.ml`
- `lib/pt2/pt2_archive.ml`
- `lib/pt2/pt2_tensor.ml`
- `lib/native/shape_error.ml`

Stable shape:

```ocaml
let pp_error ppf = function
  | #Sub.error as e -> Sub.pp_error ppf e
  | `Case x -> Format.fprintf ppf "..."
```

Migration rule:

- keep row-polymorphic delegation structure
- replace ad hoc message assembly with `Fmt.pf`, `Fmt.string`, and helper
  combinators for common container shapes

### 2. Container printers built manually

Repeated in:

- `lib/pt2/pt2_tensor.ml`
- `lib/aten_spec_run/aten_spec_run.ml`
- `lib/native/expr.ml`
- `lib/native_aten_bridge/op_bridge.ml`
- `test/aten_ops_test.ml`

Stable shapes:

- bracketed lists
- comma or `"; "` separated lists
- optional values rendered as `"none"`
- small tuple/record summaries

Migration rule:

- centralize separators and bracket wrappers
- prefer `Fmt.list`, `Fmt.option`, and tiny reusable combinators over
  `Format.pp_print_list` lambdas repeated inline

### 3. Stringification from printers

Repeated in:

- `lib/native/tensor.ml`
- `lib/native_op_walk/*`
- `lib/native_aten_bridge/tensor_bridge.ml`
- `test/native/*`
- `bin/pt2_spec_gen.ml`

Stable shape:

```ocaml
Format.asprintf "%a" pp x
```

Migration rule:

- expose a single shared helper for printer-to-string conversion
- use it consistently in exception/error message construction

### 4. Result printers in tests

Repeated in:

- `test/native/core_test.ml`
- `test/native/compute_test.ml`
- `test/native/graph_test.ml`
- `test/native/graph_json_test.ml`
- `test/native/graph_symbolic_test.ml`
- `test/native/symbolic_test.ml`
- `test/native_bridge_test.ml`

Stable shape:

```ocaml
Fmt.result ~ok:pp_ok ~error:(fun ppf e -> pp_error ppf e.Core.Error.kind)
```

Migration rule:

- move the `Core.Error.kind` unwrapping pattern into a shared helper
- stop rebuilding the same lambda per test file

### 5. Capturing formatter output in expect tests

Repeated in:

- `test/native_walk_test.ml`
- `test/native/native_walk_test.ml`
- `test/aten_spec_run_test.ml`

Stable shape:

```ocaml
let capture f =
  let buf = Buffer.create ... in
  let ppf = Format.formatter_of_buffer buf in
  f ppf;
  Format.pp_print_flush ppf ();
  print_string (Buffer.contents buf)
```

Migration rule:

- create one shared helper for expect-test capture
- keep it out of per-file local boilerplate

## Shared helper module

If repeated helpers are introduced, they should live in a dedicated module under
`lib/core/`.

Preferred API shape:

- `Core.Pretty.to_string : 'a Fmt.t -> 'a -> string`
- `Core.Pretty.result : ok:'a Fmt.t -> error:'e Fmt.t -> ('a, 'e) result Fmt.t`
- `Core.Pretty.core_result :
    ok:'a Fmt.t -> error:'e Fmt.t -> ('a, 'e Core.result) Fmt.t`
- `Core.Pretty.option_or : none:string -> 'a Fmt.t -> 'a option Fmt.t`
- `Core.Pretty.brackets : 'a Fmt.t -> 'a Fmt.t`
- `Core.Pretty.parens : 'a Fmt.t -> 'a Fmt.t`
- `Core.Pretty.list_comma : 'a Fmt.t -> 'a list Fmt.t`
- `Core.Pretty.list_semi : 'a Fmt.t -> 'a list Fmt.t`

Possible test-only companion:

- `Core.Pretty.capture_to_string : (Format.formatter -> unit) -> string`

### Dependency decision

Before coding, make one explicit choice:

1. Preferred if acceptable:
   extend `lib/core` with a `fmt` dependency and add `Core.Pretty`.
2. If `core` must remain stdlib-only:
   add a companion library/module under `lib/core/` with the same API and make
   migration stages depend on that helper instead of the base `core` library.

This should be resolved in the infrastructure commit, not deferred.

## Migration rules

For the whole project:

- do not change public printer signatures unless there is a compelling reason
- migrate implementation internals, not the surface type of `pp`
- prefer one small `pp_*` helper per record/variant payload
- prefer `Fmt.pf` over `Format.fprintf` for compositional printers
- prefer `Fmt.list`/`Fmt.option`/`Fmt.result` over local loops and inline lambdas
- keep exact output stable where possible
- where output necessarily changes, update the relevant expect/cram snapshots in
  the same commit
- do not mix large behavioral refactors into printer commits

## Staged rollout

Each stage ends with:

1. targeted test run
2. promotions for changed expect/cram output
3. `make format`
4. one commit

### Stage 0: freeze scope and helper design

Deliverables:

- this plan in `.ai/`
- a short decision note in the implementation commit message describing whether
  `lib/core` itself now depends on `fmt` or whether a companion helper library
  was added

No code migration yet.

### Stage 1: infrastructure only

Deliverables:

- add the shared pretty-printing helper module under `lib/core/`
- wire the necessary dune dependency edges
- add focused expect tests for the helper module itself

Tests to add here:

- list/bracket rendering
- option rendering
- plain `result` rendering
- `Core.result` rendering through `e.Core.Error.kind`
- printer-to-string helper
- formatter capture helper if it is included in the shared module

Commit target:

- `core: add shared Fmt pretty-printing helpers`

### Stage 2: native core cleanup

Why first:

- `lib/native/` already contains the target style
- the dependency on `fmt` is already present
- this stage establishes the project-wide idioms before touching other areas

Migrate:

- `lib/native/expr.ml`
- `lib/native/shape_error.ml`
- `lib/native/symint.ml`
- `lib/native/vec6.ml`
- `lib/native/payload.ml`
- any remaining ad hoc `Format` printers inside `lib/native/ops/*`

Update tests:

- `test/native/*.ml`
- promote expect output where layout changes

Commit target:

- `native: migrate remaining printers to shared Fmt style`

### Stage 3: native bridge and walk/report layers — done

See "Stage 3 commits" and "A pitfall found doing Stage 3" under "Current
state" above for what actually shipped and the `Fmt.brackets`/`Fmt.float`
exactness pitfalls hit along the way.

Migrated:

- `lib/native_aten_bridge/*`
- `lib/native_op_walk/*` (all 17 `_nwalk.ml` walk builders, plus
  `native_verify.ml`)
- `lib/aten_walk_recipes/recipe_bounds.ml`
- `lib/aten_spec_run/aten_spec_run.ml` (value printers only — see the
  printer-vs-control-flow distinction it introduced, below)
- `lib/aten_native_verify/*`
- `lib/native_walk/op_walk.ml` — audited, nothing to migrate (a thin
  `Walk_core.Walk.run` wrapper with no printer of its own)

New distinction this stage established: not every `match` on `Option`/
`Result` inside a "printer-adjacent" function is a value printer. Several
functions here (`aten_spec_run.ml`'s `run`/`eval_print`/`eval_report`/
`walk_eval`/`compare_report`, and `pp_aten`/`pp_tensor_summary`'s
dtype-probing fallback) are operational control flow that also reports —
converting their `match` to `Fmt.result`/`Fmt.option` would obscure the
control flow for no benefit. Left as `match`; only the genuine value
printers (`pp_arg_value`, `pp_int_list`, `pp_op_call`, `pp_shape`,
`pp_source`, `pp_tensor_spec`, `pp_scalar_value`, `pp_native`,
`pp_interp_error`) were converted.

Update tests:

- `test/native_bridge_test.ml`
- any affected cram coverage (`test/pt2_node_bridge_cram.t` and similar)

Commit targets (see "Stage 3 commits" above for the full, corrected list —
this stage ended up split into several smaller commits rather than one, per
this doc's own commit-discipline rule):

- `native bridge: migrate reports and errors to Fmt helpers`

### Stage 4: pt2 and interpreter layers

Migrate:

- `lib/pt2/*`
- `lib/interp/*`
- related CLI/report entry points in `bin/`

Focus:

- error printers
- tensor/shape summaries
- any `Format.asprintf` paths used for user-visible errors

Update tests:

- `test/pt2_test.ml`
- `test/pt2_load.ml`
- `test/interp_run.ml`
- any cram tests whose expected output shifts

Commit target:

- `pt2 interp: migrate runtime printers to Fmt helpers`

### Stage 5: schema and ATen handwritten runtime printers

Migrate only human-facing printers in:

- `lib/aten_schema/*`
- `lib/aten_spec/*`
- `lib/aten_op_spec/*`
- `lib/walk_core/*`
- small printer helpers in `test/` and `bin/`

Keep this stage focused on runtime/value formatting, not code emission.

Update tests:

- parser/AST expect tests
- relevant cram tests such as `parse_cram.t`, `default_cram.t`,
  `model_cram.t`, `models_cram.t`, and any schema-facing tests whose output
  changes

Commit target:

- `schema aten: migrate handwritten runtime printers to Fmt`

### Stage 6: generator/emitter audit

Audit the remaining high-count `Format` users:

- `lib/pytorch_schema/schema_codegen.ml`
- `lib/aten_gen/aten_emit.ml`
- `lib/aten_gen/aten_config_gen.ml`

For each file, choose one of two outcomes:

1. migrate to `Fmt` because composition improves the implementation without
   obscuring source layout, or
2. keep on `Format` and document it as an intentional exception because the file
   is a source emitter rather than a runtime pretty-printer

This stage should end with the tree having no accidental ad hoc printer
patterns left.

Update tests:

- codegen expect tests
- any generator-facing tests

Commit target:

- `codegen: audit remaining Format emitters`

### Stage 7: final cleanup and documentation

Deliverables:

- remove now-dead local printer helpers
- unify any remaining duplicate test wrappers
- update `.ai/` docs to reflect the new project-wide conventions

Docs to update:

- `.ai/native_impl_plan.md`
- `.ai/native_graph_design.md`
- `.ai/testing_strategy.md`
- any design doc that still teaches `Format.pp_print_result` or ad hoc printer
  patterns as the preferred approach

Add one short new section somewhere authoritative stating:

- `Fmt` is the default composition layer for handwritten printers
- printer signatures remain `Format.formatter -> t -> unit`
- shared helpers live under `lib/core/`
- source emitters may stay on raw `Format` when justified

Commit target:

- `docs: record project-wide Fmt printing conventions`

## Test strategy during migration

Run only the affected surface per stage, then a final full pass.

Per-stage minimum:

- targeted inline expect tests
- targeted cram tests
- any stage-specific binaries exercised by those tests

Final pass after Stage 7:

```sh
make build
make runtest
make format
```

If working in a git worktree, use the worktree-safe `dune --root .` forms from
`.ai/worktree_setup.md`.

## Commit discipline

Do not batch multiple stages into one commit. The useful review unit here is
"one helper introduction or one area migration plus its promoted tests".

Expected commit sequence:

1. infrastructure helper commit
2. native migration commit
3. native bridge/report migration commit
4. pt2/interp migration commit
5. schema/ATen runtime printer migration commit
6. generator audit commit
7. documentation commit

If one area turns out to be too large, split that stage into sub-area commits,
but keep the same rule: code + tests + promotions together.

## Success criteria

The migration is complete when all of the following are true:

- handwritten runtime/value printers compose through `Fmt` or shared helpers
- repeated printer boilerplate has been removed from tests and libraries
- any remaining raw `Format` usage is either low-level formatter plumbing or a
  documented source-emitter exception
- expect/cram tests are updated and green
- `.ai/` documentation describes the new default printing style
