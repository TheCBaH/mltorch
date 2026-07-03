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

The tree is mixed:

- `lib/native/` already contains the best examples of the target style:
  `Graph_ir`, most `native/ops/*`, and `Op_config` use `Fmt` composition.
- many runtime printers still use ad hoc `Format`:
  `lib/interp/*`, `lib/pt2/*`, `lib/native_aten_bridge/*`,
  `lib/native/shape_error.ml`, `lib/native/expr.ml`, `lib/native/symint.ml`,
  `lib/aten_spec_run/aten_spec_run.ml`
- tests duplicate the same printer wrappers:
  multiple `let pp_result = Fmt.result ...` forms and several local `capture`
  helpers based on `Format.formatter_of_buffer`
- cram tests still mostly demonstrate `Format.pp_print_result`

There is also one architectural constraint:

- `lib/core/dune` currently documents `core` as stdlib-only with no external
  deps

That means the shared helper story must be decided deliberately rather than by
smuggling `Fmt` into `core` as part of an unrelated printer rewrite.

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

### Stage 3: native bridge and walk/report layers

Migrate:

- `lib/native_aten_bridge/*`
- `lib/native_op_walk/*`
- `lib/native_walk/*`
- `lib/aten_spec_run/aten_spec_run.ml`
- `lib/aten_native_verify/*`

These modules are heavy users of report-style printers, list formatting, and
stringification for failures. They should consume the Stage 1 helpers rather
than inventing local variants.

Update tests:

- `test/native_bridge_test.ml`
- `test/native_walk_test.ml`
- `test/native/native_walk_test.ml`
- `test/aten_spec_run_test.ml`
- any affected cram coverage

Commit target:

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
