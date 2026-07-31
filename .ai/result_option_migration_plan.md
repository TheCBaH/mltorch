# Result and Option migration plan

## Goal

Replace mechanically open-coded `Result` and `Option` matches with the
project's existing high-level operations:

- `Core.Syntax` (`let*`, `let+`) and `Core.List` for `Core.result` pipelines
- `Core.map_error` when widening a `Core.result` error row
- `Core.or_raise` at a deliberate exception boundary
- `Result`/`Option` mapping, folding, iteration, predicates, conversions, and
  defaults for plain standard-library values
- `Fmt.result`, `Fmt.option`, and `Core.Pretty` for value rendering

The goal is **not** to make `git grep Error` or `git grep None` empty. Those
constructors remain necessary in types, parsers, recovery logic, and real
branching. The completion criterion is that no match which is merely spelling a
standard operation by hand remains.

This plan is a companion to `.ai/fmt_migration_plan.md`. That plan owns general
printer composition; this one owns the broader `Result`/`Option` control-flow
audit, including printer-shaped matches found by the same search.

## Baseline

At the time this plan was written:

```text
git grep -n Error     650 lines in 127 files
git grep -n None      887 lines in 158 files
```

Restricting the census to an `Error` or `None` match arm in tracked OCaml source
still finds 615 arms in 123 files:

```text
304 Error arms
311 None arms
402 arms under lib/ and bin/
213 arms under test/
```

The largest production concentrations are `lib/native`, `lib/aten_gen`,
`lib/native4d`, `lib/aten_spec_run`, `lib/native_interp`, `lib/interp`, and
`lib/pt2`. These numbers are discovery data, not a quality metric: a type
declaration and a reducible match both contribute to the raw grep, while an
operational match may be exactly the right code.

### Classified baseline (Stage 0, done)

The raw figures above are preserved as measured. The semantic census that
Stage 0 asks for now exists — `.ai/result_option_census.md` (readable) and
`.ai/result_option_census.tsv` (603 rows, one per site, each in exactly one
cluster). Reproducible arm-head count at `d4d7e1f`:

```sh
git grep -cE '^[[:space:]]*\|[[:space:]]*(Error|None)\b' -- '*.ml' '*.mli' \
  | awk -F: '{s+=$2} END{print s}'      # 603, in 120 of 301 tracked files
```

| verdict | n |
|---|---:|
| `defect` — loses a detection backtrace | 1 |
| `reduce:*` — mechanically reducible | 95 |
| `audit:*` — needs a decision | 13 |
| `exempt-lowering` — deliberate wrapper drop at a boundary | 15 |
| `emitted-text` — generated OCaml inside `{\| … \|}` | 15 |
| `keep:*` — real branching, error leaves, combinator definitions | 464 |

**The migration is 96 sites**, not 615: 82% of the tree's `Error`/`None` arms
are already correct, and the largest single cluster (160 printer arms) is 85%
legitimate. Sizing this work from the raw grep overstates it roughly six-fold.

The census also turned up one thing this plan's framing missed: the tree
contains a **live instance of the backtrace-losing bug** that
`.ai/fmt_migration_plan.md` documents as wrong — `lib/native/transform/
graph_view.ml:352` — so this is not purely a style migration.

## Classification rule

Check every match in context before changing it. Classify it by what the
function is doing, not merely by its constructors.

| Open-coded intent | Preferred form |
|---|---|
| Sequence a fallible computation | `let*` / `Result.bind` |
| Transform only the success value | `let+` / `Result.map` |
| Transform a `Core.result` error payload | `Core.map_error` |
| Test success/failure only | `Result.is_ok` / `Result.is_error` |
| Deliberately discard a plain result's error | `Result.to_option` |
| Apply a side effect only on success/error | `Result.iter` / `Result.iter_error` |
| Render a result value | `Fmt.result` / `Core.Pretty.result` / `Core.Pretty.core_result` |
| Cross a `Core.result` into exceptions | `Core.or_raise` |
| Supply a cheap, total option default | `Option.value ~default` |
| Transform or flatten an option | `Option.map` / `Option.bind` / `Option.join` |
| Test presence only | `Option.is_some` / `Option.is_none` |
| Apply a side effect only when present | `Option.iter` |
| Convert an option | `Option.to_result` / `Option.to_list` as appropriate |
| Render an option value | `Fmt.option` / `Core.Pretty.option_or` |

`Core.result` needs special care. `Stdlib.Result.map_error` maps the complete
`Core.Error.t` wrapper; it must not replace `Core.map_error`, which maps only the
payload and preserves the original detection backtrace.

`Option.value ~default:e` evaluates `e` eagerly. Use it only when `e` is cheap,
total, and has no effects. Keep explicit branching for a raising, expensive, or
effectful fallback unless the census demonstrates enough repetition to justify
one small lazy-default helper. Do not add such a helper speculatively.

## Matches that should remain explicit

Retain a match when the constructors select genuinely different behavior:

- inspecting a particular error payload, such as treating `Unhandled_op`
  differently from other failures
- recovery, fallback, retry, backtracking, or deliberately continuing after an
  error
- combining several results and choosing which error or partial result wins
- distinguishing multiple nested states such as `None`, `Some (Error _)`, and
  `Some (Ok _)`
- preserving a lazy or effectful default
- refining the payload shape, validating an invariant, or changing domain state
- implementing the project's combinators themselves
- emitting source text which contains `Error`/`None` patterns

Printer-adjacent code is not automatically a value printer. If each branch
updates state, controls traversal, asserts, or chooses subsequent work, the
match is operational control flow and should remain. Pure rendering of an
already-computed `option` or `result` should use a specialized printer.

Non-obvious retained matches should get a short comment explaining the semantic
distinction. Obvious domain branching does not need lint-directed ceremony.

## Rollout

### Stage 0: produce a semantic census — **done**

> **Found while implementing:** the census is `.ai/result_option_census.md` and
> `.tsv`. Three things it changed about this plan's assumptions:
>
> - The reducible surface is **96 sites**, not ~615. Staging the work bottom-up
>   by dependency layer (Stage 3 below) is the wrong shape: `lib/` holds ~40 of
>   them, and two thirds sit in `test/native`, `test/native4d` and
>   `bin/native_graph.ml`. `lib/native4d`, `lib/native_op_walk` and `lib/interp`
>   are already at the target idiom.
> - One site is an outright **defect**, not a style issue
>   (`graph_view.ml:352`), and it is fixed first, as a `fixup!` onto `6ce482f`.
> - A wrapper drop can be written by **destructuring in the pattern**
>   (`lib/native_aten_bridge/tensor_bridge.ml:34`:
>   `| Error { Core.Error.kind = e; _ } ->`), so it never contains the string
>   `e.Core.Error.kind`. Any search — or future checker — keyed on that
>   expression misses it.

Original intent, kept for the record:

Create an inventory grouped by the classification table rather than one flat
grep dump. Record, per area:

- mechanical replacement candidates
- deliberate matches
- uncertain matches requiring an owner/design decision
- error-discarding conversions, which deserve extra scrutiny
- `Core.result` matches which unwrap `e.Core.Error.kind`

Do not infer that converting `Error _` to `None`, `false`, an empty collection,
or a placeholder string is harmless. Confirm that the caller intentionally
forgets the diagnostic. In tests, prefer surfacing fixture/setup failures with
`Core.or_raise` instead of silently turning them into absence.

The census should also separate handwritten runtime code, tests, generators,
and generated/emitted source. This keeps generator templates from being
mistaken for runtime anti-patterns.

### Stage 1: close only demonstrated helper gaps

Use the existing `Core` surface first. It already supplies result syntax,
backtrace-preserving error mapping, result-aware list traversal, exception
crossing, and specialized printers.

Add a helper only when the census finds the same semantic operation repeatedly
and the standard library cannot express it without losing laziness, diagnostics,
or a `Core.Error.t` backtrace. Each new helper needs focused tests covering:

- success and failure/absence
- evaluation of lazy fallbacks only on the absent path
- preservation of the original `Core.Error.backtrace`
- exact printer output where applicable

Do not add aliases for standard `Result` or `Option` functions.

### Stage 2: remove mechanical test boilerplate

Start with tests because their recurring patterns are concentrated and their
expected output makes printer changes easy to review:

- replace fixture-building `Ok`/`Error -> failwith` matches with
  `Core.or_raise`
- replace boolean constructor checks with `Result.is_*` and `Option.is_*`
- replace deliberate conversions with `Result.to_option`/`Option.to_result`
- replace pure expect-test rendering with `Core.Pretty.core_result`,
  `Core.Pretty.result`, or `Core.Pretty.option_or`
- retain matches that assert on a specific payload or exercise a particular
  branch

Keep fixture construction failures visible. A helper named as a conversion to
option must not conceal an unexpected build/evaluation failure.

### Stage 3: migrate production pipelines by dependency layer

Work bottom-up so each area can reuse conventions established below it:

1. `lib/core` audit, excluding the combinator implementations themselves
2. native graph/transform and Native4D
3. PT2, interpreter, native interpreter, and native bridge
4. command-line entry points and report runners
5. schema and ATen handwritten runtime code
6. generators and emitters, auditing generated runtime patterns separately

Within each area, split changes by semantic family rather than by grep keyword:

1. transparent propagation and success mapping
2. error-row widening and exception boundaries
3. option defaults, maps, iteration, predicates, and conversions
4. specialized value printers

This ordering keeps backtrace/error-preservation changes separate from output
layout changes. Use small commits per area/family, with tests and any promoted
expect/cram output in the same commit.

### Stage 4: add a narrow regression check

Add a small syntax-aware checker for exact, semantics-preserving anti-patterns.
It should inspect formatted OCaml syntax and reject only patterns for which the
replacement is unambiguous, for example:

- `Ok` success followed by unchanged `Error` propagation
- `Some` mapping followed by unchanged `None`
- `Some`-only side effects with a no-op `None` arm
- constructor-only presence/success predicates
- cheap constant defaults returned unchanged from `Some`
- rebuilding a `Core.result` from `e.Core.Error.kind` where
  `Core.map_error` applies
- `Core.result` to `failwith` boundaries where `Core.or_raise` applies

Do not make raw grep counts a CI gate and do not ban all matches on these
constructors. A text-only ban would reject correct recovery logic and miss
equivalent formatting. The checker should have unit fixtures for both rejected
and intentionally accepted shapes, and should exclude the definitions of the
combinators it recommends.

Run the checker in report/ratchet mode during migration so no category grows.
Once a category reaches zero, make it a hard failure. At the end, wire the
fully-zero mechanical categories into a `make lint` target and CI. Any narrow
suppression mechanism must require a reason adjacent to the match; do not keep a
line-number allowlist.

### Stage 5: final audit and documentation

Repeat the original greps and review every remaining match-arm hit by category.
The final report should explain why the high-count remaining areas are
legitimate, rather than presenting a misleading target of zero constructor
mentions.

Update `.ai/fmt_migration_plan.md`, `.ai/jsont_patterns.md`, and other affected
design records so examples teach the new forms. In particular, document the
eager-default limitation of `Option.value` precisely: cheap values are fine;
raising, expensive, and effectful expressions require lazy branching.

## Verification

For every migration batch:

1. run the most local expect/cram tests
2. add or preserve a negative-path test when error propagation changes
3. verify `Core.Error.backtrace` preservation when replacing error wrapping
4. run `make format`
5. inspect the edited source after formatting

At the end of each area, run its Dune test subtree. At the end of the rollout:

```sh
make build
make runtest
make lint
make format
```

For a finding about a vacuous check or swallowed error, prove the test can fail:
temporarily restore the old behavior, observe the targeted test fail, then
restore the fix.

## Success criteria

The migration is complete when:

- the semantic census has no unresolved entries
- every mechanically reducible `Result`/`Option` match uses a standard or
  project helper
- `Core.result` error mapping preserves the original detection backtrace
- deliberate error-to-option/default conversions are named and tested
- pure option/result rendering uses specialized printers
- remaining explicit matches perform real branching and are clear in context
- the syntax-aware lint has no unsuppressed findings and runs in CI
- targeted tests, `make build`, `make runtest`, and `make format` pass

