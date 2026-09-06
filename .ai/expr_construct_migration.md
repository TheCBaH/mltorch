# Adding a new `Expr`/Region construct: the reusable extension checklist

This is the `Expr`/Region-level counterpart of `native_add_op.md` (which covers adding a
*Native operation*). Read that doc first if the new work is an op; read this one if it is a
new AST node, index/value primitive, or Region-level construct — the level the bounded scan
primitive itself landed at (`native_scan_design.md`), and the level LSTM's own `divmod`
encoding, `Scan_at`/`Local_scan_at` constructors, and the `Vector`/`Scan` binder namespaces
were built at. Project step 17 asks for this doc explicitly: "turn the migration inventory
into a reusable extension checklist."

## Why this is a distinct checklist from `native_add_op.md`

An op extension touches one dispatch table per site (`Graph_ir.op`, `Graph_shape`,
`Eval_op`, `Output_transfer`, ...) — a handful of files, each with one exhaustive match. A
new `Expr`/Region-level construct is different: it touches every function that already
walks the recursive representation — `fold.ml`, `rewrite.ml`, `value.ml`, the printer,
`compare`/`hash`, Region propagation — because each of those encodes binder behavior for
every existing construct already. Adding one more construct means teaching all of them
about it, not adding one more case to a shared table.

## 1. Verify migration completeness by construction, not by count

Do not track "migrated N of M call sites" as a number to reach. Make the compiler enforce
completeness instead:

- Add the new constructor to the closed `Expr.Value.t` (or `Index.t`/`Bool.t`) variant, in
  its alphabetical position, and build. Every non-exhaustive match across `fold.ml`,
  `rewrite.ml`, `value.ml`, the printer, and Region propagation now fails to compile — that
  is the migration inventory, and it disappears from the codebase (not "reaches zero") once
  every file builds again.
- Keep every one of those matches genuinely exhaustive, with **no default arm** — the same
  rule `Output_transfer.classify` and `Domain.check_node` already follow (see
  `native_add_op.md`, `native4d_design.md` §4.1). A defaulting match would let the *next*
  construct after this one skip the same discipline silently.
- A grep-based "count of call sites" is a useful sanity check while working, never the
  completion criterion — it undercounts sites reachable only through a shared helper and
  overcounts sites that do not actually need to know about the new construct at all (e.g. a
  reduction-only traversal that a scan, having no reducer of its own load, never reaches).

## 2. When a construct needs its own `freshen_*`-style splice helper

`Rewrite.freshen` already freshens every bound identity in a subtree before it is spliced
into a different context, for any construct with a single, uniform binder shape (see the
`Vector` case in `substitute_locals`, which calls plain `freshen` inline — no named helper,
because one namespace needs no special handling).

Give a construct its **own** named helper (the shape of `Rewrite.freshen_scan`,
`lib/expr_internal/rewrite.ml`) only when its binder shape is not a single namespace `freshen`
already handles uniformly — concretely, when splicing it standalone (not as part of a larger
expression being freshened top-down) requires wrapping it in a placeholder first so the
binders `freshen` needs to see are actually in scope. `Scan.t` needs this because a bare
`Scan.t` has no `Value.t` shape of its own to run `freshen` over directly; wrapping it in a
throwaway `Scan_at` gives `freshen` a real expression, and unwrapping after recovers the
freshened `Scan.t`. If a future construct is spliced the same way (standalone, not embedded),
check whether it needs the same wrap-freshen-unwrap shape before assuming plain `freshen`
suffices — that assumption failing silently is exactly what "unsafe independent composition"
in the design record this checklist replaces was warning about.

Do not generalize this into a fragment-import framework ahead of a second real case. One
call site (`freshen_scan`, used once) does not justify a public composition API; add the
second helper only when a second construct's binder shape actually needs it, and only then
consider whether the two share enough shape to warrant one generic operation.

## 3. The three fixture types every migrated construct needs

Each proves a different thing, and none substitutes for the others:

- **External public-API fixture** — build the construct only through its public
  constructors/builder, never by hand-assembling the internal representation, and check the
  result's shape/behavior end-to-end. Example: `test/native/region_scan_construction_test.ml`.
- **Observable-resource fixture** — the construct's cost (slots, depth, scan updates, ...)
  is measured through the same counters/limits a real caller would hit, not asserted from the
  formula alone. Example: `test/native/region_preflight_test.ml`; see also
  `test/native/lstm_scale_test.ml` for the same discipline applied at real operand scale
  (project step 16).
- **Executable ATen/oracle smoke fixture** — an end-to-end comparison against a real,
  independent oracle (ATen, or a hand-derived reference where no ATen op names the exact
  computation). Example: the generated ATen walk (`test/native_walk_coverage_test.ml`) and
  `test/native_bridge/lstm_test.ml`'s `Interp_verify.dispatch ~verify:true` fixtures.

## 4. Consolidating durable contracts

When a migration lands, fold whatever it proves into the tracked design record for that
construct (`native_scan_design.md` for scan-shaped work, this file for the general
checklist) in the *same* change, per this repo's "the plan is scaffolding, the design record
is the deliverable" rule. Retire the working-plan proposal that motivated the migration by
adding a `Status:` note pointing at what actually landed and why it stayed narrower or wider
than proposed — never by deleting the proposal's own rationale, which is often the only place
that records *why* a constraint exists at all.
