# Error handling: `Err` (err_trace)

The repo's recoverable-error framework is `Err`, the single module of the
`err_trace` library vendored at `vendored/err_trace` (name and public_name are
the same). It replaced `lib/core`'s hand-rolled `Core.Error.t` / `Core.result`.

`lib/core` still exists and still holds `Core.Pretty`. That is Fmt glue, and the
dependency runs one way only: `Core` may print an `Err.Error.t`, and `Err` may
never depend on Fmt — Melange's reachable closure is `walk_core` + `core`, and a
printer dependency inside the error library would put Fmt in every consumer's
closure. See `js_backends_design.md`.

## What the repo gets that `Core` did not have

- **The wrapper is abstract.** A transparent record let a caller write
  `{ e with kind = f e.kind }`, which typechecks, reads as a widening, and
  republishes the failure under someone else's backtrace.
  `graph_view.ml:352` was a live instance. `test/native/error_opacity.t` is the
  type-level proof that this is now rejected, with a control beside every
  negative case.
- **Provenance is a policy.** `Err.Config` chooses whether a stack is captured
  and which semantic transitions are retained, and `Err.Error.origin` is
  therefore an *option*. Nothing under `lib/` reads it or sets it.
- **Explicit positions.** `~pos:__POS__` is compiled in, so it is the only
  provenance that exists on the JavaScript backends.
- **Stack-safe traversal.** `Err.List` is tail-recursive; `Core.List.map` was
  not, and overflowed between 200k and 300k elements.
- **Structured exceptions.** `Err.or_raise` raises `Err.Exn.E`, carrying the
  whole wrapper, with a registered `Printexc` printer.

## Payloads: what a tag has to carry

A polymorphic-variant row tracks the *case*. It does not, by itself, track the
*data* — and a caller that cannot act on an error is not much better off than
one that got a string. `err_trace`'s `examples/poly_errors.ml` is the reference:
its `handle` branches on `` `A (`Missing_record { table; key }) `` and provisions
the missing row. Two rules make that possible here.

**A closed value set is a variant, not a string.** If the payload ranges over a
set the code knows in full, spell the set. `Me_limits.Field.t` (which aggregate
overran) and `Me_limits.Scope.t` (which validator was counting) were nine
independent sets of string literals before they were two types; a new ceiling
that reaches neither list is now a compile error rather than a new spelling.
Give such a type an `all` built by walking a successor chain — as
`Me_ids.Layer.all` and `Me_limits.Diagnostic.Code` do — so a constructor cannot
be added without reaching everything that iterates the vocabulary.

**A multi-field payload is a named record with named fields.** A tuple of
same-typed components documents its meaning in a comment, which the type checker
does not read: `` `Duplicate_node of string * string `` admits its two arguments in
either order. `lib/native/shape_error.mli` is the long-standing example
(`channels_mismatch`, `dims_mismatch`, `Window.t`), `Me_limits.Over_limit.t` the
recent one.

**A single-tag, single-id payload is already precise** — `` `Unknown_state of
string `` names its case and its payload is unambiguous. Do not churn these.

**Where the payload goes.** The base-versus-seam rule decides that, and it is
the same rule `poly_errors.ml` demonstrates:

- **Flat-include a shared base domain** (`[ Me_ids.error | … ]`,
  `[ Me_limits.over_limit_error | … ]`, `[ Graph_shape.error | … ]`). The
  example's `Base.error`, which both `A` and `B` include so a timeout is not
  rewrapped at every internal boundary. Nothing is wrapped and nothing is lost:
  where a base error came from belongs *in its payload* if it matters at all.
- **Wrap when crossing a real domain seam** (`` `Eval of Eval_direct.error ``,
  `` `Transform of Pass.error ``) — the example's `` `A of A.local_error ``, through
  `Err.map_error ~pos:__POS__` so the detection origin survives.
  `lib/native_interp/native_interp.ml` wraps its six upstream domains this way.

### An escape may carry a typed row

`Native_interp` exits its lowering walk through `Err.Escape`: the walk is deeply
recursive and threading a result through every arm would rewrite it, so it
throws from ~32 sites and `lower` establishes one `with_escape` frame. That is a
legitimate non-local exit — but what crosses it is a **typed row**, not a
rendered sentence.

It used to be `Printf.ksprintf` into one `` `Malformed_graph of string `` from 32
sites. `Native_interp.malformed` is now thirteen cases with structured payloads,
and `test/me_visualize_unsupported_cram.t` drives each from the smallest mutation
of a valid program that produces it — per *fault*, not per row, since
`` `Bad_dimension ``'s five faults and `` `Bad_config ``'s two are what a caller
branches on. Three lessons generalise:

- **What the collapse hid.** `` `Unsupported_input `` covered both a non-tensor
  graph input and a user-input arity the runner cannot satisfy;
  `` `Tensor_bridge `` covered six shapes, two of which were
  `Format.asprintf "%a"` of a whole `Pt2_archive.error` — a seam crossed with
  the payload thrown away. Splitting them is what made those visible.
- **Not every raise is a recoverable error.** `add_env`'s arity check is an
  invariant of this module, not a fact about the model, so it is `invalid_arg`
  — the `Graph_ir.Index.assert_matches` precedent — and never reaches a caller
  dressed as a malformed graph. It is therefore *not* a site for
  `Err.List.iter2 ~unequal_lengths` either: that would turn a defect here into
  an error row the caller is invited to handle.
- **And the converse, which is the easier one to get wrong.** A raise that *is*
  a fact about the model must not stay a raise. The engine's guarded
  types — `Op_config.Pos`, `Op_config.Nonneg`, `Dim.extent` — assert a trusted
  precondition and `invalid_arg` when it fails, which is right for a caller that
  built the value. `Native_interp` is not that caller: its strides, paddings,
  dilations and group counts are decoded from the model. Applied there they
  raised straight through the `Err.Escape` frame and out of `lower`, so a
  serialized `groups: 0` reached the Model Explorer boundary as an exception
  rather than as the row that boundary exists to classify. The fix is one
  approved route — `pos`/`nonneg`/`extent`/`dim_extent` in `native_interp.ml`,
  throwing `` `Bad_config `` (a config field) or `` `Bad_dimension `` (a
  metadata extent) — and the rule that no decoded argument may reach those
  constructors directly. `Dim.extent_checked` already existed for exactly this
  and simply was not being used. Two of the three sites were live and shipped;
  `test/native_interp/malformed_test.ml` pins each with the escape it used to
  produce.

## Escaping a recursive walk: `Err.Escape`

Five modules — `kernel_eval.ml`, `transform/ground_eval.ml`, `expr.ml` (twice)
and `native_interp.ml` — had independently declared the same private exception
plus catcher. They are now one idiom: `Err.Escape.with_escape` establishes the
frame, `throw` detects at the throw site, `throw_error` re-throws a wrapper
already built, and `or_throw` bridges an ordinary result-returning call into the
walk without threading results through its arms.

**The token is the point, and it is threaded, not global.** `Native_interp`'s
version carried a bare `error` row rather than an `Err.Error.t`, so the catch
rebuilt the wrapper with `Err.fail` and every malformed graph was reported as
detected at `lower`'s boundary instead of where the fault was found. It also let
`tensor_of_pt2`'s re-labelled row escape *uncaught* past a signature promising
`Err.t` — nothing checked, because nothing could. Both are unexpressible now:
the wrapper is built at the `throw` and the catch is by construction. The cost
is an `esc` parameter on ~20 lowering helpers, which is what makes the reach of
the escape visible in the types.

Two frictions are worth knowing before writing the next one, because both are
inherent rather than mistakes:

- **A token is invariant in its payload**, so a callee working at a narrower row
  cannot take its caller's token directly. `Expr.Eval.eval_index` fails at
  `index_error` and is called from `value` at the wider `error`, which passes
  `Err.Escape.map (fun e -> (e :> error)) esc`. The derived token exits the same
  frame and allocates no second generative exception, which is what keeps a
  nested `with_escape` off the per-output-pixel path.
- **`or_throw` is monomorphic**, so a helper handling *several* sub-rows at once
  still needs a three-line local wrapper coercing through `Error.t`'s
  covariance (`ground_eval.ml`'s `or_throw`, `expr.ml`'s `vchk`). `Escape.map`
  does not cover these: it produces a token at one narrower row, and these
  widen a different sub-row at each call. `map_error` is wrong too — it would
  record a `Map` event for a static coercion.

The first was reported upstream in `err_trace_accum.md` and is fixed; the second
is inherent (OCaml cannot quantify a row bound under an abstract `'e`) and is
documented upstream as the idiom to write.

### When an unreachable row is deleted, and when it is kept

Two rows in this migration had no test able to construct them, and they got
opposite treatments. The distinction is worth stating because "no witness" alone
does not decide it.

`Me_session.Session`'s `` `Dangling_edge `` was **deleted**. It had no
construction site *anywhere* — declared and printed, never raised — and its
condition was already caught by `` `Unknown_node `` through `Graph_index.node`.
It was a printer arm for a value nothing could build.

`Native_interp`'s `` `Output_not_evaluated `` was **kept**. It is raised, at both
run sites, and nothing checks the condition earlier: `Graph_builder.build` does
not verify that a graph's outputs are produced, and `Map_verify` runs only under
verification. No *model* can reach it — `lower` resolves every output through
`env_find`, so a lowered graph's outputs are produced by construction — but a
pass that drops a producer would, and this row is what stands between that and a
silent wrong answer.

So the test is not "can a test construct it" but **"is something else already
catching this, and what happens if nothing is"**. Record the answer where a
reader will meet it: `test/me_visualize_unsupported_cram.t` carries the argument
for `` `Output_not_evaluated `` beside the twelve rows that do have witnesses.

## Rules

**Configuration belongs to the host.** `Err` reads no environment: a library
that reconfigures itself from the ambient environment cannot be reasoned about
by the program linking it. `lib/err_host` parses `MLTORCH_ERROR_TRACE`,
`_BACKTRACE`, `_MAX_EVENTS`, `_MAX_FRAMES`, `_MAX_EXTERNAL_BYTES` and is called
exactly once, from an executable's entry point, before anything else runs.
Nothing under `lib/` other than that module may call `Err.Config.set`, and
nothing at all may call it as a module-load side effect.

Two behaviours there are deliberate. **No variable set means leave the policy
alone** — not the same as passing five `None`s to `Config.of_strings`, which
returns `Err`'s defaults and would silently override a policy the host had
already chosen. And **a bad setting is fatal** (exit 124, a usage error): a
silent fallback leaves the operator debugging under a policy they did not ask
for, which is the situation the knob exists to prevent.

**Monitors are the host's too.** A host may install more than one
(`Err.Monitor.install` returns a removable handle) and owns removal at shutdown
or in test cleanup — under `Fun.protect`, since the registry is process-wide.
Libraries install none. No monitor is installed in-tree today.

**Tests that change the policy must restore it.** `Err.Config` is process-wide
and expect tests share a process; `Err.Config.with_config` restores the previous
policy even when the body raises, and `test/native/core_test.ml` uses it. It is
not thread- or domain-scoped — it is for tests and single-threaded startup.

**`Config.fast` is not the production preset its name suggests.** It empties the
action set, so it discards the whole event trail and not merely stack capture.
The reproducible-diagnostics policy is `Config.deterministic` — boundary events
kept, no automatic stacks — which is `MLTORCH_ERROR_TRACE=boundaries` with
`MLTORCH_ERROR_BACKTRACE=off`, and gives identical output on native, bytecode,
js_of_ocaml and Melange wherever a call site supplied `~pos`.

**Validators short-circuit, and on untrusted input that is the bound.**
`Me_session.validate` and `Me_flow.validate` check their aggregate ceilings
*before* the walks that are linear in those counts — a bound checked after the
work it bounds is not a bound. `Err.Accum`, which runs every element by
contract, must therefore not be reached for above a ceiling that has not yet
passed.

**`Me_limits.Limits.check_against` is the one exception, and two conditions make
it one.** The work is bounded by a compile-time constant — `fields` and
`zip_fields`, 41 entries the document cannot influence — so "run every element"
is not something the input can turn into a denial of service. And each check is
an independent `Int64.compare`: none gates another, so there is no ordering to
preserve. It accumulates, and an operator with three bad fields learns about
three.

`Err.Accum.fold_errors` is what keeps that local. Accumulation would otherwise
be a one-way signature change — `('a, 'e Error.t list) Err.t` has no route back
to `('a, 'e) Err.t`, so the list would propagate through `Limits.validate`,
`Limits.create`, `Wire_limits.of_limits` and onto the wire. Instead the batch
collapses into this module's own `` `Invalid_limits of Invalid.t list ``, one
new row and one new `pp_error` arm; a single rejection still reports as
`` `Invalid_limit ``, so the common message is unchanged.

What that revealed is the argument for it: `me_limits_test.ml`'s nested-field
case passed the whole of `Pt2_zip.Limits.trusted`, widening five fields, and the
short-circuiting golden had been showing only the first for as long as the test
existed.

The per-call-site rules — when to use `map_error` vs `map_kind`, where `~pos`
earns its place, why `guard`/`of_option`'s eagerness matters, and what a
named `Export` boundary looks like — live in `printer_conventions.md`, next to
the printing rules they interact with.

## Upstream fixes this adoption required

All in the vendored submodule. The first two were found by putting `Err` on a
path Melange reaches:

- `Stack.is_available`/`Stack.pp` called `Printexc.raw_backtrace_length`, which
  Melange compiles to `bt.length` while its `get_callstack` yields `undefined` —
  so *printing* an error threw a TypeError. Both now go through
  `Printexc.backtrace_slots`, which every backend answers safely.
- `Exn.pp_kind` was added: rendering a packed exception's payload without its
  provenance, which a wire-facing boundary needs and `Exn.pp` cannot give.
- `protect`/`with_escape` re-raised through `Printexc.raise_with_backtrace`,
  which Melange does not implement — so under any stack-capturing policy,
  including the default, every foreign exception crossing `protect` on Melange
  was *replaced* by a "not polyfilled" error. They now reattach a backtrace only
  when it has resolvable frames.
- `Config.pp_backtrace` printed `never` while `of_strings` accepted only `off`,
  so a logged policy could not be fed back. `off` is now both the printed and
  the accepted spelling on both axes; `err_host.mli` documents `never` as the
  accepted alias, and `test/err_host_test.ml`'s golden moved with it.

Three more came from this repo's own adoption feedback (`err_trace_accum.md`),
and each removed a workaround that is now gone from the tree:

- `Accum.fold_errors` and `Accum.lift` — accumulation used to be a one-way
  signature change, which is why `check_against` short-circuited over 41
  independent checks. See the rule above.
- `Escape.map` — a token viewed through a narrower row, which replaced the
  `throw` callback `Expr.Eval.eval_index` had to take.
- `List.map2`/`iter2` now validate both lengths **before** the first callback,
  so a mismatch produces no partial effects and a callback failure cannot hide
  it. That retires the caveat on `22f141d`: a prefix failure no longer outranks
  the arity error in `graph_map.ml` or `graph_view.ml`.

The `devel` API this repo vendors also added `payload`, `import`, `export`,
`Error.pp_kind`, `List.filter_map`/`exists`/`for_all`, `Config.deterministic`,
`Config.with_config` and `Err.Make`. All are adopted above except `Err.Make`:
the binder does accept this repo's `[< error ] -> unit` printers — upstream's
wording was corrected after this repo reported it — but the files holding the
204 `~pp_error` uses each touch four to six domains, so binding them trades one
labelled argument for several functor applications.

## Not built: JavaScript stack import

`Err.Stack.of_external` exists for importing a foreign runtime's stack, and an
earlier plan called for js_of_ocaml (`Js_error.stack`) and Melange
(`Js.Exn.stack`) adapters.

**There is no such boundary in this repo.** No OCaml here is compiled into a
browser worker: nothing links `js_of_ocaml` as a *library*, no OCaml code
handles `postMessage` or returns a Promise, and `web/src/session.js` is plain
JavaScript driving the upstream renderer against a session document the native
CLI produced ahead of time. The jsoo build exists as a differential probe, not
as a deployment.

Adapters would therefore have had zero call sites. When an OCaml worker does
appear, the rules already settled are: the adapter goes beside the worker and
**never** in the base `Err` library (it would put a browser dependency in
Melange's pure closure); expected failures resolve as outcome values rather
than Promise rejections; unclassified throws escape unchanged; and no OCaml
runtime representation crosses the public boundary.
