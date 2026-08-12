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

### An exception may carry a typed row

`Native_interp` raises `Lower_error` internally: the lowering walk is deeply
recursive and threading a result through every arm would rewrite it, so the
exception is caught once, at `lower`'s boundary, and converted to an `Err.t`
there. That is a legitimate use of an exception for control flow — but what
crosses it is a **typed row**, not a rendered sentence.

It used to be `Printf.ksprintf` into one `` `Malformed_graph of string `` from 32
sites. `Native_interp.malformed` is now ten cases with structured payloads, and
`test/me_visualize_unsupported_cram.t` drives each from the smallest mutation of
a valid program that produces it. Two lessons generalise:

- **What the collapse hid.** `` `Unsupported_input `` covered both a non-tensor
  graph input and a user-input arity the runner cannot satisfy;
  `` `Tensor_bridge `` covered six shapes, two of which were
  `Format.asprintf "%a"` of a whole `Pt2_archive.error` — a seam crossed with
  the payload thrown away. Splitting them is what made those visible.
- **Not every raise is a recoverable error.** `add_env`'s arity check is an
  invariant of this module, not a fact about the model, so it is `invalid_arg`
  — the `Graph_ir.Index.assert_matches` precedent — and never reaches a caller
  dressed as a malformed graph.

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
and expect tests share a process; `test/native/core_test.ml`'s `with_config`
is the pattern.

The per-call-site rules — when to use `map_error` vs `map_kind`, where `~pos`
earns its place, why `guard`/`of_option`'s eagerness matters, and what a
named `Export` boundary looks like — live in `printer_conventions.md`, next to
the printing rules they interact with.

## Upstream fixes this adoption required

Both are in the vendored submodule, found by putting `Err` on a path Melange
reaches:

- `Stack.is_available`/`Stack.pp` called `Printexc.raw_backtrace_length`, which
  Melange compiles to `bt.length` while its `get_callstack` yields `undefined` —
  so *printing* an error threw a TypeError. Both now go through
  `Printexc.backtrace_slots`, which every backend answers safely.
- `Exn.pp_kind` was added: rendering a packed exception's payload without its
  provenance, which a wire-facing boundary needs and `Exn.pp` cannot give.

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
