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
