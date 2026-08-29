# Printer and result-crossing conventions

How this project renders values and errors, and how an `Err.t` is allowed to
cross a boundary that cannot carry one. These are the durable rules; the staged
record of the migration that produced them is a local working document
(`fmt_migration_plan.md`), not part of the tracked design record.

## Shared helpers

`lib/core/core.mli` exposes the printer helpers; `lib/core/dune` depends on `fmt`:

```ocaml
module Pretty : sig
  val to_string : 'a Fmt.t -> 'a -> string
  val option_or : none:string -> 'a Fmt.t -> 'a option Fmt.t
  val result : ok:'a Fmt.t -> error:'e Fmt.t -> ('a, 'e) Stdlib.result Fmt.t
  val error_kind : 'e Fmt.t -> 'e Err.Error.t Fmt.t
  val err_result : ok:'a Fmt.t -> error:'e Fmt.t -> ('a, 'e) Err.t Fmt.t
  val capture_to_string :
    ?like:Format.formatter -> (Format.formatter -> unit) -> string
end
```

The public printer shape stays `val pp : Format.formatter -> t -> unit` — `Fmt.t` is
that same type, so composition costs no signature churn.

## Compose printers; don't open-code them

Five shapes recur across the tree. Each has a combinator; use it rather than
re-deriving the shape inline.

### 1. Delegating error printers

`lib/interp/interp.ml`, `lib/interp/interp_decode.ml`,
`lib/native_aten_bridge/op_bridge_error.ml` (split from op_bridge.ml), `lib/pt2/pt2_archive.ml`,
`lib/pt2/pt2_tensor.ml`, `lib/native/shape_error.ml`:

```ocaml
let pp_error ppf = function
  | #Sub.error as e -> Sub.pp_error ppf e
  | `Case x -> Fmt.pf ppf "..."
```

Keep the row-polymorphic delegation; assemble the message with `Fmt.pf` and
combinators, not by hand.

**A shared domain is printed by its owner, once.** Delegation is not only for a
whole included row — an individual tag whose payload is another module's type
delegates too. `Me_limits.over_limit_error` is the worked example: nine
validators flat-include it and each renders it as

```ocaml
| `Over_limit o -> Me_limits.Over_limit.pp fmt o
```

rather than re-deriving the sentence from the payload's fields. Before that,
nine copies of one `Fmt.pf` differed only in a leading word, and that word — the
validator that rejected — was information *the error did not carry*. Whenever
re-deriving a message would need a fact the payload lacks, the fix is the
payload, not the printer.

### 2. Container printers

Bracketed lists, `", "`/`"; "`-separated lists, optional values rendered as
`"none"`, small tuple summaries. Use `Fmt.list`, `Fmt.option`,
`Core.Pretty.option_or`; centralise separators and brackets rather than repeating
`Format.pp_print_list` lambdas inline.

### 3. Stringification from a printer

`Format.asprintf "%a" pp x` becomes `Core.Pretty.to_string pp x` — one helper,
used consistently when building exception and error messages.

### 4. Result printers in tests

`test/native/*`, `test/native_bridge/*` and friends all need the same
`Err.Error.kind` unwrap:

```ocaml
Core.Pretty.err_result ~ok:pp_ok ~error:pp_error
```

Don't rebuild the `fun ppf e -> pp_error ppf (Err.Error.kind e)` lambda per file.

### 5. Capturing formatter output in expect tests

Use `Core.Pretty.capture_to_string`; keep the buffer/flush boilerplate out of
per-file helpers.

## Fmt combinators that are not drop-in replacements

Three silent behaviour changes, each found by a cram golden that moved:

- **`Fmt.brackets pp` wraps its content in `box ~indent:1`** — a *breakable* box.
  A bare `Format.fprintf ppf "[%a]" ...` with no box never wraps, whatever the
  width. When the original was unboxed and must stay on one line, write
  `Fmt.pf ppf "[%a]" (Fmt.list ~sep:... ...) x`. This bit `aten_spec_run.ml`'s
  `pp_op_call` and `op_bridge_error.ml`'s `pp_int_list` (split from
  op_bridge.ml).
- **`Fmt.float` prints `%g`** (6 significant digits); `Format.pp_print_float`
  round-trips full precision. They are not interchangeable — keep
  `Format.pp_print_float` wherever the value must survive exactly, e.g. a scalar
  argument round-tripped into a cram golden. Calling a `Format` primitive from an
  otherwise-`Fmt`-composed printer is fine; what these conventions target is ad hoc
  *composition*.
- **`Fmt.comma` inserts a breakable `",@ "`**, not a literal `", "`. Safe inside a
  box that is expected to wrap, wrong for a separator that must never break — use
  `Fmt.any ","` / `Fmt.any ", "` there.

## Result-crossing rules (not just printing)

> The first rule below had a **live instance** in the tree while the rule was
> already written down: `lib/native/transform/graph_view.ml:352` rebuilt
> `D.validate_sig`'s result by hand, so `Error.make` captured a fresh
> callstack at the re-raise and the dialect's detection site was lost. Fixed with
> a test that reverting the fix alone turns red. Writing the rule down was not
> enough to prevent it, which is the argument for the checker this project has
> not yet built.

- **The `Error.t` wrapper is abstract, and rebuilding it is now a type error.**
  It used to be a transparent record, which made
  `| Error e -> Err.fail (`Build (Err.Error.kind e))` typecheck, read like a
  widening, and silently re-capture provenance at the re-raise. `Err.map_error`
  is the supported spelling and keeps the original detection origin:
  ```ocaml
  Graph_builder.build ... |> Err.map_error (fun e -> `Build e)
  ```
  `test/native/error_opacity.t` is the evidence that the type checker enforces
  this, paired-control style like `version_safety.t`.
  `Stdlib.Result.map_error` is **not** a substitute for `Err.map_error` on an
  `Err.t` — it would map the whole `Error.t` wrapper, not just the payload,
  producing the wrong error shape. `Err.Error.map_kind` is not an untraced
  escape hatch either: it records the same `Map` event.
- **Add `~pos:__POS__` at subsystem seams, not everywhere.** It is the only
  provenance that survives on the JavaScript backends, where no callstack is
  captured at all, so the seams the browser path crosses carry one
  (`Pt2_archive`, `Native_interp`, `Me_export`). Per-row coercions do not:
  every `Map` event is retained memory and printed output.
- **Never open-code `match r with Ok x -> x | Error e -> failwith (...)`** at a
  boundary that cannot consume an `Err.t` (a fixed non-result signature, e.g.
  the walk framework's `build : pcg -> c -> subject * pcg`). Use
  `Err.or_raise ~pp_error:pp` — one call site, and the exception it raises is
  `Err.Exn.E`, which carries the whole wrapper and registers a `Printexc`
  printer. The handful of one-off `(string, string) result` boundaries
  (`lib/pt2_spec_gen/pt2_spec_gen.ml`, a few `test/*.ml` decode/encode helpers)
  stay plain `match ... | Error e -> failwith e` — generalising the helper for
  call sites with no `Err.Error.t` wrapper would be speculative abstraction.
  Note the *dependency*: `js/probe/probe_tensor_json.ml` justified its `failwith`
  in a comment reading "plain `result` from Jsont" — true until `Graph_json`
  started returning an `Err.t`, at which point the exemption silently stopped
  applying. When a channel is lifted into the framework, its consumers' `failwith`
  exemptions expire with it.
- **A third-party message may stay a string, but name the tag for its source.**
  Jsont's decode error is a position and an expectation this repository did not
  author and cannot classify further, so `Graph_json.error` and
  `Me_response.Meta.error` are both `` `Jsont of string ``, and `Me_export`'s is
  `` `Model_json_decode of string ``. What the name buys is that it does not read
  as a case the module *declined* to classify. **Lifting it into `Err.t` is still
  worth doing** even when the payload cannot improve: callers compose with
  `let*` instead of re-matching at each entry point, and they get a wrapper with
  provenance. That is a different judgement from dropping the wrapper outward
  (`to_cli`, `or_jsont`), which remains correct where the far side genuinely
  cannot carry one.
- **Flat-including a base domain replaces a rendered payload, not just a tag.**
  `Tensor_bridge.of_aten` returned `(_, string) result` and built it by
  destructuring an `Err.t` to `Fmt.str "%a"` its kind — the wrapper discarded and
  the caller left a sentence. It now flat-includes `Aten_shape.error` and widens
  with `Err.map_error`, so `op_bridge`'s `` `Tensor_bridge `` carries the bridge's
  own row (`cause`) instead of a `message : string`.
- **A boundary that emits outward must print the payload ALONE.** Because
  `Err` registers a `Printexc` printer, the default rendering of `Err.Exn.E` —
  and therefore `Printexc.to_string` — is payload *plus* detection stack *plus*
  semantic trace. That is a developer diagnostic. Anything a browser, a wire
  response, or a third party reads must match `Err.Exn.E` and use
  `Err.Exn.pp_kind`; see `Me_limits.Diagnostic.of_exn`, whose test fails if the
  match is removed.
- **Bridge an option into the framework with `Err.of_option`**, not
  `Stdlib.Option.to_result`: the latter yields a bare `Stdlib.result` with no
  `Err.Error.t`, so a bridged absence carries no provenance. Its payload is built
  eagerly, before the option is inspected — use `Err.map_none ~error:(fun () -> ...)`
  where building it raises, has effects, or costs something on the success path
  (`graph_view.ml`'s `sig_of`, whose payload is an unbounded functor callback).
  On the ~20 graph-lookup sites `of_option` replaced, the cost is below
  measurement noise (measured).
- **`Err.guard` is also eager**, and for the same reason is NOT used by the
  per-node and per-edge limit checks in `lib/model_explorer_export`: it takes
  `~error:'e` rather than a thunk, so it would allocate the payload on every
  successful check in the hottest export loop. The explicit
  `if over then fail else return ()` is the faster form there.
- **A deliberate drop of the wrapper needs a NAME, and goes through
  `Err.export`.** Crossing out of the framework is legitimate — Cmdliner wants
  `(_, string) result`, Jsont wants a raised decode error, and the pattern monad
  has its own `failure` type — but an anonymous
  `| Error e -> Error (… Err.Error.kind e …)` is textually indistinguishable
  from the defect above. The named helpers are `to_cli` (`bin/native_graph_common.ml`),
  `Pattern.of_err`, `Quant.of_err_for_jsont`, `Me_request.or_jsont`,
  `Pass.Outcome.to_jsont`, and `Me_limits`' wire decoder; each is
  `Err.export ~pos:__POS__`, which marks `Export` and *then* unwraps, so the
  marking is part of the operation rather than something the helper has to
  remember — `to_jsont` is the one that had forgotten it.
  `Err.export` keeps the payload **typed**; rendering it is the boundary's own
  decision, and a boundary that only wants to render uses `Err.Error.pp_kind`.
  `test/native/dce_test.ml:147` is left inline behind the comment that already
  explained it.
- **The inbound twin is `Err.import`, not `Err.fail`.** Lifting a third party's
  bare `(_, string) result` into the framework — `Graph_json.of_jsont`,
  `Me_response.Meta.decode`, `Pt2_archive`'s three `Jsont_bytesrw` decodes —
  records `Import` and no `Detect`, because the failure was detected by whoever
  produced the result. `Err.fail` there would claim this module found it.
  `Err.payload` is the *unmarked* unwrap, for tests and assertions, so uses that
  are not boundaries do not dilute the marked form.
- **`Option.value ~default:e` evaluates `e` eagerly.** Fine for a cheap total
  constant; for a raising, expensive, or effectful fallback keep explicit
  branching — or use `Err.map_none`, which is that caveat expressed in a type.
