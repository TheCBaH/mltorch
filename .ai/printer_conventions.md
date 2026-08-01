# Printer and result-crossing conventions

How this project renders values and errors, and how a `Core.result` is allowed to
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
  val error_kind : 'e Fmt.t -> 'e Error.t Fmt.t
  val core_result : ok:'a Fmt.t -> error:'e Fmt.t -> ('a, 'e) result Fmt.t
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
`lib/native_aten_bridge/op_bridge.ml`, `lib/pt2/pt2_archive.ml`,
`lib/pt2/pt2_tensor.ml`, `lib/native/shape_error.ml`:

```ocaml
let pp_error ppf = function
  | #Sub.error as e -> Sub.pp_error ppf e
  | `Case x -> Fmt.pf ppf "..."
```

Keep the row-polymorphic delegation; assemble the message with `Fmt.pf` and
combinators, not by hand.

### 2. Container printers

Bracketed lists, `", "`/`"; "`-separated lists, optional values rendered as
`"none"`, small tuple summaries. Use `Fmt.list`, `Fmt.option`,
`Core.Pretty.option_or`; centralise separators and brackets rather than repeating
`Format.pp_print_list` lambdas inline.

### 3. Stringification from a printer

`Format.asprintf "%a" pp x` becomes `Core.Pretty.to_string pp x` — one helper,
used consistently when building exception and error messages.

### 4. Result printers in tests

`test/native/*`, `test/native_bridge_test.ml` and friends all need the same
`Core.Error.kind` unwrap:

```ocaml
Core.Pretty.core_result ~ok:pp_ok ~error:pp_error
```

Don't rebuild the `fun ppf e -> pp_error ppf e.Core.Error.kind` lambda per file.

### 5. Capturing formatter output in expect tests

Use `Core.Pretty.capture_to_string`; keep the buffer/flush boilerplate out of
per-file helpers.

## Fmt combinators that are not drop-in replacements

Three silent behaviour changes, each found by a cram golden that moved:

- **`Fmt.brackets pp` wraps its content in `box ~indent:1`** — a *breakable* box.
  A bare `Format.fprintf ppf "[%a]" ...` with no box never wraps, whatever the
  width. When the original was unboxed and must stay on one line, write
  `Fmt.pf ppf "[%a]" (Fmt.list ~sep:... ...) x`. This bit `aten_spec_run.ml`'s
  `pp_op_call` and `op_bridge.ml`'s `pp_int_list`.
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
  `Core.result` — it would map the whole `Error.t` wrapper, not just `.kind`,
  producing the wrong error shape.
- **Never open-code `match r with Ok x -> x | Error e -> failwith (...)`** at a
  boundary that cannot consume a `Core.result` (a fixed non-result signature, e.g.
  the walk framework's `build : pcg -> c -> subject * pcg`). Use
  `Core.or_raise : (Format.formatter -> 'e -> unit) -> ('a, 'e) result -> 'a` —
  one call site, payload-only message (no backtrace dump; it should read like a
  normal exception, not a diagnostic). `Core.or_raise` is deliberately scoped to
  `Core.result`; the handful of one-off `(string, string) result` boundaries
  (`lib/pt2_spec_gen/pt2_spec_gen.ml`, a few `test/*.ml` decode/encode helpers)
  stay plain `match ... | Error e -> failwith e` — generalising the helper for call
  sites with no `Core.Error.t` wrapper would be speculative abstraction.
