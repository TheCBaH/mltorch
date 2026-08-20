# Code Generator Design

`lib/pytorch_schema/schema_codegen.ml` — `Type_def.t String_map.t → string`

## Output Shape

The generated file (`schema_pytorch.ml`) opens `Schema_runtime` (for `String_map`) and
then emits one module per schema type, in topological order (dependencies first). The
entry point is `generate : Type_def.t String_map.t -> string`.

```ocaml
open Schema_runtime          (* gives String_map *)

module SchemaVersion = struct
  type t = { major : int; minor : int }
  let make major minor = { major; minor }
  let jsont = Jsont.Object.map ~kind:"SchemaVersion" make
    |> Jsont.Object.mem "major" Jsont.int
    |> Jsont.Object.mem "minor" Jsont.int
    |> Jsont.Object.finish
end

module ArgumentKind = struct
  type t = UNKNOWN | POSITIONAL | KEYWORD
  let jsont = Jsont.map ~kind:"ArgumentKind"
    ~dec:(fun n -> match n with 0 -> UNKNOWN | 1 -> POSITIONAL | 2 -> KEYWORD
                 | n -> Jsont.Error.msgf ...)
    ~enc:(fun v -> match v with UNKNOWN -> 0 | ...)
    Jsont.int
  let make v = v
end
...
```

## Three Schema Type Kinds

### Struct → OCaml record + `Jsont.Object.map`

Fields are classified as:
- `Mem_required`: non-optional, no default → `Jsont.Object.mem`
- `Mem_optional`: `Optional[T]` in schema → `Jsont.Object.opt_mem "f" (Jsont.option T_jsont)`
  with `Option.join` in the constructor to flatten `'a option option → 'a option`
- `Mem_default expr`: non-optional field with a Python default value →
  `Jsont.Object.opt_mem "f" T_jsont` with `(match param with None -> expr | Some v -> v)`

### Union → OCaml variant + `Jsont.json` dispatch

PyTorch unions serialize as `{"key": value}` (single-key objects). The decoder matches
on `Jsont.Object([((key,_), value)], _)` and dispatches on the key string. OCaml
constructors are derived from field names: `as_tensor` → `Tensor`, `as_name` → `Name`.

### Enum → OCaml variant + `Jsont.int` decode/encode

Enums serialize as their integer value. The decoder is a match on the integer.
Constructor names strip leading underscores and capitalize: `_UNKNOWN` → `Unknown`.

## SCC-Based Recursive Type Handling

`compute_sccs` runs Tarjan's algorithm (the generic `Scc.Make(String)` functor) on the
type dependency graph. SCCs determine which modules need `module rec ... and ...`.

For a recursive group, the generator emits three passes:

```ocaml
(* Pass 1: forward-declare types via module rec *)
module rec Node_Type : sig
  type t = { target : string; ... }
  val make : ...
end = struct ... end

and Argument_Type : sig ... end = struct ... end

(* Pass 2: lazy jsont values (break the cycle) *)
let rec node_jsont : Node_Type.t Jsont.t Lazy.t = lazy (
  Jsont.Object.map ~kind:"Node" (fun target ... -> ({ ... } : Node_Type.t))
  |> Jsont.Object.mem "target" Jsont.string
  ...
  |> Jsont.Object.finish)

and argument_jsont : Argument_Type.t Jsont.t Lazy.t = lazy (
  Jsont.map ~kind:"Argument" ~dec:(fun json -> match json with
    | Jsont.Object ([((key,_), value)], _) -> (match key with
      | "as_tensor" -> (match Jsont.Json.decode (Lazy.force node_jsont) value with
                        | Ok v -> Argument_Type.Tensor v | Error s -> ...)
      | ...))
  ...)

(* Pass 3: public facades with Jsont.rec' *)
module Node = struct
  include Node_Type
  let jsont = Jsont.rec' node_jsont
end

module Argument = struct
  include Argument_Type
  let jsont = Jsont.rec' argument_jsont
end
```

`Jsont.rec'` unwraps the `Lazy.t` at first use, breaking the initialization cycle.

## Key Design Decisions

### Why generate to a single `.ml` file, not a library?

The generated code is used via `#use` in the OCaml toplevel (for cram tests). A single
file is simpler to load than a compiled library. If later refactored into a proper
library the generator output is already valid OCaml.

### Why `module rec` + `let rec lazy` instead of `let rec` directly?

OCaml `let rec` requires the RHS to be a syntactic value (a lambda or a constructor
application). `Jsont.Object.map ... |> ...` is an application chain, not a syntactic
value, so it cannot appear directly in `let rec`. Wrapping in `lazy` (a value) makes
it legal. `Jsont.rec'` then forces the lazy at first decode.

### Why `Option.join` for `Mem_optional`?

`Jsont.Object.opt_mem "f" (Jsont.option T)` delivers `T option option` to the
constructor: `None` when the key is absent, `Some None` when the key is present with
JSON `null`, `Some (Some v)` for a real value. `Option.join` collapses both `None` and
`Some None` to `None`, giving `T option` in the record.

Using bare `opt_mem "f" T_jsont` rejects explicit JSON `null` with a parse error,
which breaks real `model.json` files that write `null` on optional fields.

### Why `match param with None -> expr | Some v -> v` for `Mem_default`?

`Option.value ~default:expr` evaluates `expr` unconditionally before calling
`Option.value`. When `expr` is `(failwith "TODO default: ...")` this crashes even when
the key is present. The `match` is the correct lazy pattern. Discovered when `Graph`'s
`is_single_tensor_return` field (default `False`) raised `Failure` at parse time.

### Naming conventions

- JSON field `as_foo_bar` in a union → OCaml constructor `Foo_bar` (strip `as_`, capitalize)
- JSON field `_UNKNOWN` in an enum → OCaml constructor `Unknown` (strip leading `_`, capitalize)
- OCaml keyword clash (e.g. field named `type`) → append `_` (`type_`)
- Recursive type support module: `Foo_Type` (not `Foo`) → avoids shadowing the facade

**The ATen generators share one escape, and that is not tidiness.** `Aten_ident.ml_id`
(`lib/aten_gen/aten_ident.ml`) is used by `Aten_spec_gen` and `Aten_decode_gen` alike.
Before it was lifted out, only the spec generator escaped; the decode generator bound
`a.name` raw, which is fine until an op has an argument literally named `end` —
`slice.Tensor` does. The failure mode is what makes sharing the rule load-bearing: the
generated `let* end = …` does not *parse*, so the break is loud, but it only surfaces
after the missing decode arm is added, i.e. a run later than the change that caused it.
When adding an arm to either generator, escape through `Aten_ident.ml_id` and keep the
JSON key at the wire name — they are different strings whenever the escape fires.

## Default Value YAML Quoting Trap

`default: 'False'` in YAML strips the single quotes — OCaml sees `"False"`, not
`"'False'"`. The `default_ocaml_expr` function therefore accepts both the quoted and
unquoted forms:

```ocaml
| ("False" | "'False'"), Type_expr.Bool -> Some "false"
| ("True"  | "'True'" ), Type_expr.Bool -> Some "true"
| ("[]"    | "'[]'"   ), Type_expr.List _ -> Some "[]"
| ("{}"    | "'{}'"   ), Type_expr.Dict _ -> Some "String_map.empty"
```

## Format Conventions

The generator uses `Format.fprintf` with box combinators throughout.

- `@[<v 0>..@]`: vertical box at column 0 (top-level structure)
- `@;<0 2>`: break with 2-space indent relative to enclosing box
- `@\n@;`: hard newline then a soft break (blank line between declarations)
- `%a`: apply a `pp` function (passed as argument)

All output funnels through `Format.formatter_of_buffer`, flushed by `@.` at the end.
