# func_ast.ml module refactor

## Goal
Wrap every type in `func_ast.ml` in a named module so qualified names
are used throughout (`Base.t`, `Type.t`, `Annotation.t`, …).  Enum
modules (`Base`, `Default`) get `to_string` + `pp`; structural modules
get `pp` only.

## Type → module mapping

| Old name        | New module  | Category |
|-----------------|-------------|----------|
| `base_ty`       | `Base`      | enum → `to_string` + `pp` |
| `ty`            | `Type`      | recursive variant → `to_string` + `pp` |
| `annotation`    | `Annotation`| record → `pp` |
| `default_val`   | `Default`   | enum → `to_string` + `pp` |
| `argument`      | `Argument`  | record → `pp` |
| `return_val`    | `Return`    | record → `pp` |
| `op_name`       | `Op_name`   | record → `pp` |
| `arguments`     | `Arguments` | record → `pp` |
| `func_schema`   | top-level `t` + `pp` | |

## Constructor renames (Default module)

`Default.None` — note: not OCaml's `option None`; always accessed qualified
so no shadowing in practice.

Old → New:
- `DefaultNone`      → `Default.None`
- `DefaultBool b`    → `Default.Bool b`
- `DefaultInt n`     → `Default.Int n`
- `DefaultFloat s`   → `Default.Float s`
- `DefaultStr s`     → `Default.Str s`
- `DefaultIntList ns`→ `Default.IntList ns`
- `DefaultIdent s`   → `Default.Ident s`

## Files changed

1. `lib/aten_schema/func_ast.ml` — full rewrite into modules
2. `lib/aten_schema/func_parser.mly` — semantic actions + `%start` type
3. `lib/aten_schema/func_schema.ml` — return type `t` instead of `func_schema`
4. `test/func_parser_test.ml` — drop hand-rolled `show_*`, use `pp` functions

## pp design

All `pp` use `Format.formatter` convention.  Composite `pp` functions
call sub-`pp` functions directly; `to_string` is `Format.asprintf "%a" pp`.
`Arguments.pp` uses a `sep` ref to insert `, ` between items and inserts
`*` before the kwarg/out section only when that section is non-empty.
