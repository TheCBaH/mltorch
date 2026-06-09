# raw.ml typed enum/union refactor

## Fields promoted to proper types

| Field | Old type | New type | Values |
|-------|----------|----------|--------|
| `variants` | `string option` | `Variant.t list` | `Function`, `Method` (comma-sep string in YAML) |
| `tags` | `string list` | `Tag.t list` | 16 values from `tags.yaml` |
| `device_check` | `string option` | `Device_check.t option` | `NoCheck`, `ExactSame` (from `torchgen/model.py`) |
| `category_override` | `string option` | `Category.t option` | `Dummy`, `Factory` |
| `dispatch_entry.keys` | `string` | `backends : Backend.t list` | 27 backends (comma-sep) |
| `ufunc_inner_loop` | `(string * string) list` | `(Backend.t * string) list` | uses same Backend enum |

Fields left as strings: `structured_delegate`, `structured_inherits`,
`python_module`, `cpp_no_default_args`, `autogen`, `precomputed`
(kernel refs / argument names / op name lists — not enums).

## Module structure

Each enum is a sub-module with `t`, `of_string : string -> t option`,
`to_string : t -> string`, `pp : Format.formatter -> t -> unit`.

`Backend` covers all 27 dispatch key names including `Generic` and
`ScalarOnly` (used by `ufunc_inner_loop`).

## Jsont decoding helper

`string_enum ~kind of_string` wraps `Jsont.Base.string`/`Jsont.Base.map`
to give proper `Meta.t`-aware error messages on unknown enum values.
`tags_jsont` uses `Jsont.any` (scalar-or-array) and wraps single tags in a list.
Dispatch and ufunc keys are parsed inside `fold_object` where `Meta.t` is available.

## Backward compatibility

`aten_schema_raw_test.ml` updated: `String.concat "," e.Raw.tags` →
`String.concat "," (List.map Tag.to_string e.Raw.tags)`.
Cram output is identical.
