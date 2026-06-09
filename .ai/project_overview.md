# Project Overview

## Goal

Parse PyTorch `.pt2` exported model archives in OCaml. The immediate deliverable is a
typed JSON parser for `model.json` (the serialized `ExportedProgram` graph). The parser
is generated automatically from PyTorch's own `schema.yaml`, so it stays in sync with
the upstream schema without manual maintenance.

## Component Map

```
modules/pytorch/torch/_export/serde/schema.yaml   ← upstream schema (git submodule)
                │
                ▼
         bin/schema_gen.ml                         ← CLI: reads schema.yaml, prints OCaml
                │ (dune rule: test/dune)
                ▼
         test/schema_pytorch.ml                    ← generated file, in build tree only
                │
                ▼ (#use in ocaml toplevel)
         test/*_cram.t                             ← cram tests decode real model.json
```

### Libraries

| Path | Dune name | Role |
|---|---|---|
| `lib/pytorch_schema/` | `pytorch_schema` | Schema meta-parser + code generator |
| `lib/schema_runtime/` | `schema_runtime` | Thin runtime: just `String_map = Map.Make(String)` |
| `bin/schema_gen.ml` | `schema_gen` (exe) | CLI wrapper: reads YAML, calls generator |

### Key sub-modules in `pytorch_schema`

| Module | Role |
|---|---|
| `Pytorch_schema` (pytorch_schema.ml) | Parse `schema.yaml` into `Type_def.t String_map.t` |
| `Type_expr` | AST for field type expressions (`Optional[List[int]]` etc.) |
| `Type_expr_lexer` / `Type_expr_parser` | ocamllex + Menhir parser for type strings |
| `Type_expr_parse` | Thin wrapper: `of_string s` → `Type_expr.t` |
| `Scc` | Generic Tarjan SCC functor |
| `Schema_codegen` | Type map → OCaml source string |

## Data Flow

1. `schema_gen schema.yaml` → `schema_pytorch.ml`
2. `schema_pytorch.ml` opens `Schema_runtime` (gets `String_map`), then defines one
   OCaml module per schema type, each with a `.jsont` decoder and an OCaml `type t`.
3. Cram tests `#use "schema_pytorch.ml"` in the OCaml toplevel after loading
   `schema_runtime.cma`, then call `Jsont_bytesrw.decode_string ExportedProgram.jsont`.

## External Dependencies

| Package | Version | Role |
|---|---|---|
| `jsont` | opam | Zero-copy JSON codec; the generated decoders use this API |
| `jsont.bytesrw` | (part of jsont) | Decodes from a string (`decode_string`) |
| `yamlt` | vendored | YAML→jsont bridge; parses schema.yaml via jsont's value type |
| `menhir` | opam | Parser generator for type expressions |

`yamlt` is vendored under `vendored/ocaml-yamlt/` because it is not yet on opam.

## What Exists vs What Is Ahead

Done: schema parsing → code generation → single-file OCaml → parse any model.json.

Not yet: a proper library API, a viewer UI, TFLite support (the `.tflite` files in
`models/` are from an earlier phase and are currently unused).

## Side track: ATen "core" C++ build

Separate from the schema/JSON pipeline, [`modules/aten_core/`](../modules/aten_core/)
builds a minimal slice of PyTorch's C++ (the dispatcher / IValue / type-system layer,
122 sources) into `libaten_core.a` via dune — no OCaml bindings yet. See
[aten_core_build.md](aten_core_build.md).
