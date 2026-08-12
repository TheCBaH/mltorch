# ATen Op Schema

## Finding a schema at runtime

```python
import torch
schema = torch.ops.aten.conv2d.default._schema
# → aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None,
#                SymInt[2] stride=[1,1], SymInt[2] padding=[0,0],
#                SymInt[2] dilation=[1,1], SymInt groups=1) -> Tensor
```

## Overload naming: `<namespace>.<opname>.<overload>`

The `target` field in a `Node` is always the fully-qualified overload name:

```
torch.ops.aten.conv2d.default
│          │     │      └── overload name
│          │     └── op name
│          └── namespace (aten = core ATen library)
└── always "torch.ops"
```

`.default` is the unnamed (primary) overload — the one without an `out=` tensor
parameter. Other overloads disambiguate argument types:

```
aten::add.Tensor   (Tensor + Tensor)
aten::add.Scalar   (Tensor + Scalar)
aten::add.int      (int + int)
aten::add.str      (str + str)
aten::add.default  (Scalar + Scalar)
```

List all overloads:
```python
torch.ops.aten.add.overloads()
```

## Authoritative source

- Runtime: `torch.ops.aten.<name>.<overload>._schema`
- Source: `aten/src/ATen/native/native_functions.yaml` in the PyTorch repo

## ATen type → JSON Argument variant mapping

| ATen type | JSON representation |
|---|---|
| `Tensor` | `as_tensor: {name: "..."}` |
| `Tensor?` | `as_tensor: {name}` or `as_none: true` |
| `Tensor[]` | `as_tensors: [{name}, ...]` |
| `Tensor?[]` | `as_optional_tensors: [...]` |
| `int` | `as_int: N` |
| `int[]` | `as_ints: [...]` |
| `float` | `as_float: N` |
| `bool` | `as_bool: true/false` |
| `str` | `as_string: "..."` |
| `Scalar` | `as_int`, `as_float`, `as_bool`, or `as_complex` (runtime type wins) |
| `ScalarType` | `as_scalar_type: N` (enum int) |
| `MemoryFormat` | `as_memory_format: N` (enum int) |
| `Layout` | `as_layout: N` (enum int) |
| `Device` | `as_device: {type, index}` |
| `SymInt` | `as_int` (concrete) or `as_sym_int: {as_int\|as_name}` (symbolic) |
| `SymInt[]` | `as_ints` (all concrete) or `as_sym_ints: [...]` (any symbolic) |
| `SymFloat` | `as_float` or `as_sym_float: {as_float\|as_name}` |
| `SymBool` | `as_bool` or `as_sym_bool: {as_bool\|as_name}` |

## Sym* concrete vs symbolic

`SymInt` / `SymFloat` / `SymBool` in an ATen schema means the argument *can* be a
symbolic expression (only happens with `torch.export.export(..., dynamic_shapes=...)`).
The exporter picks the flattest JSON representation:

- All inputs have known constant values at export time → `as_int` / `as_ints`
- Any input involves a dynamic dimension → `as_sym_int` / `as_sym_ints`
  - Element is a concrete integer in a mixed list → `{as_int: N}`
  - Element is a symbolic variable → `{as_name: "sym_size_int_1"}`

Plain `bool` / `int` / `float` (without `Sym` prefix) are never symbolic.

## Schema text is not comment-safe

The generators echo each op's schema signature into a doc comment above its
binding. `unbind.int`'s is

```text
unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor[](a)
```

and `Tensor(a -> *)` — the "may alias anything" annotation — contains a literal
`*)`, which **closes the OCaml comment early** and makes the whole generated
module a syntax error. Nothing catches this before the build; it surfaces as an
unmatched-`struct` error hundreds of lines away.

`Aten_emit.ocaml_comment_body` handles it: OCaml's comment lexer parses string
literals inside comments, so a signature containing `*)` or `(*` is emitted
quoted (`(* "…Tensor(a -> *)…" *)`), which hides the sequence without altering a
character of the text. Only signatures that need it are quoted, so every other
op's comment stays byte-identical.

The C emitters need no equivalent — `*)` is not `*/`.
