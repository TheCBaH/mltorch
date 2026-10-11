# Inference comparison coordinator: schema layer

`lib/inference_compare_schema` is the pure data layer of the inference
comparison coordinator: nominal identifiers and digests, graph and mapping
descriptions, recipes, comparison policies, selectors, contracts, results and
evidence records, each with a strict schema-v1 wire codec. It depends on
`err_trace`, `fmt` and `jsont` only, so no engine, compiler, Unix or libtorch
dependency can reach it; `dune build lib/inference_compare_schema` proves the
boundary.

## Rules that outlive the plan

- **Every semantic role is its own abstract type.** `Text_value.Make_identifier`
  and `Make_prose` are generative, so `Id.Unit.t`, `Id.Case.t`,
  `Digest.Graph.t`, `Digest.Payload.t`, `Prose.Description.t` and the two
  port types never unify although all of them serialize as strings.
  `test/inference_compare/type_boundaries.t` pairs each rejected substitution
  with a control that compiles. Do not add aliases that equate two roles.
- **Record owners are modules named after the record**, with `decode`/`encode`
  beside the type. Sub-records that would reuse a label (`Producer`,
  `Consumer`, `Hop` in `Tensor_selector`) are submodules rather than sibling
  types, so no label is shadowed and warning 30 stays on.
- **The wire is strict.** Decoding rejects duplicate keys, unknown members,
  unknown tags, malformed decimals and a repeated identifier inside a list
  (`Wire.unique`). 64-bit counts are decimal strings; an absent optional field
  is an explicit `null`; no field has a default. A measurement that can be
  nonfinite is a tagged record, never a JSON number, and `Wire.enc_float`
  refuses a nonfinite value.
- **Digests carry their algorithm** (`algorithm:hex`); equality includes it,
  and the algorithm is never inferred from length.
- **A digest preimage is canonical text under a domain tag**
  (`Digest_preimage`): object members sorted, set-like tables sorted by
  identifier, and ordered lists (operands, results, final outputs, cluster
  inputs and outputs, recipe arguments) kept in declared order because that
  order is the occurrence order. Rule order carries no precedence; priority is
  part of the preimage.
- **Integers are bounded for every backend.** `Wire.int` accepts at most
  2^30 - 1 in magnitude; positions and axes are further bounded
  (`Port.max_index`); extents and counts are `int64`, and
  `Tensor_spec.element_count` divides into the ceiling before multiplying.

## Not here

Hashing (the payload digest backend), graph validation, planning, execution and
evidence writing belong to the layers above. The schema only decodes, encodes
and canonicalizes.
