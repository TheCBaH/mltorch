# Float32 Shortest JSON Serialization

Current `Aten_spec.Float32` serialization may emit finite f32 values as JSON
numbers only after proving the `Jsont` default `%.17g` spelling round-trips back
to the same f32 bits. This preserves correctness but can write noisy decimals
such as `9.9999997473787516e-06` for a logical `1e-5` f32 value.

Future improvement: replace the `%.17g`-based numeric spelling with shortest
round-tripping decimal serialization for finite f32 values.

Desired behavior:

- Keep raw 32-bit hex strings for NaN, infinities, and negative zero.
- Emit ordinary finite f32 values as the shortest decimal that parses back to
  the same f32 bit pattern.
- Preserve decode compatibility with both JSON numbers and raw hex strings.
- Add tests for common awkward values such as f32 `1e-5`, f32 `1e-8`, `0.1`,
  `Float.pi`, subnormals, and values around powers of ten.
- Add a randomized or exhaustive bit-pattern check:
  `bits -> shortest_decimal -> decode -> bits`.

Implementation options:

- Use a proven shortest-float algorithm such as Dragonbox or Ryu, if adding a
  small C++ binding is acceptable.
- For f32 only, an OCaml implementation can be simpler: search decimal
  candidates up to the f32-safe digit budget, parse each candidate, round through
  `Float32.to_f32`, and choose the shortest spelling whose bits match.

Important integration detail: `Jsont.Number` defers decimal formatting to
`Jsont_bytesrw`, so per-value shortest f32 spellings cannot be controlled by the
current `Jsont.Number` value alone. The future implementation likely needs a
custom JSON rendering path for f32 payloads, or a broader encoder hook that can
format these numbers with an f32-aware shortest formatter.
