(* The named helpers generated code calls, as JavaScript source. Each is a
   transcription of its OCaml counterpart, tested directly against it, so the
   emitter references a helper by name instead of open-coding a semantic that
   two backends could then disagree on. Kept as data, ordered, so the prelude
   the emitter prepends is deterministic. *)

(* [Expr_bridge.bound_in_range]: the FIRST axis, in N T D H W C order, whose
   component lies outside the buffer's shape. The record's kind and fields are
   the ones the interpreter's [`Coord_out_of_range] row carries. *)
let coord_failure =
  {js|function coord_failure(buffer, extents, coord) {
  for (let axis = 0; axis < 6; axis++) {
    if (coord[axis] < 0 || coord[axis] >= extents[axis]) {
      return { kind: "coord_out_of_range", buffer: buffer, axis: axis, index: coord[axis], coord: coord };
    }
  }
  return { kind: "defect" };
}|js}

(* [Max_op.apply Float_max] is [Float.max]: NaN propagates and [+0.] is above
   [-0.]. [Math.max] has exactly that contract, so the helper is a name for it
   rather than a second definition of the rule. *)
let float_max = {js|function float_max(a, b) {
  return Math.max(a, b);
}|js}

(* [Max_op.pool_better]: the candidate wins on strict greater-than OR on NaN, so
   an ordinary tie keeps the incumbent and every NaN re-triggers (the last wins). *)
let pool_better =
  {js|function pool_better(best, value) {
  return value > best || value !== value;
}|js}

(* [Value.erf_approx], operation for operation: the Abramowitz-Stegun
   polynomial, not libm's erf. Only its inner [exp] is a transcendental, and so
   the only place it may differ from OCaml. No FMA is formed by JavaScript, while
   [ocamlopt] may contract a multiply-add here: a native-vs-JS difference
   confined to this function is triaged against the bytecode build first. *)
let erf =
  {js|function erf(x) {
  const p = 0.3275911;
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + p * ax);
  const poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
  return sign * (1 - poly * Math.exp(-ax * ax));
}|js}

(* [Half.Bf16.to_float]: bfloat16 is the high half of a binary32 pattern. The
   shared scratch pair reinterprets the 32 bits without allocating; [bits << 16]
   is negative when the sign bit is set, and a [Uint32Array] store wraps it back
   to the pattern. *)
let bf16_to_float =
  {js|const bf16_bits = new Uint32Array(1);
const bf16_view = new Float32Array(bf16_bits.buffer);
function bf16_to_float(bits) {
  bf16_bits[0] = bits << 16;
  return bf16_view[0];
}|js}

(* [Half.Half.to_float]: IEEE binary16, transcribed branch for branch. [ldexp] is
   an exact power-of-two scaling, so it is a multiplication by an exactly
   representable power of two. *)
let f16_to_float =
  {js|function f16_to_float(h) {
  const sign = (h >> 15) & 1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;
  let m;
  if (exp === 0) m = mant * Math.pow(2, -24);
  else if (exp === 0x1f) m = mant === 0 ? Infinity : NaN;
  else m = (mant | 0x400) * Math.pow(2, exp - 25);
  return sign === 1 ? -m : m;
}|js}

(* The failure [Value.i64_of_float] reports for a value outside the int64 range:
   NaN, an infinity, or a finite value beyond [-2^63, 2^63). *)
let i64_from_float_failure =
  {js|function i64_from_float_failure(x) {
  if (Number.isNaN(x)) return { kind: "i64_from_float_nan" };
  if (!Number.isFinite(x)) return { kind: "i64_from_float_infinite" };
  return { kind: "i64_from_float_out_of_range", value: x };
}|js}

let helpers : (string * string) list =
  [
    ("bf16_to_float", bf16_to_float);
    ("coord_failure", coord_failure);
    ("erf", erf);
    ("f16_to_float", f16_to_float);
    ("float_max", float_max);
    ("i64_from_float_failure", i64_from_float_failure);
    ("pool_better", pool_better);
  ]

let source = String.concat "\n" (List.map snd helpers)
