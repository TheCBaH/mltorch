// Runs every exported op of the generated module over edge values and compares
// with a JavaScript reference. Floats compare with Object.is (NaN payloads are
// not distinguished by JS; -0 is).
const fs = require("fs");
const bytes = fs.readFileSync(process.argv[2]);
const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), {});
const ex = inst.exports;

const dv = new DataView(new ArrayBuffer(8));
const I = (n) => BigInt.asIntN(64, n);
const U = (n) => BigInt.asUintN(64, n);
const b = (x) => (x ? 1 : 0);
const f32bits = (x) => { dv.setFloat32(0, x); return dv.getInt32(0); };
const bitsf32 = (x) => { dv.setInt32(0, x); return dv.getFloat32(0); };
const f64bits = (x) => { dv.setFloat64(0, x); return dv.getBigInt64(0); };
const bitsf64 = (x) => { dv.setBigInt64(0, x); return dv.getFloat64(0); };
const nearest = (x) => {
  if (!Number.isFinite(x) || x === 0) return x;
  const f = Math.floor(x), d = x - f;
  let r = d < 0.5 ? f : d > 0.5 ? f + 1 : (f % 2 === 0 ? f : f + 1);
  return r === 0 && x < 0 ? -0 : r;
};
const satS = (x, lo, hi) => (Number.isNaN(x) ? 0 : Math.min(Math.max(Math.trunc(x), lo), hi) + 0);
const copysign = (a, b2) => {
  const neg = b2 < 0 || Object.is(b2, -0) || (Number.isNaN(b2) && (dv.setFloat64(0, b2), dv.getUint8(0) & 0x80));
  const m = Math.abs(a);
  return neg ? -m : m;
};
const ctz32 = (a) => (a === 0 ? 32 : 31 - Math.clz32(a & -a));
const pop32 = (a) => { let n = 0; for (let x = a >>> 0; x; x &= x - 1) n++; return n; };
const i64sat = (x, signed) => {
  if (Number.isNaN(x)) return 0n;
  const lo = signed ? -(2n ** 63n) : 0n, hi = signed ? 2n ** 63n - 1n : 2n ** 64n - 1n;
  if (x === Infinity) return signed ? I(hi) : I(hi);
  if (x === -Infinity) return I(lo);
  let t = BigInt(Math.trunc(x));
  if (t < lo) t = lo; if (t > hi) t = hi;
  return I(t);
};

const i32s = [0, 1, -1, 2, 7, -7, 31, 32, 33, 65535, 2147483647, -2147483648, 123456789, -123456789];
const i64s = [0n, 1n, -1n, 2n, 63n, 64n, -64n, 2n ** 31n, -(2n ** 31n), 2n ** 63n - 1n, -(2n ** 63n), 123456789012345678n, 2n ** 32n + 1n];
const f64s = [0, -0, 1, -1, 0.5, -0.5, 1.5, 2.5, -2.5, 3.5, 1e300, -1e300, Infinity, -Infinity, NaN,
  5e-324, 2 ** 31, -(2 ** 31) - 1, 2 ** 32, 2 ** 63, -(2 ** 63), 4294967295.5, 1e10, 2 ** 31 - 0.5];
const f32s = [...new Set(f64s.map(Math.fround))];

// name -> [operand kinds, reference, skip?]
const T = { i32: i32s, i64: i64s, f32: f32s, f64: f64s };
const ref = {
  "f32.demote_f64": [["f64"], (a) => Math.fround(a)],
  "f32.reinterpret_i32": [["i32"], (a) => bitsf32(a)],
  "f64.abs": [["f64"], Math.abs],
  "f64.add": [["f64", "f64"], (a, c) => a + c],
  "f64.ceil": [["f64"], Math.ceil],
  "f64.convert_i32_s": [["i32"], (a) => a],
  "f64.convert_i32_u": [["i32"], (a) => a >>> 0],
  "f64.convert_i64_s": [["i64"], (a) => Number(a)],
  "f64.convert_i64_u": [["i64"], (a) => Number(U(a))],
  "f64.copysign": [["f64", "f64"], copysign],
  "f64.div": [["f64", "f64"], (a, c) => a / c],
  "f64.eq": [["f64", "f64"], (a, c) => b(a === c)],
  "f64.floor": [["f64"], Math.floor],
  "f64.ge": [["f64", "f64"], (a, c) => b(a >= c)],
  "f64.gt": [["f64", "f64"], (a, c) => b(a > c)],
  "f64.le": [["f64", "f64"], (a, c) => b(a <= c)],
  "f64.lt": [["f64", "f64"], (a, c) => b(a < c)],
  "f64.max": [["f64", "f64"], Math.max],
  "f64.min": [["f64", "f64"], Math.min],
  "f64.mul": [["f64", "f64"], (a, c) => a * c],
  "f64.ne": [["f64", "f64"], (a, c) => b(a !== c)],
  "f64.nearest": [["f64"], nearest],
  "f64.neg": [["f64"], (a) => -a],
  "f64.promote_f32": [["f32"], (a) => a],
  "f64.reinterpret_i64": [["i64"], bitsf64],
  "f64.sqrt": [["f64"], Math.sqrt],
  "f64.sub": [["f64", "f64"], (a, c) => a - c],
  "f64.trunc": [["f64"], Math.trunc],
  "i32.add": [["i32", "i32"], (a, c) => (a + c) | 0],
  "i32.and": [["i32", "i32"], (a, c) => a & c],
  "i32.clz": [["i32"], Math.clz32],
  "i32.ctz": [["i32"], ctz32],
  "i32.div_s": [["i32", "i32"], (a, c) => (a / c) | 0, (a, c) => c === 0 || (a === -2147483648 && c === -1)],
  "i32.div_u": [["i32", "i32"], (a, c) => Math.trunc((a >>> 0) / (c >>> 0)) | 0, (a, c) => c === 0],
  "i32.eq": [["i32", "i32"], (a, c) => b(a === c)],
  "i32.eqz": [["i32"], (a) => b(a === 0)],
  "i32.extend16_s": [["i32"], (a) => (a << 16) >> 16],
  "i32.extend8_s": [["i32"], (a) => (a << 24) >> 24],
  "i32.ge_s": [["i32", "i32"], (a, c) => b(a >= c)],
  "i32.ge_u": [["i32", "i32"], (a, c) => b(a >>> 0 >= c >>> 0)],
  "i32.gt_s": [["i32", "i32"], (a, c) => b(a > c)],
  "i32.gt_u": [["i32", "i32"], (a, c) => b(a >>> 0 > c >>> 0)],
  "i32.le_s": [["i32", "i32"], (a, c) => b(a <= c)],
  "i32.le_u": [["i32", "i32"], (a, c) => b(a >>> 0 <= c >>> 0)],
  "i32.lt_s": [["i32", "i32"], (a, c) => b(a < c)],
  "i32.lt_u": [["i32", "i32"], (a, c) => b(a >>> 0 < c >>> 0)],
  "i32.mul": [["i32", "i32"], Math.imul],
  "i32.ne": [["i32", "i32"], (a, c) => b(a !== c)],
  "i32.or": [["i32", "i32"], (a, c) => a | c],
  "i32.popcnt": [["i32"], pop32],
  "i32.reinterpret_f32": [["f32"], f32bits],
  "i32.rem_s": [["i32", "i32"], (a, c) => (a % c) | 0, (a, c) => c === 0],
  "i32.rem_u": [["i32", "i32"], (a, c) => ((a >>> 0) % (c >>> 0)) | 0, (a, c) => c === 0],
  "i32.shl": [["i32", "i32"], (a, c) => a << (c & 31)],
  "i32.shr_s": [["i32", "i32"], (a, c) => a >> (c & 31)],
  "i32.shr_u": [["i32", "i32"], (a, c) => (a >>> (c & 31)) | 0],
  "i32.sub": [["i32", "i32"], (a, c) => (a - c) | 0],
  "i32.trunc_sat_f64_s": [["f64"], (a) => satS(a, -2147483648, 2147483647)],
  "i32.trunc_sat_f64_u": [["f64"], (a) => satS(a, 0, 4294967295) | 0],
  "i32.wrap_i64": [["i64"], (a) => Number(BigInt.asIntN(32, a))],
  "i32.xor": [["i32", "i32"], (a, c) => a ^ c],
  "i64.add": [["i64", "i64"], (a, c) => I(a + c)],
  "i64.and": [["i64", "i64"], (a, c) => a & c],
  "i64.div_s": [["i64", "i64"], (a, c) => I(a / c), (a, c) => c === 0n || (a === -(2n ** 63n) && c === -1n)],
  "i64.div_u": [["i64", "i64"], (a, c) => I(U(a) / U(c)), (a, c) => c === 0n],
  "i64.eq": [["i64", "i64"], (a, c) => b(a === c)],
  "i64.eqz": [["i64"], (a) => b(a === 0n)],
  "i64.extend_i32_s": [["i32"], (a) => BigInt(a)],
  "i64.extend_i32_u": [["i32"], (a) => BigInt(a >>> 0)],
  "i64.ge_s": [["i64", "i64"], (a, c) => b(a >= c)],
  "i64.ge_u": [["i64", "i64"], (a, c) => b(U(a) >= U(c))],
  "i64.gt_s": [["i64", "i64"], (a, c) => b(a > c)],
  "i64.gt_u": [["i64", "i64"], (a, c) => b(U(a) > U(c))],
  "i64.le_s": [["i64", "i64"], (a, c) => b(a <= c)],
  "i64.le_u": [["i64", "i64"], (a, c) => b(U(a) <= U(c))],
  "i64.lt_s": [["i64", "i64"], (a, c) => b(a < c)],
  "i64.lt_u": [["i64", "i64"], (a, c) => b(U(a) < U(c))],
  "i64.mul": [["i64", "i64"], (a, c) => I(a * c)],
  "i64.ne": [["i64", "i64"], (a, c) => b(a !== c)],
  "i64.or": [["i64", "i64"], (a, c) => a | c],
  "i64.reinterpret_f64": [["f64"], f64bits],
  "i64.rem_s": [["i64", "i64"], (a, c) => I(a % c), (a, c) => c === 0n],
  "i64.rem_u": [["i64", "i64"], (a, c) => I(U(a) % U(c)), (a, c) => c === 0n],
  "i64.shl": [["i64", "i64"], (a, c) => I(a << (c & 63n))],
  "i64.shr_s": [["i64", "i64"], (a, c) => a >> (c & 63n)],
  "i64.shr_u": [["i64", "i64"], (a, c) => I(U(a) >> (c & 63n))],
  "i64.sub": [["i64", "i64"], (a, c) => I(a - c)],
  "i64.trunc_sat_f64_s": [["f64"], (a) => i64sat(a, true)],
  "i64.trunc_sat_f64_u": [["f64"], (a) => i64sat(a, false)],
  "i64.xor": [["i64", "i64"], (a, c) => a ^ c],
};

let cases = 0, bad = 0;
const names = Object.keys(ex).filter((n) => typeof ex[n] === "function").sort();
const SIMD = /^(f64x2|f32x4|i32x4|i64x2|v128)\./;
const missing = Object.keys(ref).filter((n) => !(n in ex));
const unknown = names.filter((n) => !(n in ref) && !SIMD.test(n));
for (const n of [...missing.map((m) => "missing " + m), ...unknown.map((m) => "unreferenced " + m)]) { console.log(n); bad++; }
for (const [name, [kinds, f, skip]] of Object.entries(ref)) {
  if (!(name in ex)) continue;
  const grid = kinds.length === 1 ? T[kinds[0]].map((x) => [x]) : T[kinds[0]].flatMap((x) => T[kinds[1]].map((y) => [x, y]));
  for (const args of grid) {
    if (skip && skip(...args)) continue;
    const got = ex[name](...args), want = f(...args);
    cases++;
    if (!Object.is(got, want)) {
      if (bad++ < 20) console.log(`MISMATCH ${name}(${args.map(String)}) = ${String(got)}, want ${String(want)}`);
    }
  }
}

// ---- 128-bit operations: operands and results live in memory --------------
{
  const buf = ex.memory.buffer, dv = new DataView(buf);
  const A = 0, B = 16, C = 32, D = 48;
  const MASK = 0xFFFFFFFFFFFFFFFFn;
  const setF64 = (at, l) => { dv.setFloat64(at, l[0], true); dv.setFloat64(at + 8, l[1], true); };
  const getF64 = (at) => [dv.getFloat64(at, true), dv.getFloat64(at + 8, true)];
  const setU64 = (at, l) => { dv.setBigUint64(at, BigInt.asUintN(64, l[0]), true); dv.setBigUint64(at + 8, BigInt.asUintN(64, l[1]), true); };
  const getU64 = (at) => [dv.getBigUint64(at, true), dv.getBigUint64(at + 8, true)];
  const setI32 = (at, l) => l.forEach((x, k) => dv.setInt32(at + 4 * k, x, true));
  const getI32 = (at) => [0, 1, 2, 3].map((k) => dv.getInt32(at + 4 * k, true));
  const setF32 = (at, l) => l.forEach((x, k) => dv.setFloat32(at + 4 * k, x, true));
  const getF32 = (at) => [0, 1, 2, 3].map((k) => dv.getFloat32(at + 4 * k, true));
  const same = (x, y) => x.length === y.length && x.every((v, k) => Object.is(v, y[k]));
  const sat = (x, lo, hi) => (Number.isNaN(x) ? 0 : Math.min(Math.max(Math.trunc(x), lo), hi) + 0);
  const pairs = f64s.flatMap((a) => f64s.map((c) => [a, c]));
  const check = (name, args, got, want) => {
    cases++;
    const ok = got.length === want.length && got.every((v, k) => Object.is(v, want[k]));
    if (!ok && bad++ < 20) console.log(`MISMATCH ${name}(${args.map(String)}) = ${got.map(String)}, want ${want.map(String)}`);
  };
  const lane2 = (f) => (x) => [f(x[0]), f(x[1])];
  const fbin = { add: (a, c) => a + c, sub: (a, c) => a - c, mul: (a, c) => a * c, div: (a, c) => a / c, min: Math.min, max: Math.max };
  for (const [op, f] of Object.entries(fbin))
    for (const [x, y] of pairs.map((p, k) => [p, pairs[(k * 7 + 3) % pairs.length]])) {
      setF64(A, x); setF64(B, y); ex["f64x2." + op](A, B, D);
      check("f64x2." + op, [x, y], getF64(D), [f(x[0], y[0]), f(x[1], y[1])]);
    }
  const fun = { abs: Math.abs, neg: (a) => -a, sqrt: Math.sqrt, ceil: Math.ceil, floor: Math.floor, trunc: Math.trunc, nearest };
  for (const [op, f] of Object.entries(fun))
    for (const x of pairs) { setF64(A, x); ex["f64x2." + op](A, D); check("f64x2." + op, [x], getF64(D), lane2(f)(x)); }
  const fcmp = { eq: (a, c) => a === c, ne: (a, c) => a !== c, lt: (a, c) => a < c, gt: (a, c) => a > c, le: (a, c) => a <= c, ge: (a, c) => a >= c };
  for (const [op, f] of Object.entries(fcmp))
    for (const [x, y] of pairs.map((p, k) => [p, pairs[(k * 5 + 1) % pairs.length]])) {
      setF64(A, x); setF64(B, y); ex["f64x2." + op](A, B, D);
      check("f64x2." + op, [x, y], getU64(D), [f(x[0], y[0]) ? MASK : 0n, f(x[1], y[1]) ? MASK : 0n]);
    }
  for (const x of pairs) {
    setF64(A, x); ex["f32x4.demote_f64x2_zero"](A, D);
    check("f32x4.demote_f64x2_zero", [x], getF32(D), [Math.fround(x[0]), Math.fround(x[1]), 0, 0]);
    ex["i32x4.trunc_sat_f64x2_s_zero"](A, D);
    check("i32x4.trunc_sat_f64x2_s_zero", [x], getI32(D), [sat(x[0], -2147483648, 2147483647), sat(x[1], -2147483648, 2147483647), 0, 0]);
    ex["i32x4.trunc_sat_f64x2_u_zero"](A, D);
    check("i32x4.trunc_sat_f64x2_u_zero", [x], getI32(D), [sat(x[0], 0, 4294967295) | 0, sat(x[1], 0, 4294967295) | 0, 0, 0]);
  }
  for (let k = 0; k + 3 < f32s.length; k++) {
    const q = [f32s[k], f32s[k + 1], f32s[k + 2], f32s[k + 3]];
    setF32(A, q); ex["f64x2.promote_low_f32x4"](A, D);
    check("f64x2.promote_low_f32x4", [q], getF64(D), [q[0], q[1]]);
  }
  for (let k = 0; k + 3 < i32s.length; k++) {
    const q = [i32s[k], i32s[k + 1], i32s[k + 2], i32s[k + 3]];
    setI32(A, q);
    ex["f64x2.convert_low_i32x4_s"](A, D); check("f64x2.convert_low_i32x4_s", [q], getF64(D), [q[0], q[1]]);
    ex["f64x2.convert_low_i32x4_u"](A, D); check("f64x2.convert_low_i32x4_u", [q], getF64(D), [q[0] >>> 0, q[1] >>> 0]);
  }
  for (const x of f64s) { ex["f64x2.splat"](x, D); check("f64x2.splat", [x], getF64(D), [x, x]); }
  for (const x of f32s) { ex["f32x4.splat"](x, D); check("f32x4.splat", [x], getF32(D), [x, x, x, x]); }
  for (const x of i32s) { ex["i32x4.splat"](x, D); check("i32x4.splat", [x], getI32(D), [x, x, x, x]); }
  for (const x of i64s) { ex["i64x2.splat"](x, D); check("i64x2.splat", [x], getU64(D), [BigInt.asUintN(64, x), BigInt.asUintN(64, x)]); }
  const words = i64s.map((x) => BigInt.asUintN(64, x));
  const vecs = words.flatMap((a) => words.map((c) => [a, c]));
  const vb = { and: (a, c) => a & c, andnot: (a, c) => a & ~c & MASK, or: (a, c) => a | c, xor: (a, c) => a ^ c };
  for (const [op, f] of Object.entries(vb))
    for (const [x, y] of vecs.map((p, k) => [p, vecs[(k * 11 + 2) % vecs.length]])) {
      setU64(A, x); setU64(B, y); ex["v128." + op](A, B, D);
      check("v128." + op, [x, y], getU64(D), [f(x[0], y[0]), f(x[1], y[1])]);
    }
  for (const x of vecs) {
    setU64(A, x); ex["v128.not"](A, D); check("v128.not", [x], getU64(D), [~x[0] & MASK, ~x[1] & MASK]);
    check("v128.any_true", [x], [ex["v128.any_true"](A)], [b(x[0] !== 0n || x[1] !== 0n)]);
  }
  for (let k = 0; k < vecs.length; k++) {
    const x = vecs[k], y = vecs[(k * 3 + 1) % vecs.length], m = vecs[(k * 5 + 2) % vecs.length];
    setU64(A, x); setU64(B, y); setU64(C, m); ex["v128.bitselect"](A, B, C, D);
    check("v128.bitselect", [x, y, m], getU64(D), [(x[0] & m[0]) | (y[0] & ~m[0] & MASK), (x[1] & m[1]) | (y[1] & ~m[1] & MASK)]);
  }
  const ibin = { add: (a, c) => a + c, sub: (a, c) => a - c, mul: (a, c) => a * c };
  const sgn = (x) => BigInt.asIntN(64, x);
  for (const [op, f] of Object.entries(ibin))
    for (const [x, y] of vecs.map((p, k) => [p, vecs[(k * 7 + 4) % vecs.length]])) {
      setU64(A, x); setU64(B, y); ex["i64x2." + op](A, B, D);
      check("i64x2." + op, [x, y], getU64(D), [BigInt.asUintN(64, f(x[0], y[0])), BigInt.asUintN(64, f(x[1], y[1]))]);
    }
  const icmp = { eq: (a, c) => a === c, ne: (a, c) => a !== c, lt_s: (a, c) => sgn(a) < sgn(c), gt_s: (a, c) => sgn(a) > sgn(c), le_s: (a, c) => sgn(a) <= sgn(c), ge_s: (a, c) => sgn(a) >= sgn(c) };
  for (const [op, f] of Object.entries(icmp))
    for (const [x, y] of vecs.map((p, k) => [p, vecs[(k * 9 + 5) % vecs.length]])) {
      setU64(A, x); setU64(B, y); ex["i64x2." + op](A, B, D);
      check("i64x2." + op, [x, y], getU64(D), [f(x[0], y[0]) ? MASK : 0n, f(x[1], y[1]) ? MASK : 0n]);
    }
  const sh = { shl: (a, n) => a << n, shr_s: (a, n) => sgn(a) >> n, shr_u: (a, n) => a >> n };
  for (const [op, f] of Object.entries(sh))
    for (const x of vecs) for (const n of [0, 1, 7, 31, 32, 63, 64, 65, 200]) {
      setU64(A, x); ex["i64x2." + op](A, n, D);
      const k = BigInt(n & 63);
      check("i64x2." + op, [x, n], getU64(D), [BigInt.asUintN(64, f(x[0], k)), BigInt.asUintN(64, f(x[1], k))]);
    }
}
console.log(`${names.length} ops, ${bad} failures`);
process.exit(bad ? 1 : 0);
