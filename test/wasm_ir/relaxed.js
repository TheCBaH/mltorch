// The relaxed operations, run under a node that enables them (the module is not
// valid on a default node). A relaxed madd lane is a*b + c either fused (one
// rounding) or not (the product rounded, then the sum): each lane must be one of
// the two, and the two must agree where they are exactly equal, so a wrong opcode
// or operand order cannot hide.
const fs = require("fs");
const bytes = fs.readFileSync(process.argv[2]);
if (!WebAssembly.validate(bytes)) { console.log("the module does not validate: relaxed-simd is off"); process.exit(1); }
const ex = new WebAssembly.Instance(new WebAssembly.Module(bytes), {}).exports;
const dv = new DataView(ex.memory.buffer);
const f = Math.fround;
const bitsOf = (x) => { const t = new DataView(new ArrayBuffer(8)); t.setFloat64(0, x); return t.getBigInt64(0); };
const ofBits = (b) => { const t = new DataView(new ArrayBuffer(8)); t.setBigInt64(0, b); return t.getFloat64(0); };
// binary32 fmaf through an exact product, TwoSum and round to odd
function fma32(a, b, c) {
  if (![a, b, c].every(Number.isFinite)) return f(a * b + c);
  const p = a * b, s = p + c, bb = s - p, e = (p - (s - bb)) + (c - bb);
  if (e === 0) return f(s);
  let bits = bitsOf(s);
  if ((bits & 1n) === 0n) bits += ((e > 0) === (s > 0)) ? 1n : -1n;
  return f(ofBits(bits));
}
const unfused = (a, b, c) => f(f(a * b) + c);
const A = 0, B = 16, C = 32, D = 48;
const vals = [0, -0, 1, -1, 0.5, 3, 1.0000001, 16777216, 0.1, 0.3, 1e20, -1e20, 1e-20, 3.4028235e38, 1e-45, Infinity, -Infinity, NaN].map(f);
let cases = 0, bad = 0, differ = 0;
for (let i = 0; i < vals.length; i++)
  for (let j = 0; j < vals.length; j++)
    for (let k = 0; k < vals.length; k++) {
      const a = [vals[i], vals[j], vals[k], vals[(i + j) % vals.length]];
      const b = [vals[j], vals[k], vals[i], vals[(j + k) % vals.length]];
      const c = [vals[k], vals[i], vals[j], vals[(k + i) % vals.length]];
      a.forEach((x, l) => { dv.setFloat32(A + 4 * l, x, true); dv.setFloat32(B + 4 * l, b[l], true); dv.setFloat32(C + 4 * l, c[l], true); });
      ex["f32x4.relaxed_madd"](A, B, C, D);
      for (let l = 0; l < 4; l++) {
        const got = dv.getFloat32(D + 4 * l, true);
        const fu = fma32(a[l], b[l], c[l]), un = unfused(a[l], b[l], c[l]);
        cases++;
        if (!(Object.is(got, fu) || Object.is(got, un))) { if (bad++ < 10) console.log(`MISMATCH madd(${a[l]}, ${b[l]}, ${c[l]}) = ${got}, fused ${fu}, unfused ${un}`); }
        if (!Object.is(fu, un)) differ++;
      }
    }
console.log(`${cases} lanes, ${bad} outside {fused, unfused}; the two differ on ${differ > 0 ? "some" : "no"} lanes`);
process.exit(bad ? 1 : 0);
