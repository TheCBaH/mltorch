const fs = require("fs");
const inst = new WebAssembly.Instance(
  new WebAssembly.Module(fs.readFileSync(process.argv[2])),
  { m: { twice: (x) => 2 * x } });
const e = inst.exports;
const assert = require("assert");
const mem = new DataView(e.memory.buffer);
[1.5, 2.25, -4, 1e10].forEach((x, i) => mem.setFloat64(64 + 8 * i, x, true));
assert.strictEqual(e.sum_f64(64, 4), 2 * (1.5 + 2.25 - 4 + 1e10));
assert.strictEqual(e.sum_f64(64, 0), 0);
assert.strictEqual(e.fact(20n), 2432902008176640000n);
assert.strictEqual(e.fact(21n), BigInt.asIntN(64, 51090942171709440000n)); // wraps
assert.strictEqual(e.pick(1, 16), 65);
assert.strictEqual(e.pick(1, 17), 66);
assert.strictEqual(e.pick(0, 0x7FFFFFFF), -1); // the load is lazy: no trap
assert.strictEqual(e.bump(), 6);
assert.strictEqual(e.counter.value, 6);
assert.strictEqual(e.put_f32(128, 16777217), 16777216);
assert.strictEqual(e.put_f32(128, 0.1), Math.fround(0.1));
assert.strictEqual(new DataView(e.memory.buffer).getUint8(16), 65);
const custom = WebAssembly.Module.customSections(new WebAssembly.Module(fs.readFileSync(process.argv[2])), "abi");
assert.strictEqual(Buffer.from(custom[0]).toString(), "loop-wasm/1");
console.log("fixture ok");

// Independent validation: the host rejects deliberately corrupted encodings,
// so the check above is not vacuous.
const good = fs.readFileSync(process.argv[2]);
assert.ok(WebAssembly.validate(good));
const bad = (f) => { const c = Buffer.from(good); f(c); return WebAssembly.validate(c); };
assert.ok(!bad((c) => { c[0] = 1; }), "bad magic");
assert.ok(!bad((c) => { c[8 + 1] += 1; }), "wrong type-section length");
assert.ok(!WebAssembly.validate(good.subarray(0, good.length - 3)), "truncated");
// The last i32.const of `pick` is -1 (0x7f); making it 0x3f is the literal 63:
// still valid, so find the func-3 else-arm instead and break a branch depth.
const idx = good.indexOf(Buffer.from([0x0d, 0x01]));
assert.ok(idx > 0);
assert.ok(!bad((c) => { c[idx + 1] = 9; }), "wrong branch depth");
console.log("negative checks ok");
