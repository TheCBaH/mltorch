The relaxed operations, executed under a node that enables them. Each lane of
f32x4.relaxed_madd must be the fused or the unfused result; the inputs include
cases where the two differ, so a wrong operation cannot pass.

  $ ./wasm_relaxed_gen.exe relaxed.wasm
  $ node --experimental-wasm-relaxed-simd relaxed.js relaxed.wasm
  23328 lanes, 0 outside {fused, unfused}; the two differ on some lanes
