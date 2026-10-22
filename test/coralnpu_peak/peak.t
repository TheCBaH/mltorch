Peak arithmetic throughput of the RVV core (rvv-core-mini-axi) and of the matrix engine of the
VME core (vme-core-mini-axi), measured by coralview's `ops` on waveforms of the RTL simulator. Each kernel is a typed Rivet module, built into a Coral guest
image around the runtime; `coral_peak.exe` prints its timed loop as coralview decodes the
image's bytes, runs it with the waveform traced, and checks its output slot against the host's.
The runtime's input transfer and result frame are in the trace too, outside every burst.
The same 47 cases are pinned byte for byte, without the Coral data, by baseline/peak_baseline.ml;
keep the two lists in step.

Each elementwise kernel repeats one instruction on independent accumulators, so no uop waits
for another's result, and every accumulator element must hold the sum when it is done.

VLEN is 128 bits and the backend has two units of each kind (integer ALU, integer multiply-add,
floating point), each taking one 128-bit uop a cycle. A unit's ceiling is therefore 2 x 128 / SEW
operations a cycle, twice that for multiply-adds; a widening instruction is limited by its
destination's width. At LMUL=4 one instruction is four uops, which keeps both units of its
kind busy. `ops` counts an instruction's operations at its first uop, so a burst or window can
exceed the ceiling by the part of one instruction its last cycles have not yet executed.

INT8: 32 operations a cycle, 64 for an 8-bit (modular) multiply-add, 32 for one widening to 16
bits.

  $ ./coral_peak.exe i8add e8,m4 vadd.vv
  64 iterations of:
    +00  vadd.vv v16, v8, v12
    +04  vadd.vv v20, v8, v12
    +08  vadd.vv v24, v8, v12
    +0c  vadd.vv v28, v8, v12
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  rvv-core-mini-axi: output as expected, 592 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 i8add.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 1937 cycles, 266 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu             262        16768        16384
    lsu               4          256            0
  16384 ops (0 floating point) in 1 bursts over 509 cycles (26.3% of the trace)
  ops/cycle: 8.458 over the trace, 32.189 in bursts, peak 32.250 over 256 cycles to cycle 1172
  bursts:
         first       last   cycles   instrs        ops ops/cycle
           918       1426      509      256      16384    32.189

  $ ./coral_peak.exe i8mul e8,m4 vmul.vv
  64 iterations of:
    +00  vmul.vv v16, v8, v12
    +04  vmul.vv v20, v8, v12
    +08  vmul.vv v24, v8, v12
    +0c  vmul.vv v28, v8, v12
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  rvv-core-mini-axi: output as expected, 592 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 i8mul.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 1937 cycles, 266 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               6          384            0
    lsu               4          256            0
    mul             256        16384        16384
  16384 ops (0 floating point) in 1 bursts over 509 cycles (26.3% of the trace)
  ops/cycle: 8.458 over the trace, 32.189 in bursts, peak 32.250 over 256 cycles to cycle 1172
  bursts:
         first       last   cycles   instrs        ops ops/cycle
           918       1426      509      256      16384    32.189

  $ ./coral_peak.exe i8macc e8,m4 vmacc.vv
  64 iterations of:
    +00  vmacc.vv v16, v8, v12
    +04  vmacc.vv v20, v8, v12
    +08  vmacc.vv v24, v8, v12
    +0c  vmacc.vv v28, v8, v12
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  rvv-core-mini-axi: output as expected, 595 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 i8macc.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 1940 cycles, 266 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               6          384            0
    lsu               4          256            0
    mac             256        16384        32768
  32768 ops (0 floating point) in 1 bursts over 511 cycles (26.3% of the trace)
  ops/cycle: 16.891 over the trace, 64.125 in bursts, peak 64.000 over 256 cycles to cycle 1172
  bursts:
         first       last   cycles   instrs        ops ops/cycle
           918       1428      511      256      32768    64.125

  $ ./coral_peak.exe i8wmacc e8,m2 vwmacc.vv
  64 iterations of:
    +00  vwmacc.vv v16, v8, v12
    +04  vwmacc.vv v20, v8, v12
    +08  vwmacc.vv v24, v8, v12
    +0c  vwmacc.vv v28, v8, v12
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  rvv-core-mini-axi: output as expected, 594 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 i8wmacc.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 1939 cycles, 266 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               6          192            0
    lsu               4          128            0
    mac             256         8192        16384
  16384 ops (0 floating point) in 1 bursts over 511 cycles (26.4% of the trace)
  ops/cycle: 8.450 over the trace, 32.063 in bursts, peak 32.000 over 256 cycles to cycle 1171
  bursts:
         first       last   cycles   instrs        ops ops/cycle
           917       1427      511      256      16384    32.063

INT16: 16, 32 for a multiply-add, 16 widening to 32 bits.

  $ ./coral_peak.exe i16add e16,m4 vadd.vv
  64 iterations of:
    +00  vadd.vv v16, v8, v12
    +04  vadd.vv v20, v8, v12
    +08  vadd.vv v24, v8, v12
    +0c  vadd.vv v28, v8, v12
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  rvv-core-mini-axi: output as expected, 592 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 i16add.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 1937 cycles, 266 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu             262         8384         8192
    lsu               4          128            0
  8192 ops (0 floating point) in 1 bursts over 509 cycles (26.3% of the trace)
  ops/cycle: 4.229 over the trace, 16.094 in bursts, peak 16.125 over 256 cycles to cycle 1172
  bursts:
         first       last   cycles   instrs        ops ops/cycle
           918       1426      509      256       8192    16.094

  $ ./coral_peak.exe i16mul e16,m4 vmul.vv
  64 iterations of:
    +00  vmul.vv v16, v8, v12
    +04  vmul.vv v20, v8, v12
    +08  vmul.vv v24, v8, v12
    +0c  vmul.vv v28, v8, v12
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  rvv-core-mini-axi: output as expected, 592 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 i16mul.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 1937 cycles, 266 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               6          192            0
    lsu               4          128            0
    mul             256         8192         8192
  8192 ops (0 floating point) in 1 bursts over 509 cycles (26.3% of the trace)
  ops/cycle: 4.229 over the trace, 16.094 in bursts, peak 16.125 over 256 cycles to cycle 1172
  bursts:
         first       last   cycles   instrs        ops ops/cycle
           918       1426      509      256       8192    16.094

  $ ./coral_peak.exe i16macc e16,m4 vmacc.vv
  64 iterations of:
    +00  vmacc.vv v16, v8, v12
    +04  vmacc.vv v20, v8, v12
    +08  vmacc.vv v24, v8, v12
    +0c  vmacc.vv v28, v8, v12
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  rvv-core-mini-axi: output as expected, 595 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 i16macc.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 1940 cycles, 266 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               6          192            0
    lsu               4          128            0
    mac             256         8192        16384
  16384 ops (0 floating point) in 1 bursts over 511 cycles (26.3% of the trace)
  ops/cycle: 8.445 over the trace, 32.063 in bursts, peak 32.000 over 256 cycles to cycle 1172
  bursts:
         first       last   cycles   instrs        ops ops/cycle
           918       1428      511      256      16384    32.063

  $ ./coral_peak.exe i16wmacc e16,m2 vwmacc.vv
  64 iterations of:
    +00  vwmacc.vv v16, v8, v12
    +04  vwmacc.vv v20, v8, v12
    +08  vwmacc.vv v24, v8, v12
    +0c  vwmacc.vv v28, v8, v12
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  rvv-core-mini-axi: output as expected, 594 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 i16wmacc.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 1939 cycles, 266 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               6           96            0
    lsu               4           64            0
    mac             256         4096         8192
  8192 ops (0 floating point) in 1 bursts over 511 cycles (26.4% of the trace)
  ops/cycle: 4.225 over the trace, 16.031 in bursts, peak 16.000 over 256 cycles to cycle 1171
  bursts:
         first       last   cycles   instrs        ops ops/cycle
           917       1427      511      256       8192    16.031

INT32: 8, 16 for a multiply-add.

  $ ./coral_peak.exe i32add e32,m4 vadd.vv
  64 iterations of:
    +00  vadd.vv v16, v8, v12
    +04  vadd.vv v20, v8, v12
    +08  vadd.vv v24, v8, v12
    +0c  vadd.vv v28, v8, v12
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  rvv-core-mini-axi: output as expected, 592 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 i32add.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 1937 cycles, 266 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu             262         4192         4096
    lsu               4           64            0
  4096 ops (0 floating point) in 1 bursts over 509 cycles (26.3% of the trace)
  ops/cycle: 2.115 over the trace, 8.047 in bursts, peak 8.062 over 256 cycles to cycle 1172
  bursts:
         first       last   cycles   instrs        ops ops/cycle
           918       1426      509      256       4096     8.047

  $ ./coral_peak.exe i32mul e32,m4 vmul.vv
  64 iterations of:
    +00  vmul.vv v16, v8, v12
    +04  vmul.vv v20, v8, v12
    +08  vmul.vv v24, v8, v12
    +0c  vmul.vv v28, v8, v12
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  rvv-core-mini-axi: output as expected, 592 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 i32mul.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 1937 cycles, 266 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               6           96            0
    lsu               4           64            0
    mul             256         4096         4096
  4096 ops (0 floating point) in 1 bursts over 509 cycles (26.3% of the trace)
  ops/cycle: 2.115 over the trace, 8.047 in bursts, peak 8.062 over 256 cycles to cycle 1172
  bursts:
         first       last   cycles   instrs        ops ops/cycle
           918       1426      509      256       4096     8.047

  $ ./coral_peak.exe i32macc e32,m4 vmacc.vv
  64 iterations of:
    +00  vmacc.vv v16, v8, v12
    +04  vmacc.vv v20, v8, v12
    +08  vmacc.vv v24, v8, v12
    +0c  vmacc.vv v28, v8, v12
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  rvv-core-mini-axi: output as expected, 595 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 i32macc.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 1940 cycles, 266 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               6           96            0
    lsu               4           64            0
    mac             256         4096         8192
  8192 ops (0 floating point) in 1 bursts over 511 cycles (26.3% of the trace)
  ops/cycle: 4.223 over the trace, 16.031 in bursts, peak 16.000 over 256 cycles to cycle 1172
  bursts:
         first       last   cycles   instrs        ops ops/cycle
           918       1428      511      256       8192    16.031

FP32: 8, 16 for a fused multiply-add.

  $ ./coral_peak.exe f32add e32,m4 vfadd.vv
  64 iterations of:
    +00  vfadd.vv v16, v8, v12
    +04  vfadd.vv v20, v8, v12
    +08  vfadd.vv v24, v8, v12
    +0c  vfadd.vv v28, v8, v12
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  rvv-core-mini-axi: output as expected, 594 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 f32add.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 1939 cycles, 266 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               6           96            0
    fma             256         4096         4096
    lsu               4           64            0
  4096 ops (4096 floating point) in 1 bursts over 510 cycles (26.3% of the trace)
  ops/cycle: 2.112 over the trace, 8.031 in bursts, peak 8.062 over 256 cycles to cycle 1173
  bursts:
         first       last   cycles   instrs        ops ops/cycle
           918       1427      510      256       4096     8.031

  $ ./coral_peak.exe f32mul e32,m4 vfmul.vv
  64 iterations of:
    +00  vfmul.vv v16, v8, v12
    +04  vfmul.vv v20, v8, v12
    +08  vfmul.vv v24, v8, v12
    +0c  vfmul.vv v28, v8, v12
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  rvv-core-mini-axi: output as expected, 594 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 f32mul.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 1939 cycles, 266 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               6           96            0
    fma             256         4096         4096
    lsu               4           64            0
  4096 ops (4096 floating point) in 1 bursts over 510 cycles (26.3% of the trace)
  ops/cycle: 2.112 over the trace, 8.031 in bursts, peak 8.062 over 256 cycles to cycle 1173
  bursts:
         first       last   cycles   instrs        ops ops/cycle
           918       1427      510      256       4096     8.031

  $ ./coral_peak.exe f32macc e32,m4 vfmacc.vv
  64 iterations of:
    +00  vfmacc.vv v16, v8, v12
    +04  vfmacc.vv v20, v8, v12
    +08  vfmacc.vv v24, v8, v12
    +0c  vfmacc.vv v28, v8, v12
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  rvv-core-mini-axi: output as expected, 595 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 f32macc.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 1940 cycles, 266 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               6           96            0
    fma             256         4096         8192
    lsu               4           64            0
  8192 ops (8192 floating point) in 1 bursts over 511 cycles (26.3% of the trace)
  ops/cycle: 4.223 over the trace, 16.031 in bursts, peak 16.000 over 256 cycles to cycle 1172
  bursts:
         first       last   cycles   instrs        ops ops/cycle
           918       1428      511      256       8192    16.031

BF16 into FP32 accumulators: 16, as FP32, since the destination's width sets the lanes.

  $ ./coral_peak.exe bf16wmacc e16,m2 vfwmaccbf16.vv
  64 iterations of:
    +00  vfwmaccbf16.vv v16, v8, v12
    +04  vfwmaccbf16.vv v20, v8, v12
    +08  vfwmaccbf16.vv v24, v8, v12
    +0c  vfwmaccbf16.vv v28, v8, v12
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  rvv-core-mini-axi: output as expected, 594 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 bf16wmacc.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 1939 cycles, 266 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               6           96            0
    fma             256         4096         8192
    lsu               4           64            0
  8192 ops (8192 floating point) in 1 bursts over 511 cycles (26.4% of the trace)
  ops/cycle: 4.225 over the trace, 16.031 in bursts, peak 16.000 over 256 cycles to cycle 1171
  bursts:
         first       last   cycles   instrs        ops ops/cycle
           917       1427      511      256       8192    16.031

At LMUL=1 an instruction is one uop, and the scalar core passes about one vector instruction a
cycle, so the same FP32 multiply-add on sixteen accumulators stays near half its ceiling:

  $ ./coral_peak.exe f32macc_m1 e32,m1 vfmacc.vv
  64 iterations of:
    +00  vfmacc.vv v16, v8, v12
    +04  vfmacc.vv v17, v8, v12
    +08  vfmacc.vv v18, v8, v12
    +0c  vfmacc.vv v19, v8, v12
    +10  vfmacc.vv v20, v8, v12
    +14  vfmacc.vv v21, v8, v12
    +18  vfmacc.vv v22, v8, v12
    +1c  vfmacc.vv v23, v8, v12
    +20  vfmacc.vv v24, v8, v12
    +24  vfmacc.vv v25, v8, v12
    +28  vfmacc.vv v26, v8, v12
    +2c  vfmacc.vv v27, v8, v12
    +30  vfmacc.vv v28, v8, v12
    +34  vfmacc.vv v29, v8, v12
    +38  vfmacc.vv v30, v8, v12
    +3c  vfmacc.vv v31, v8, v12
    +40  addi x11, x11, -1
    +44  bne x11, x0, -68
  rvv-core-mini-axi: output as expected, 1085 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 f32macc_m1.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 2454 cycles, 1058 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu              18           72            0
    fma            1024         4096         8192
    lsu              16           64            0
  8192 ops (8192 floating point) in 1 bursts over 953 cycles (38.8% of the trace)
  ops/cycle: 3.338 over the trace, 8.596 in bursts, peak 8.562 over 256 cycles to cycle 1197
  bursts:
         first       last   cycles   instrs        ops ops/cycle
           942       1894      953     1024       8192     8.596

Matrix multiplies, RVV only. The loads run through one unit, which takes about 2 + n cycles
for a vector of n 16-byte beats and 3 for a scalar load, whatever its width. Each kernel holds
two or three rows of C in LMUL=8 accumulators, streams B a row at a time and takes A's values
as scalars (`.vx`, `.vf`), so a row of B serves every row of C and each value of A a whole
row of B. Every multiply's C is checked against the product.

Integer A values pack several to a word, so one `lw` serves two k steps (INT8) or two rows
(INT16), and the widening multiply-adds reach their ceilings, 32 and 16:

  $ ./coral_peak.exe gemm_i8
  C[8x64] int16 = A[8x128] int8 x B[128x64] int8, 2 rows at a time; inner loop:
    +00  lw x15, 4(x11)
    +04  vle8.v v4, 0(x13)
    +08  srli x28, x14, 8
    +0c  vwmacc.vx v16, x14, v0
    +10  vwmacc.vx v24, x28, v0
    +14  addi x12, x12, 128
    +18  vle8.v v0, 0(x12)
    +1c  srli x29, x14, 16
    +20  srli x30, x14, 24
    +24  vwmacc.vx v16, x29, v4
    +28  vwmacc.vx v24, x30, v4
    +2c  addi x13, x13, 128
    +30  vle8.v v4, 0(x13)
    +34  srli x28, x15, 8
    +38  vwmacc.vx v16, x15, v0
    +3c  vwmacc.vx v24, x28, v0
    +40  addi x12, x12, 128
    +44  addi x11, x11, 8
    +48  lw x14, 0(x11)
    +4c  vle8.v v0, 0(x12)
    +50  srli x29, x15, 16
    +54  srli x30, x15, 24
    +58  vwmacc.vx v16, x29, v4
    +5c  vwmacc.vx v24, x30, v4
    +60  addi x13, x13, 128
    +64  addi x10, x10, -1
    +68  bne x10, x0, -104
  rvv-core-mini-axi: output as expected, 4283 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 gemm_i8.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 7356 cycles, 1557 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               8          512            0
    lsu             525        33600            0
    mac            1024        65536       131072
  131072 ops (0 floating point) in 1 bursts over 4181 cycles (56.8% of the trace)
  ops/cycle: 17.818 over the trace, 31.349 in bursts, peak 32.500 over 256 cycles to cycle 2832
  bursts:
         first       last   cycles   instrs        ops ops/cycle
          2577       6757     4181     1024     131072    31.349

  $ ./coral_peak.exe gemm_i16
  C[9x32] int32 = A[9x128] int16 x B[128x32] int16, 3 rows at a time; inner loop:
    +00  lw x5, 12(x11)
    +04  lw x6, 16(x11)
    +08  lw x7, 20(x11)
    +0c  vle16.v v4, 0(x13)
    +10  srli x28, x14, 16
    +14  srli x29, x15, 16
    +18  srli x30, x16, 16
    +1c  vwmacc.vx v8, x14, v0
    +20  vwmacc.vx v16, x28, v0
    +24  vwmacc.vx v24, x15, v0
    +28  addi x12, x12, 128
    +2c  vle16.v v0, 0(x12)
    +30  vwmacc.vx v8, x29, v4
    +34  vwmacc.vx v16, x16, v4
    +38  vwmacc.vx v24, x30, v4
    +3c  addi x13, x13, 128
    +40  addi x11, x11, 24
    +44  lw x14, 0(x11)
    +48  lw x15, 4(x11)
    +4c  lw x16, 8(x11)
    +50  vle16.v v4, 0(x13)
    +54  srli x28, x5, 16
    +58  srli x29, x6, 16
    +5c  srli x30, x7, 16
    +60  vwmacc.vx v8, x5, v0
    +64  vwmacc.vx v16, x28, v0
    +68  vwmacc.vx v24, x6, v0
    +6c  addi x12, x12, 128
    +70  vle16.v v0, 0(x12)
    +74  vwmacc.vx v8, x29, v4
    +78  vwmacc.vx v16, x7, v4
    +7c  vwmacc.vx v24, x30, v4
    +80  addi x13, x13, 128
    +84  addi x10, x10, -1
    +88  bne x10, x0, -136
  rvv-core-mini-axi: output as expected, 4806 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 gemm_i16.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 8098 cycles, 1558 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               9          288            0
    lsu             397        12704            0
    mac            1152        36864        73728
  73728 ops (0 floating point) in 1 bursts over 4685 cycles (57.9% of the trace)
  ops/cycle: 9.104 over the trace, 15.737 in bursts, peak 16.250 over 256 cycles to cycle 3050
  bursts:
         first       last   cycles   instrs        ops ops/cycle
          2796       7480     4685     1152      73728    15.737

A 32-bit value of A costs a scalar load of its own. With 128-byte rows of B, a k step keeps
the load unit busy for about 19 cycles against 12 of multiply-adds, which bounds INT32 and
FP32 near 16 x 12 / 19 = 10:

  $ ./coral_peak.exe gemm_i32
  C[9x32] int32 = A[9x64] int32 x B[64x32] int32, 3 rows at a time; inner loop:
    +00  lw x6, 12(x11)
    +04  lw x7, 16(x11)
    +08  lw x28, 20(x11)
    +0c  vle32.v v0, 0(x12)
    +10  vmacc.vx v8, x14, v0
    +14  vmacc.vx v16, x15, v0
    +18  vmacc.vx v24, x16, v0
    +1c  addi x12, x12, 128
    +20  addi x11, x11, 24
    +24  lw x14, 0(x11)
    +28  lw x15, 4(x11)
    +2c  lw x16, 8(x11)
    +30  vle32.v v0, 0(x12)
    +34  vmacc.vx v8, x6, v0
    +38  vmacc.vx v16, x7, v0
    +3c  vmacc.vx v24, x28, v0
    +40  addi x12, x12, 128
    +44  addi x10, x10, -1
    +48  bne x10, x0, -72
  rvv-core-mini-axi: output as expected, 3857 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 gemm_i32.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 7137 cycles, 786 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               9          288            0
    lsu             201         6432            0
    mac             576        18432        36864
  36864 ops (0 floating point) in 1 bursts over 3740 cycles (52.4% of the trace)
  ops/cycle: 5.165 over the trace, 9.857 in bursts, peak 10.250 over 256 cycles to cycle 3036
  bursts:
         first       last   cycles   instrs        ops ops/cycle
          2782       6521     3740      576      36864     9.857

  $ ./coral_peak.exe gemm_f32
  C[9x32] fp32 = A[9x64] fp32 x B[64x32] fp32, 3 rows at a time; inner loop:
    +00  flw f1, 12(x11)
    +04  flw f2, 16(x11)
    +08  flw f3, 20(x11)
    +0c  vle32.v v0, 0(x12)
    +10  vfmacc.vf v8, f14, v0
    +14  vfmacc.vf v16, f15, v0
    +18  vfmacc.vf v24, f16, v0
    +1c  addi x12, x12, 128
    +20  addi x11, x11, 24
    +24  flw f14, 0(x11)
    +28  flw f15, 4(x11)
    +2c  flw f16, 8(x11)
    +30  vle32.v v0, 0(x12)
    +34  vfmacc.vf v8, f1, v0
    +38  vfmacc.vf v16, f2, v0
    +3c  vfmacc.vf v24, f3, v0
    +40  addi x12, x12, 128
    +44  addi x10, x10, -1
    +48  bne x10, x0, -72
  rvv-core-mini-axi: output as expected, 3861 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 gemm_f32.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 7141 cycles, 786 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               9          288            0
    fma             576        18432        36864
    lsu             201         6432            0
  36864 ops (36864 floating point) in 1 bursts over 3742 cycles (52.4% of the trace)
  ops/cycle: 5.162 over the trace, 9.851 in bursts, peak 10.250 over 256 cycles to cycle 3037
  bursts:
         first       last   cycles   instrs        ops ops/cycle
          2783       6524     3742      576      36864     9.851

BF16 halves B's rows, to about 15 cycles of loads a k step, and 16 x 12 / 15 = 12.8. `flh`
loads each of A's BF16 values (Zfbfmin provides it):

  $ ./coral_peak.exe gemm_bf16
  C[9x32] fp32 = A[9x128] bf16 x B[128x32] bf16, 3 rows at a time; inner loop:
    +00  flh f13, 6(x11)
    +04  flh f14, 8(x11)
    +08  flh f15, 10(x11)
    +0c  vle16.v v4, 0(x13)
    +10  vfwmaccbf16.vf v8, f10, v0
    +14  vfwmaccbf16.vf v16, f11, v0
    +18  vfwmaccbf16.vf v24, f12, v0
    +1c  addi x11, x11, 12
    +20  flh f10, 0(x11)
    +24  flh f11, 2(x11)
    +28  flh f12, 4(x11)
    +2c  addi x12, x12, 128
    +30  vle16.v v0, 0(x12)
    +34  vfwmaccbf16.vf v8, f13, v4
    +38  vfwmaccbf16.vf v16, f14, v4
    +3c  vfwmaccbf16.vf v24, f15, v4
    +40  addi x13, x13, 128
    +44  addi x10, x10, -1
    +48  bne x10, x0, -72
  rvv-core-mini-axi: output as expected, 5953 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 gemm_bf16.fst
  root VRvvCoreMiniAxi.core.RvvCoreMiniAxi: 9237 cycles, 1558 vector instructions dispatched (0 not decoded)
    unit   instructions     elements          ops
    alu               9          288            0
    fma            1152        36864        73728
    lsu             397        12704            0
  73728 ops (73728 floating point) in 1 bursts over 5838 cycles (63.2% of the trace)
  ops/cycle: 7.982 over the trace, 12.629 in bursts, peak 13.000 over 256 cycles to cycle 3049
  bursts:
         first       last   cycles   instrs        ops ops/cycle
          2782       8619     5838     1152      73728    12.629

The width of C need not be the lanes of one vector (64 for INT8, 32 for the others): `gemm_<type>_w<N>`
sets it. Narrower, three quarters of the lanes, one tile runs a vector with `vl` = W: the
instructions and cycles are the same, so the operations, and the rate, drop to three quarters.
Wider, twice the lanes, each block of rows runs once per column tile over the same A, so the
instructions double and the rate in the burst stays within 0.5% of the full-width kernel's. Only the
shape line, the result and the rates are shown:

  $ ./coral_peak.exe gemm_i8_w48 | grep -v '^  +'
  C[8x48] int16 = A[8x128] int8 x B[128x48] int8, 2 rows at a time; inner loop:
  rvv-core-mini-axi: output as expected, 4283 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 gemm_i8_w48.fst | grep -E 'bursts over|^ops/cycle:'
  98304 ops (0 floating point) in 1 bursts over 4181 cycles (60.0% of the trace)
  ops/cycle: 14.106 over the trace, 23.512 in bursts, peak 24.375 over 256 cycles to cycle 2471

  $ ./coral_peak.exe gemm_i8_w128 | grep -v '^  +'
  C[8x128] int16 = A[8x128] int8 x B[128x128] int8, 2 rows at a time; inner loop:
  rvv-core-mini-axi: output as expected, 8516 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 gemm_i8_w128.fst | grep -E 'bursts over|^ops/cycle:'
  262144 ops (0 floating point) in 1 bursts over 8406 cycles (64.0% of the trace)
  ops/cycle: 19.965 over the trace, 31.185 in bursts, peak 32.500 over 256 cycles to cycle 4299

  $ ./coral_peak.exe gemm_i16_w24 | grep -v '^  +'
  C[9x24] int32 = A[9x128] int16 x B[128x24] int16, 3 rows at a time; inner loop:
  rvv-core-mini-axi: output as expected, 4806 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 gemm_i16_w24.fst | grep -E 'bursts over|^ops/cycle:'
  55296 ops (0 floating point) in 1 bursts over 4685 cycles (60.9% of the trace)
  ops/cycle: 7.193 over the trace, 11.803 in bursts, peak 12.188 over 256 cycles to cycle 2665

  $ ./coral_peak.exe gemm_i16_w64 | grep -v '^  +'
  C[9x64] int32 = A[9x128] int16 x B[128x64] int16, 3 rows at a time; inner loop:
  rvv-core-mini-axi: output as expected, 9565 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 gemm_i16_w64.fst | grep -E 'bursts over|^ops/cycle:'
  147456 ops (0 floating point) in 1 bursts over 9436 cycles (65.1% of the trace)
  ops/cycle: 10.172 over the trace, 15.627 in bursts, peak 16.250 over 256 cycles to cycle 4602

  $ ./coral_peak.exe gemm_i32_w24 | grep -v '^  +'
  C[9x24] int32 = A[9x64] int32 x B[64x24] int32, 3 rows at a time; inner loop:
  rvv-core-mini-axi: output as expected, 3857 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 gemm_i32_w24.fst | grep -E 'bursts over|^ops/cycle:'
  27648 ops (0 floating point) in 1 bursts over 3740 cycles (55.7% of the trace)
  ops/cycle: 4.119 over the trace, 7.393 in bursts, peak 7.688 over 256 cycles to cycle 2651

  $ ./coral_peak.exe gemm_i32_w64 | grep -v '^  +'
  C[9x64] int32 = A[9x64] int32 x B[64x64] int32, 3 rows at a time; inner loop:
  rvv-core-mini-axi: output as expected, 7666 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 gemm_i32_w64.fst | grep -E 'bursts over|^ops/cycle:'
  73728 ops (0 floating point) in 1 bursts over 7541 cycles (59.8% of the trace)
  ops/cycle: 5.846 over the trace, 9.777 in bursts, peak 10.250 over 256 cycles to cycle 4614

  $ ./coral_peak.exe gemm_f32_w24 | grep -v '^  +'
  C[9x24] fp32 = A[9x64] fp32 x B[64x24] fp32, 3 rows at a time; inner loop:
  rvv-core-mini-axi: output as expected, 3861 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 gemm_f32_w24.fst | grep -E 'bursts over|^ops/cycle:'
  27648 ops (27648 floating point) in 1 bursts over 3742 cycles (55.7% of the trace)
  ops/cycle: 4.116 over the trace, 7.389 in bursts, peak 7.688 over 256 cycles to cycle 2652

  $ ./coral_peak.exe gemm_f32_w64 | grep -v '^  +'
  C[9x64] fp32 = A[9x64] fp32 x B[64x64] fp32, 3 rows at a time; inner loop:
  rvv-core-mini-axi: output as expected, 7674 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 gemm_f32_w64.fst | grep -E 'bursts over|^ops/cycle:'
  73728 ops (73728 floating point) in 1 bursts over 7547 cycles (59.8% of the trace)
  ops/cycle: 5.843 over the trace, 9.769 in bursts, peak 10.250 over 256 cycles to cycle 4615

  $ ./coral_peak.exe gemm_bf16_w24 | grep -v '^  +'
  C[9x24] fp32 = A[9x128] bf16 x B[128x24] bf16, 3 rows at a time; inner loop:
  rvv-core-mini-axi: output as expected, 5953 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 gemm_bf16_w24.fst | grep -E 'bursts over|^ops/cycle:'
  55296 ops (55296 floating point) in 1 bursts over 5838 cycles (66.1% of the trace)
  ops/cycle: 6.265 over the trace, 9.472 in bursts, peak 9.750 over 256 cycles to cycle 2664

  $ ./coral_peak.exe gemm_bf16_w64 | grep -v '^  +'
  C[9x64] fp32 = A[9x128] bf16 x B[128x64] bf16, 3 rows at a time; inner loop:
  rvv-core-mini-axi: output as expected, 11859 cycles in the kernel
  $ coralview ops --platform rvv-core-mini-axi --peak-window 256 gemm_bf16_w64.fst | grep -E 'bursts over|^ops/cycle:'
  147456 ops (147456 floating point) in 1 bursts over 11736 cycles (69.9% of the trace)
  ops/cycle: 8.787 over the trace, 12.564 in bursts, peak 13.000 over 256 cycles to cycle 4601

Zvt tile multiplies on the VME core's matrix engine, one instruction repeated on the four
independent 32-bit accumulator tiles (mt0, mt4, mt8, mt12, each 16 x 16 elements over four tile
registers), A and B held in vector registers. An instruction is 16 x 16 x tk multiply-adds,
tk = 1 for FP32, 2 for BF16 and 4 for INT8. The PE array (64 PEs, each four INT8, two BF16 or
one FP32 product a cycle) serves a 16 x 16 tile in four steps, so a tile multiply issues every
4 cycles at best, whatever its type: the ceiling is 128 x tk operations a cycle, 128 for FP32,
256 for BF16 and 512 for INT8, eight times the vector unit's INT8 multiply-add. Every kernel
reaches it over its best window; `coralview tiles` shows the tile register file's 16 write
ports (64 results a cycle) busy throughout the burst. The INT8 A is all ones: -1 to vtmms, 255
to vtmmu.

The VME simulator drains a kernel's last tile stores in a varying number of cycles, up to
three more from one run to the next, so the kernel's cycles and the trace's length are masked
below; the bursts end before the stores, and their rates do not vary.

  $ ./coral_peak.exe tile_f32 tile vtfmm.tvv | sed -E 's/[0-9]+ cycles in the kernel/N cycles in the kernel/'
  64 iterations of:
    +00  vtfmm.tvv mt0, v8, v16
    +04  vtfmm.tvv mt4, v8, v16
    +08  vtfmm.tvv mt8, v8, v16
    +0c  vtfmm.tvv mt12, v8, v16
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  vme-core-mini-axi: output as expected, N cycles in the kernel
  $ coralview ops --platform vme-core-mini-axi --peak-window 256 tile_f32.fst | grep -v '^root' | sed -E 's/ \([0-9.]+% of the trace\)//; s/^ops\/cycle: [0-9.]+ over the trace, /ops\/cycle: /'
    unit   instructions     elements          ops
    alu               2           32            0
    vme             260        65600       131072
    vmelsu           64         1024            0
  131072 ops (131072 floating point) in 1 bursts over 1031 cycles
  ops/cycle: 127.131 in bursts, peak 128.000 over 256 cycles to cycle 4075
  bursts:
         first       last   cycles   instrs        ops ops/cycle
          3805       4835     1031      256     131072   127.131

  $ ./coral_peak.exe tile_bf16 tile vtfmm.alt.tvv | sed -E 's/[0-9]+ cycles in the kernel/N cycles in the kernel/'
  64 iterations of:
    +00  vtfmm.alt.tvv mt0, v8, v16
    +04  vtfmm.alt.tvv mt4, v8, v16
    +08  vtfmm.alt.tvv mt8, v8, v16
    +0c  vtfmm.alt.tvv mt12, v8, v16
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  vme-core-mini-axi: output as expected, N cycles in the kernel
  $ coralview ops --platform vme-core-mini-axi --peak-window 256 tile_bf16.fst | grep -v '^root' | sed -E 's/ \([0-9.]+% of the trace\)//; s/^ops\/cycle: [0-9.]+ over the trace, /ops\/cycle: /'
    unit   instructions     elements          ops
    alu               4           64            0
    vme             260       131136       262144
    vmelsu           64         1024            0
  262144 ops (262144 floating point) in 1 bursts over 1031 cycles
  ops/cycle: 254.262 in bursts, peak 256.000 over 256 cycles to cycle 4078
  bursts:
         first       last   cycles   instrs        ops ops/cycle
          3808       4838     1031      256     262144   254.262

  $ ./coral_peak.exe tile_i8u tile vtmmu.tvv | sed -E 's/[0-9]+ cycles in the kernel/N cycles in the kernel/'
  64 iterations of:
    +00  vtmmu.tvv mt0, v8, v16
    +04  vtmmu.tvv mt4, v8, v16
    +08  vtmmu.tvv mt8, v8, v16
    +0c  vtmmu.tvv mt12, v8, v16
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  vme-core-mini-axi: output as expected, N cycles in the kernel
  $ coralview ops --platform vme-core-mini-axi --peak-window 256 tile_i8u.fst | grep -v '^root' | sed -E 's/ \([0-9.]+% of the trace\)//; s/^ops\/cycle: [0-9.]+ over the trace, /ops\/cycle: /'
    unit   instructions     elements          ops
    alu               8          128            0
    vme             260       262208       524288
    vmelsu           64         1024            0
  524288 ops (0 floating point) in 1 bursts over 1032 cycles
  ops/cycle: 508.031 in bursts, peak 512.000 over 256 cycles to cycle 4082
  bursts:
         first       last   cycles   instrs        ops ops/cycle
          3811       4842     1032      256     524288   508.031

  $ ./coral_peak.exe tile_i8s tile vtmms.tvv | sed -E 's/[0-9]+ cycles in the kernel/N cycles in the kernel/'
  64 iterations of:
    +00  vtmms.tvv mt0, v8, v16
    +04  vtmms.tvv mt4, v8, v16
    +08  vtmms.tvv mt8, v8, v16
    +0c  vtmms.tvv mt12, v8, v16
    +10  addi x11, x11, -1
    +14  bne x11, x0, -20
  vme-core-mini-axi: output as expected, N cycles in the kernel
  $ coralview ops --platform vme-core-mini-axi --peak-window 256 tile_i8s.fst | grep -v '^root' | sed -E 's/ \([0-9.]+% of the trace\)//; s/^ops\/cycle: [0-9.]+ over the trace, /ops\/cycle: /'
    unit   instructions     elements          ops
    alu               8          128            0
    vme             260       262208       524288
    vmelsu           64         1024            0
  524288 ops (0 floating point) in 1 bursts over 1032 cycles
  ops/cycle: 508.031 in bursts, peak 512.000 over 256 cycles to cycle 4082
  bursts:
         first       last   cycles   instrs        ops ops/cycle
          3811       4842     1032      256     524288   508.031

Matrix multiplies through Zvt: C = A x B in 16 x 16 tiles, M up to 16 as one row of tiles that
many rows deep, a multiple of 32 as blocks of two full rows, W in passes of two tile columns, the
last as wide as what is left of W. A pass keeps its tiles in the four accumulators, and each
loaded slice of A or B serves every tile of its row or column. The tile multiplies are given the
real rows (tm) and columns (tn), so `ops` counts what a partial tile computes. A and B are packed
per tile, k-major, zero beyond M and W; a k step of two tiles is 128 contiguous bytes, taken by
two `vle32.v` of 64 (at the tile configuration's SEW a 32-bit element spans four registers, so a
load fills both tiles' interleaved slices without a change of `vtype`); one tile's slices load
one at a time at their own width. The next step's operands load into a second buffer while the
tile multiplies read the first. INT8 is signed on both sides (vtmms with `altfmt`) and
accumulates into INT32, where the RVV kernel accumulates into INT16; these inputs give the same
values in either.

The engine's own shape, two full tile rows by two full tile columns: a k step is four 64-byte
loads, about 6 cycles each, against 16 cycles of tile multiplies, so the load unit sets the pace
at 16 / 24 of the ceiling, about 341, 171 and 85 operations a cycle for INT8, BF16 and FP32,
whatever the type. Each kernel has 32 k steps a block and fits DTCM: with B in testbench RAM,
its loads took three times as long.

  $ ./coral_peak.exe tgemm_i8 | sed -E 's/[0-9]+ cycles in the kernel/N cycles in the kernel/'
  C[64x32] int32 = A[64x128] int8 x B[128x32] int8, tiles 16 deep, 2 columns (the last 16 wide) in 1 pass; inner loop:
    +00  vle32.v v0, 0(x11)
    +04  addi x11, x11, 64
    +08  vle32.v v4, 0(x11)
    +0c  addi x11, x11, 64
    +10  vle32.v v24, 0(x12)
    +14  addi x12, x12, 64
    +18  vle32.v v28, 0(x12)
    +1c  addi x12, x12, 64
    +20  vtmms.tvv mt0, v8, v16
    +24  vtmms.tvv mt8, v9, v16
    +28  vtmms.tvv mt4, v8, v17
    +2c  vtmms.tvv mt12, v9, v17
    +30  vle32.v v8, 0(x11)
    +34  addi x11, x11, 64
    +38  vle32.v v12, 0(x11)
    +3c  addi x11, x11, 64
    +40  vle32.v v16, 0(x12)
    +44  addi x12, x12, 64
    +48  vle32.v v20, 0(x12)
    +4c  addi x12, x12, 64
    +50  vtmms.tvv mt0, v0, v24
    +54  vtmms.tvv mt8, v1, v24
    +58  vtmms.tvv mt4, v0, v25
    +5c  vtmms.tvv mt12, v1, v25
    +60  addi x16, x16, -1
    +64  bne x16, x0, -100
  vme-core-mini-axi: output as expected, N cycles in the kernel
  $ coralview ops --platform vme-core-mini-axi --peak-window 256 tgemm_i8.fst | grep -E 'bursts over|^ops/cycle:' | sed -E 's/ \([0-9.]+% of the trace\)//; s/^ops\/cycle: [0-9.]+ over the trace, /ops\/cycle: /'
  524288 ops (0 floating point) in 2 bursts over 1510 cycles
  ops/cycle: 347.211 in bursts, peak 352.000 over 256 cycles to cycle 8509

  $ ./coral_peak.exe tgemm_bf16 | grep -v '^  +' | sed -E 's/[0-9]+ cycles in the kernel/N cycles in the kernel/'
  C[64x32] fp32 = A[64x64] bf16 x B[64x32] bf16, tiles 16 deep, 2 columns (the last 16 wide) in 1 pass; inner loop:
  vme-core-mini-axi: output as expected, N cycles in the kernel
  $ coralview ops --platform vme-core-mini-axi --peak-window 256 tgemm_bf16.fst | grep -E 'bursts over|^ops/cycle:' | sed -E 's/ \([0-9.]+% of the trace\)//; s/^ops\/cycle: [0-9.]+ over the trace, /ops\/cycle: /'
  262144 ops (262144 floating point) in 2 bursts over 1510 cycles
  ops/cycle: 173.605 in bursts, peak 176.000 over 256 cycles to cycle 8509

  $ ./coral_peak.exe tgemm_f32 | grep -v '^  +' | sed -E 's/[0-9]+ cycles in the kernel/N cycles in the kernel/'
  C[64x32] fp32 = A[64x32] fp32 x B[32x32] fp32, tiles 16 deep, 2 columns (the last 16 wide) in 1 pass; inner loop:
  vme-core-mini-axi: output as expected, N cycles in the kernel
  $ coralview ops --platform vme-core-mini-axi --peak-window 256 tgemm_f32.fst | grep -E 'bursts over|^ops/cycle:' | sed -E 's/ \([0-9.]+% of the trace\)//; s/^ops\/cycle: [0-9.]+ over the trace, /ops\/cycle: /'
  131072 ops (131072 floating point) in 2 bursts over 1510 cycles
  ops/cycle: 86.803 in bursts, peak 88.000 over 256 cycles to cycle 8509

The RVV kernels' shapes, the same A, B and C as `gemm_<type>[_w<N>]`: M is 8 (INT8) or 9, so
one row of tiles 8 or 9 deep. A step is two tile multiplies, half or less of each tile's rows
used, for one tile of A (loaded a slice at a time) and two of B. INT8 at 4 x 3 + 2 x 6 = 24
cycles a step makes 2 x 8 x 16 x 4 x 2 operations, 85 a cycle; BF16 at 2 x 4 + 12 = 20 cycles,
58; FP32 at 6 + 12 = 18, 32. Against the RVV kernels' 31.3, 12.6 and 9.9: 2.5, 4.6 and 3.3
times. W = 24 leaves the second tile column half used (tn = 8), at the same cycles; W = 48
gives the INT8 kernel a pass of one column, a step at the same loads but half the tiles; a
wider W adds passes, each with its own start.

  $ ./coral_peak.exe tgemm_i8_rvv | sed -E 's/[0-9]+ cycles in the kernel/N cycles in the kernel/'
  C[8x64] int32 = A[8x128] int8 x B[128x64] int8, tiles 8 deep, 4 columns (the last 16 wide) in 2 passes; inner loop:
    +00  vle8.v v0, 0(x11)
    +04  addi x11, x11, 16
    +08  vle8.v v2, 0(x11)
    +0c  addi x11, x11, 16
    +10  vle8.v v4, 0(x11)
    +14  addi x11, x11, 16
    +18  vle8.v v6, 0(x11)
    +1c  addi x11, x11, 16
    +20  vle32.v v24, 0(x12)
    +24  addi x12, x12, 64
    +28  vle32.v v28, 0(x12)
    +2c  addi x12, x12, 64
    +30  vtmms.tvv mt0, v8, v16
    +34  vtmms.tvv mt4, v8, v17
    +38  vle8.v v8, 0(x11)
    +3c  addi x11, x11, 16
    +40  vle8.v v10, 0(x11)
    +44  addi x11, x11, 16
    +48  vle8.v v12, 0(x11)
    +4c  addi x11, x11, 16
    +50  vle8.v v14, 0(x11)
    +54  addi x11, x11, 16
    +58  vle32.v v16, 0(x12)
    +5c  addi x12, x12, 64
    +60  vle32.v v20, 0(x12)
    +64  addi x12, x12, 64
    +68  vtmms.tvv mt0, v0, v24
    +6c  vtmms.tvv mt4, v0, v25
    +70  addi x16, x16, -1
    +74  bne x16, x0, -116
  vme-core-mini-axi: output as expected, N cycles in the kernel
  $ coralview ops --platform vme-core-mini-axi --peak-window 256 tgemm_i8_rvv.fst | grep -E 'bursts over|^ops/cycle:' | sed -E 's/ \([0-9.]+% of the trace\)//; s/^ops\/cycle: [0-9.]+ over the trace, /ops\/cycle: /'
  131072 ops (0 floating point) in 1 bursts over 1646 cycles
  ops/cycle: 79.631 in bursts, peak 88.000 over 256 cycles to cycle 3754

  $ ./coral_peak.exe tgemm_i8_rvv_w48 | grep -v '^  +' | sed -E 's/[0-9]+ cycles in the kernel/N cycles in the kernel/'
  C[8x48] int32 = A[8x128] int8 x B[128x48] int8, tiles 8 deep, 3 columns (the last 16 wide) in 2 passes; inner loop:
  vme-core-mini-axi: output as expected, N cycles in the kernel
  $ coralview ops --platform vme-core-mini-axi --peak-window 256 tgemm_i8_rvv_w48.fst | grep -E 'bursts over|^ops/cycle:' | sed -E 's/ \([0-9.]+% of the trace\)//; s/^ops\/cycle: [0-9.]+ over the trace, /ops\/cycle: /'
  98304 ops (0 floating point) in 1 bursts over 1648 cycles
  ops/cycle: 59.650 in bursts, peak 88.000 over 256 cycles to cycle 3201

  $ ./coral_peak.exe tgemm_i8_rvv_w128 | grep -v '^  +' | sed -E 's/[0-9]+ cycles in the kernel/N cycles in the kernel/'
  C[8x128] int32 = A[8x128] int8 x B[128x128] int8, tiles 8 deep, 8 columns (the last 16 wide) in 4 passes; inner loop:
  vme-core-mini-axi: output as expected, N cycles in the kernel
  $ coralview ops --platform vme-core-mini-axi --peak-window 256 tgemm_i8_rvv_w128.fst | grep -E 'bursts over|^ops/cycle:' | sed -E 's/ \([0-9.]+% of the trace\)//; s/^ops\/cycle: [0-9.]+ over the trace, /ops\/cycle: /'
  262144 ops (0 floating point) in 1 bursts over 3446 cycles
  ops/cycle: 76.072 in bursts, peak 88.000 over 256 cycles to cycle 6031

  $ ./coral_peak.exe tgemm_bf16_rvv | grep -v '^  +' | sed -E 's/[0-9]+ cycles in the kernel/N cycles in the kernel/'
  C[9x32] fp32 = A[9x128] bf16 x B[128x32] bf16, tiles 9 deep, 2 columns (the last 16 wide) in 1 pass; inner loop:
  vme-core-mini-axi: output as expected, N cycles in the kernel
  $ coralview ops --platform vme-core-mini-axi --peak-window 256 tgemm_bf16_rvv.fst | grep -E 'bursts over|^ops/cycle:' | sed -E 's/ \([0-9.]+% of the trace\)//; s/^ops\/cycle: [0-9.]+ over the trace, /ops\/cycle: /'
  73728 ops (73728 floating point) in 1 bursts over 1262 cycles
  ops/cycle: 58.422 in bursts, peak 58.500 over 256 cycles to cycle 3197

  $ ./coral_peak.exe tgemm_bf16_rvv_w24 | grep -v '^  +' | sed -E 's/[0-9]+ cycles in the kernel/N cycles in the kernel/'
  C[9x24] fp32 = A[9x128] bf16 x B[128x24] bf16, tiles 9 deep, 2 columns (the last 8 wide) in 1 pass; inner loop:
  vme-core-mini-axi: output as expected, N cycles in the kernel
  $ coralview ops --platform vme-core-mini-axi --peak-window 256 tgemm_bf16_rvv_w24.fst | grep -E 'bursts over|^ops/cycle:' | sed -E 's/ \([0-9.]+% of the trace\)//; s/^ops\/cycle: [0-9.]+ over the trace, /ops\/cycle: /'
  55296 ops (55296 floating point) in 1 bursts over 1262 cycles
  ops/cycle: 43.816 in bursts, peak 43.875 over 256 cycles to cycle 2985

  $ ./coral_peak.exe tgemm_bf16_rvv_w64 | grep -v '^  +' | sed -E 's/[0-9]+ cycles in the kernel/N cycles in the kernel/'
  C[9x64] fp32 = A[9x128] bf16 x B[128x64] bf16, tiles 9 deep, 4 columns (the last 16 wide) in 2 passes; inner loop:
  vme-core-mini-axi: output as expected, N cycles in the kernel
  $ coralview ops --platform vme-core-mini-axi --peak-window 256 tgemm_bf16_rvv_w64.fst | grep -E 'bursts over|^ops/cycle:' | sed -E 's/ \([0-9.]+% of the trace\)//; s/^ops\/cycle: [0-9.]+ over the trace, /ops\/cycle: /'
  147456 ops (147456 floating point) in 1 bursts over 2682 cycles
  ops/cycle: 54.980 in bursts, peak 58.500 over 256 cycles to cycle 4771

  $ ./coral_peak.exe tgemm_f32_rvv | grep -v '^  +' | sed -E 's/[0-9]+ cycles in the kernel/N cycles in the kernel/'
  C[9x32] fp32 = A[9x64] fp32 x B[64x32] fp32, tiles 9 deep, 2 columns (the last 16 wide) in 1 pass; inner loop:
  vme-core-mini-axi: output as expected, N cycles in the kernel
  $ coralview ops --platform vme-core-mini-axi --peak-window 256 tgemm_f32_rvv.fst | grep -E 'bursts over|^ops/cycle:' | sed -E 's/ \([0-9.]+% of the trace\)//; s/^ops\/cycle: [0-9.]+ over the trace, /ops\/cycle: /'
  36864 ops (36864 floating point) in 1 bursts over 1136 cycles
  ops/cycle: 32.451 in bursts, peak 33.750 over 256 cycles to cycle 3203

  $ ./coral_peak.exe tgemm_f32_rvv_w24 | grep -v '^  +' | sed -E 's/[0-9]+ cycles in the kernel/N cycles in the kernel/'
  C[9x24] fp32 = A[9x64] fp32 x B[64x24] fp32, tiles 9 deep, 2 columns (the last 8 wide) in 1 pass; inner loop:
  vme-core-mini-axi: output as expected, N cycles in the kernel
  $ coralview ops --platform vme-core-mini-axi --peak-window 256 tgemm_f32_rvv_w24.fst | grep -E 'bursts over|^ops/cycle:' | sed -E 's/ \([0-9.]+% of the trace\)//; s/^ops\/cycle: [0-9.]+ over the trace, /ops\/cycle: /'
  27648 ops (27648 floating point) in 1 bursts over 1136 cycles
  ops/cycle: 24.338 in bursts, peak 25.312 over 256 cycles to cycle 2991

  $ ./coral_peak.exe tgemm_f32_rvv_w64 | grep -v '^  +' | sed -E 's/[0-9]+ cycles in the kernel/N cycles in the kernel/'
  C[9x64] fp32 = A[9x64] fp32 x B[64x64] fp32, tiles 9 deep, 4 columns (the last 16 wide) in 2 passes; inner loop:
  vme-core-mini-axi: output as expected, N cycles in the kernel
  $ coralview ops --platform vme-core-mini-axi --peak-window 256 tgemm_f32_rvv_w64.fst | grep -E 'bursts over|^ops/cycle:' | sed -E 's/ \([0-9.]+% of the trace\)//; s/^ops\/cycle: [0-9.]+ over the trace, /ops\/cycle: /'
  73728 ops (73728 floating point) in 1 bursts over 2426 cycles
  ops/cycle: 30.391 in bursts, peak 33.750 over 256 cycles to cycle 4773
