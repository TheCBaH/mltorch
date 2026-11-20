native_graph lower: what each backend family makes of one node, over the
committed mobilenetv2_050 model.json. UNGATED: the file is a submodule
checkout, not a downloaded weight. Node n0 is the first canonical node,
spelled as the canonical view prints it.

The overview: the invocation and each family's default, emitting or refusing
with a typed reason.

  $ ../bin/native_graph.exe lower --model model.json --node n0 --list
  node n0: 1 outputs, 2 edges, 0 synthetics
  asm        emits 144 lines
  c          emits 15 lines
  coral      emits 4 lines
  js         emits 9 lines
  loop       emits 5 lines
  machine_ir emits 117 lines
  ssa        emits 18 lines
  wat        emits 61 lines

Each supported family's default, the head of its text.

  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact loop | head -3
  native/n0/loop
  in t314 f32 [H=3 W=224 C=224]
  out t315 f32 [H=224 W=224 C=3]
  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact js | head -3
  native/n0/js
  "use strict";
  function loop_kernel(b0, b1) {
  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact c | head -2
  native/n0/c
  static int kernel(struct model_error *err, double *local, const float *b0, float *b1) {
  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact wat | head -3
  native/n0/wat
  (module
    (memory 1)
  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact ssa | head -2
  native/n0/ssa
  buffer b314 in f32 [1, 1, 1, 3, 1, 50176]

Machine IR: each target's realized stage (the head), the typed refusal for register
groups.

  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact machine_ir:x86_64/realized | head -3
  native/n0/machine_ir
  target x86_64 [sse2]
  fn0 main func kernel() -> (eax) frame 168 {
  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact machine_ir:aarch64/realized | head -3
  native/n0/machine_ir:aarch64
  target aarch64 [fp]
  fn0 main func kernel() -> (w0) frame 176 {
  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact machine_ir:riscv32/realized | head -3
  native/n0/machine_ir:riscv32
  target riscv32 []
  fn0 main func kernel() -> (a0) frame 128 {
  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact machine_ir:riscv32_grouped/realized
  refused: the reference allocator does not allocate register groups (realized)
  [1]

The raw Loop program is not the optimized one.

  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact loop:raw | wc -l
  10
  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact loop | wc -l
  6

--compare prints a unified diff of the two texts: the Loop JavaScript and the
structured-SSA JavaScript differ, and a text against itself is identical.

  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact js:ssa --compare js | head -4
  --- js:ssa
  +++ js
  @@ -1,26 +1,8 @@
   "use strict";
  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact js --compare js
  identical

Assembly: the typed Rivet module of each target's published artifact, printed as
GNU assembly (the head, then the notes).

  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact asm | tr "\t" "~" | head -4
  native/n0/asm
  ~.section~.text,"ax",@progbits
  ~.balign~16
  ~.globl~kernel
  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact asm | grep -c "^note: "
  3
  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact asm:aarch64 | tr "\t" "~" | sed -n 2,3p
  ~.section~.text,"ax",%progbits
  ~.balign~4
  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact asm:riscv32 | tr "\t" "~" | sed -n 2,3p
  ~.section~.text,"ax",@progbits
  ~.balign~4
  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact asm:riscv32_grouped | tr "\t" "~" | sed -n 2,3p
  ~.section~.text,"ax",@progbits
  ~.balign~4
  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact asm | grep "^note: code"
  note: code: 522 bytes

Coral: the estimate of the default, and its kernels. Node n0 is a convolution
the plan keeps as loops, so it has a count and no kernel; n424 is a matrix
product, which has a transfer plan and an RVV and a Zvt kernel (tabs shown as ~).

  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact coral | tail -2
  estimated cycles: 1757860
  note: every buffer assumed in dtcm

  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact coral:rvv_kernel
  refused: the invocation's plan holds no matrix product
  [1]

  $ ../bin/native_graph.exe lower --model model.json --node n424 --artifact coral | tail -3
  contract contract.f32.b1.m3136.n16.k48.a0_48_1.b0_48_1.c50176_16_1.zero.ordered (dtcm dtcm dtcm) x1
  estimated cycles: 1218128
  note: every buffer assumed in dtcm

  $ ../bin/native_graph.exe lower --model model.json --node n424 --artifact coral:transfer_plan
  native/n424/coral:transfer_plan
  A streamed by 58 rows, B shared, C by the DMA
  predicted cycles: 2376493
  note: DTCM budget: 32768 bytes

  $ ../bin/native_graph.exe lower --model model.json --node n424 --artifact coral:rvv_kernel | tr "\t" "~" | sed -n 2,5p
  ~.section~.text,"ax",@progbits
  ~.balign~4
  ~.globl~mlt_contract_40a4ee6b00001da1
  mlt_contract_40a4ee6b00001da1:

  $ ../bin/native_graph.exe lower --model model.json --node n424 --artifact coral:zvt_kernel | tr "\t" "~" | sed -n 2,5p
  ~.section~.text,"ax",@progbits
  ~.balign~4
  ~.globl~mlt_contract_40a4ee6b00001da1
  mlt_contract_40a4ee6b00001da1:

An unknown family or variant is a usage error.

  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact bogus 2>&1 | grep unknown
  native_graph: option --artifact: unknown family "bogus"
  $ ../bin/native_graph.exe lower --model model.json --node n0 --artifact loop:bogus 2>&1 | grep unknown
  native_graph: option --artifact: loop: unknown segment "bogus"

An unknown node is an error naming it.

  $ ../bin/native_graph.exe lower --model model.json --node n999999 --artifact js
  native_graph: no invocation for node n999999
  [123]
  $ ../bin/native_graph.exe lower --model model.json --node x --artifact js 2>&1 | grep "not a node"
  native_graph: option --node: "x" is not a node id (expected nN)
