Matrix multiplies too wide for B to stay in DTCM: A resident instead, copied
in by the prologue (a quarter by the DMA, the rest by the core beside it), and
B streamed a panel of columns a step through a two-panel ring, packed so each
panel is one descriptor. A step runs every block of A's rows against its
panel. C's rows, a panel wide, go out by the blocks' own stores straight to
SRAM (c-by-core) or staged in DTCM and copied by the next step's chain, a
descriptor a row (c-by-dma). The cycles are the whole call's, prologue
included; the waits the cycles the core spent polling the DMA.

RVV INT8 on the 32 KiB-DTCM SoC: N = 512 is eight panels of 64 lanes, and B,
64 KiB, twice the DTCM. The blocks hide B's panels, and C by the DMA beats the
core's stores to SRAM; the control, the peak kernel reading A, B and C in
SRAM, is three times slower.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal gemm rvv-i8 resident-a/64 a-shared c-by-dma 16 512 128
  rvv i8 resident-a/64 a-shared c-by-dma M=32 N=512 K=128: C as expected, 148370 cycles, 28.3 ops/cycle; 6923 waiting for A and the first panel, 4016 for the rest
  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal gemm rvv-i8 resident-a/64 a-shared c-by-core 16 512 128
  rvv i8 resident-a/64 a-shared c-by-core M=32 N=512 K=128: C as expected, 154678 cycles, 27.1 ops/cycle; 6923 waiting for A and the first panel, 140 for the rest
  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal direct rvv-i8 16 512 128
  rvv i8 M=32 N=512 K=128, operands in SRAM, no DMA: C as expected, 433983 cycles, 9.7 ops/cycle

RVV BF16 -> FP32, N = 1024 as 32 panels of 32 lanes, at the rate the peak
suite measures with every operand in DTCM (12.6).

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal gemm rvv-bf16 resident-a/32 a-shared c-by-dma 8 1024 128
  rvv bf16 resident-a/32 a-shared c-by-dma M=24 N=1024 K=128: C as expected, 518720 cycles, 12.1 ops/cycle; 6783 waiting for A and the first panel, 3530 for the rest

Zvt INT8 -> INT32 on the 32 KiB SoC: a 32-row A is 4 KiB, and B's panels
stream at the DMA's rate, which bounds the step.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal gemm zvt-i8 resident-a/32 a-shared c-by-core 1 1024 128
  zvt i8 resident-a/32 a-shared c-by-core M=32 N=1024 K=128: C as expected, 125666 cycles, 66.8 ops/cycle; 3339 waiting for A and the first panel, 43183 for the rest

Zvt INT8 on the 1 MiB SoC, with a 1 MiB B that no DTCM holds: A is 128 KiB,
and a step's multiplies and stores of C take about what the DMA needs for the
next panel.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem gemm zvt-i8 resident-a/32 a-shared c-by-core 4 1024 1024
  zvt i8 resident-a/32 a-shared c-by-core M=128 N=1024 K=1024: C as expected, 1152274 cycles, 233.0 ops/cycle; 19747 waiting for A and the first panel, 88319 for the rest
