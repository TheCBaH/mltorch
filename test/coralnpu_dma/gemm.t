Matrix multiplies far larger than DTCM, with A, B and C in SRAM and the DMA
feeding DTCM: B resident (one prologue chain), A streamed a block of rows at a
time into a two-block ring, C's rows staged in a second ring and copied out,
one chain a step started before the block and waited for after it. C is
checked by hash against the host's product; the cycles are the whole call's,
prologue included. The waits are the cycles the core spent polling the DMA:
the prologue's (all of B and A's first block), then the rest.

RVV, INT8 x INT8 -> INT16, on the 32 KiB-DTCM SoC: the blocks hide the DMA
entirely, and past the prologue the kernel runs at the peak suite's rate with
every operand in DTCM (31.3).

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal gemm rvv-i8 128 64 256
  rvv i8 resident-b M=256 N=64 K=256: C as expected, 289036 cycles, 29.0 ops/cycle; 14818 waiting for B and the first block, 2804 for the rest

The Zvt engine moves its operands four to eight times faster than RVV, so the
DMA sets its pace: every matrix crosses the bus once, A (M K), B (K W) and C
(4 M W for INT32), about 1.1 bytes a cycle. With only 32 KiB of DTCM the
panels stay small and INT8 does no better than RVV.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal gemm zvt-i8 4 32 128
  zvt i8 resident-b M=128 N=32 K=128: C as expected, 33774 cycles, 31.0 ops/cycle; 7202 waiting for B and the first block, 21516 for the rest

With 1 MiB of DTCM, B's panel and the rings grow, and so do the operations a
byte serves: INT8 at 3.4 times RVV's rate. FP32 moves four bytes an element
and is held to a third of its engine's.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem gemm zvt-i8 8 128 512
  zvt i8 resident-b M=256 N=128 K=512: C as expected, 312122 cycles, 107.5 ops/cycle; 71714 waiting for B and the first block, 128048 for the rest
  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem gemm zvt-f32 8 128 256
  zvt f32 resident-b M=256 N=128 K=256: C as expected, 512108 cycles, 32.8 ops/cycle; 143394 waiting for B and the first block, 158050 for the rest
