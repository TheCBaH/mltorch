Matrix multiplies far larger than DTCM, with A, B and C in SRAM and the DMA
feeding DTCM: B resident in DTCM, A streamed a block of rows at a time into a
two-block ring, one chain a step started before the block and waited for after
it. B arrives in a prologue, by the DMA alone (b-by-dma) or a quarter by the DMA
and the rest by the core's RVV copy beside it (b-shared). C's rows go out
through a second ring by the DMA (c-by-dma) or by the block's own stores
straight to SRAM (c-by-core). C is checked by hash against the host's product;
the cycles are the whole call's, prologue included, and the waits the cycles
the core spent polling the DMA: the prologue's, then the rest.

RVV, INT8 x INT8 -> INT16, on the 32 KiB-DTCM SoC: the blocks hide the DMA
entirely, and past the prologue the kernel runs at the peak suite's rate with
every operand in DTCM (31.3). Sharing B's copy shortens the prologue; the
core's own stores of C to SRAM cost more than the DMA's copy of them, so for
RVV C is best left to the DMA.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal gemm rvv-i8 b-by-dma c-by-dma 128 64 256
  rvv i8 resident-b b-by-dma c-by-dma M=256 N=64 K=256: C as expected, 289030 cycles, 29.0 ops/cycle; 14804 waiting for B and the first block, 2804 for the rest
  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal gemm rvv-i8 b-shared c-by-dma 128 64 256
  rvv i8 resident-b b-shared c-by-dma M=256 N=64 K=256: C as expected, 279329 cycles, 30.0 ops/cycle; 35 waiting for B and the first block, 2804 for the rest
  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal gemm rvv-i8 b-shared c-by-core 128 64 256
  rvv i8 resident-b b-shared c-by-core M=256 N=64 K=256: C as expected, 289260 cycles, 29.0 ops/cycle; 35 waiting for B and the first block, 2540 for the rest
