Matrix multiplies far larger than DTCM, with A, B and C in SRAM and the DMA
feeding DTCM: B resident in DTCM, A streamed a block of rows at a time into a
two-block ring, one chain a step started before the block and waited for after
it. B arrives in a prologue, by the DMA alone (b-by-dma) or a quarter by the DMA
and the rest by the core's RVV copy beside it (b-shared). C's rows go out
through a second ring by the DMA (c-by-dma) or by the block's own stores
straight to SRAM (c-by-core). C is checked by hash against the host's product;
the cycles are the whole call's, prologue included, and the waits the cycles
the core spent polling the DMA: the prologue's, then the rest.

The Zvt engine consumes operands four to eight times faster than RVV, so what
crosses the bus sets its pace. With only 32 KiB of DTCM the panels stay small,
and INT8 with everything moved by the DMA does no better than RVV; letting the
core write C helps a little.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal gemm zvt-i8 b-by-dma c-by-dma 4 32 128
  zvt i8 resident-b b-by-dma c-by-dma M=128 N=32 K=128: C as expected, 33768 cycles, 31.1 ops/cycle; 7188 waiting for B and the first block, 21516 for the rest
  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal gemm zvt-i8 b-shared c-by-core 4 32 128
  zvt i8 resident-b b-shared c-by-core M=128 N=32 K=128: C as expected, 18702 cycles, 56.1 ops/cycle; 3339 waiting for B and the first block, 4200 for the rest
