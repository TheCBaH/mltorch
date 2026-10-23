Matrix multiplies far larger than DTCM, with A, B and C in SRAM and the DMA
feeding DTCM: B resident in DTCM, A streamed a block of rows at a time into a
two-block ring, one chain a step started before the block and waited for after
it. B arrives in a prologue, by the DMA alone (b-by-dma) or a quarter by the DMA
and the rest by the core's RVV copy beside it (b-shared). C's rows go out
through a second ring by the DMA (c-by-dma) or by the block's own stores
straight to SRAM (c-by-core). C is checked by hash against the host's product;
the cycles are the whole call's, prologue included, and the waits the cycles
the core spent polling the DMA: the prologue's, then the rest.

Zvt FP32 on the 1 MiB-DTCM SoC: four bytes an input element as well as a
result, so the same moves serve a quarter of INT8's operations and the engine
waits for the bus whatever does the moving.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem gemm zvt-f32 b-by-dma c-by-dma 8 128 256
  zvt f32 resident-b b-by-dma c-by-dma M=256 N=128 K=256: C as expected, 512102 cycles, 32.8 ops/cycle; 143380 waiting for B and the first block, 158050 for the rest
  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem gemm zvt-f32 b-shared c-by-core 8 128 256
  zvt f32 resident-b b-shared c-by-core M=256 N=128 K=256: C as expected, 331410 cycles, 50.6 ops/cycle; 19747 waiting for B and the first block, 19992 for the rest
