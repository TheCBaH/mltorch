Matrix multiplies far larger than DTCM, with A, B and C in SRAM and the DMA
feeding DTCM: B resident in DTCM, A streamed a block of rows at a time into a
two-block ring, one chain a step started before the block and waited for after
it. B arrives in a prologue, by the DMA alone (b-by-dma) or a quarter by the DMA
and the rest by the core's RVV copy beside it (b-shared). C's rows go out
through a second ring by the DMA (c-by-dma) or by the block's own stores
straight to SRAM (c-by-core). C is checked by hash against the host's product;
the cycles are the whole call's, prologue included, and the waits the cycles
the core spent polling the DMA: the prologue's, then the rest.

Zvt INT8 -> INT32 on the 1 MiB-DTCM SoC. With the DMA alone it moves A, B and
C (four bytes a result) at about 1.1 bytes a cycle and waits for it. The core
writing C takes the largest stream off the DMA, and the shared prologue the
longest wait; then each block costs about what the DMA takes for A and the
core for the tiles and C's stores, and a deeper K gives each byte more work.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem gemm zvt-i8 b-by-dma c-by-dma 8 128 512
  zvt i8 resident-b b-by-dma c-by-dma M=256 N=128 K=512: C as expected, 312116 cycles, 107.5 ops/cycle; 71700 waiting for B and the first block, 128048 for the rest
  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem gemm zvt-i8 b-by-dma c-by-core 8 128 512
  zvt i8 resident-b b-by-dma c-by-core M=256 N=128 K=512: C as expected, 225169 cycles, 149.0 ops/cycle; 71700 waiting for B and the first block, 784 for the rest
  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem gemm zvt-i8 b-shared c-by-core 8 128 512
  zvt i8 resident-b b-shared c-by-core M=256 N=128 K=512: C as expected, 183690 cycles, 182.7 ops/cycle; 9891 waiting for B and the first block, 784 for the rest
  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem gemm zvt-i8 b-shared c-by-core 8 128 1024
  zvt i8 resident-b b-shared c-by-core M=256 N=128 K=1024: C as expected, 331410 cycles, 202.5 ops/cycle; 19747 waiting for B and the first block, 19992 for the rest
