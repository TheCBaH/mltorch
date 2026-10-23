Matrix multiplies far larger than DTCM, with A, B and C in SRAM and the DMA
feeding DTCM: B resident in DTCM, A streamed a block of rows at a time into a
two-block ring, one chain a step started before the block and waited for after
it. B arrives in a prologue, by the DMA alone (b-by-dma) or a quarter by the DMA
and the rest by the core's RVV copy beside it (b-shared). C's rows go out
through a second ring by the DMA (c-by-dma) or by the block's own stores
straight to SRAM (c-by-core). C is checked by hash against the host's product;
the cycles are the whole call's, prologue included, and the waits the cycles
the core spent polling the DMA: the prologue's, then the rest.

RVV BF16 x BF16 -> FP32 and FP32 on the 32 KiB-DTCM vector SoC: three rows of
32 lanes a block, at 94-97% of their DTCM-resident rates (12.6 and 9.9), and
about three times the same kernels with every operand left in SRAM.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-vector-normal gemm rvv-bf16 b-shared c-by-dma 64 32 256
  rvv bf16 resident-b b-shared c-by-dma M=192 N=32 K=256: C as expected, 258608 cycles, 12.2 ops/cycle; 259 waiting for B and the first block, 1636 for the rest
  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-vector-normal direct rvv-bf16 64 32 256
  rvv bf16 M=192 N=32 K=256, operands in SRAM, no DMA: C as expected, 715813 cycles, 4.4 ops/cycle
  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-vector-normal gemm rvv-f32 b-shared c-by-dma 64 32 128
  rvv f32 resident-b b-shared c-by-dma M=192 N=32 K=128: C as expected, 168368 cycles, 9.3 ops/cycle; 259 waiting for B and the first block, 1636 for the rest
  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-vector-normal direct rvv-f32 64 32 128
  rvv f32 M=192 N=32 K=128, operands in SRAM, no DMA: C as expected, 527868 cycles, 3.0 ops/cycle
