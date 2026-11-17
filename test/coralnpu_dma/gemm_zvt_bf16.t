Zvt BF16 (vtfmm.alt, BF16 inputs, FP32 accumulators) on the 1 MiB-DTCM matrix
SoC, which computes it since the v0.0.5 release. First the reference: the same
kernel with A, B and C all in DTCM (peak), against which the DMA-fed multiply
is measured, and FP32's at the same shape. BF16 moves two bytes an input
element, so its peak is twice FP32's tile rate, close to the 173.6 ops/cycle
of the peak suite's tgemm_bf16 bursts.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem peak zvt-f32 8 128 256 peak zvt-bf16 8 128 256 peak zvt-bf16 8 256 512
  zvt f32 M=256 N=128 K=256, operands in DTCM, no DMA: C as expected, 210349 cycles, 79.8 ops/cycle
  zvt bf16 M=256 N=128 K=256, operands in DTCM, no DMA: C as expected, 112045 cycles, 149.7 ops/cycle
  zvt bf16 M=256 N=256 K=512, operands in DTCM, no DMA: C as expected, 420653 cycles, 159.5 ops/cycle

Then B resident in DTCM and A streamed a block of rows at a time, as for FP32
and INT8 (gemm_zvt_f32.t, gemm_zvt_i8.t), with A, B and C in SRAM. At K = 256
the bytes and so the cycles are INT8's at K = 512: the bus, not the engine,
bounds the multiply, and the core's share of B's prologue and C's stores
straight to SRAM leave the DMA least to do.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem gemm zvt-bf16 b-by-dma c-by-dma 8 128 256 gemm zvt-bf16 b-shared c-by-core 8 128 256 gemm zvt-bf16 b-shared c-by-dma 8 128 256 gemm zvt-bf16 b-shared c-split-8 8 128 256 gemm zvt-bf16 b-shared c-by-core 8 256 512
  zvt bf16 resident-b b-by-dma c-by-dma M=256 N=128 K=256: C as expected, 262233 cycles, 64.0 ops/cycle; 66579 waiting for B and the first block, 81381 for the rest
  zvt bf16 resident-b b-shared c-by-core M=256 N=128 K=256: C as expected, 153892 cycles, 109.0 ops/cycle; 17507 waiting for B and the first block, 329 for the rest
  zvt bf16 resident-b b-shared c-by-dma M=256 N=128 K=256: C as expected, 222675 cycles, 75.3 ops/cycle; 17507 waiting for B and the first block, 81381 for the rest
  zvt bf16 resident-b b-shared c-split-8 M=256 N=128 K=256: C as expected, 202010 cycles, 83.1 ops/cycle; 17507 waiting for B and the first block, 57388 for the rest
  zvt bf16 resident-b b-shared c-by-core M=256 N=256 K=512: C as expected, 532196 cycles, 126.1 ops/cycle; 43299 waiting for B and the first block, 329 for the rest

The control, A, B and C left in SRAM, no DMA.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem direct zvt-bf16 8 128 256 direct zvt-bf16 8 256 512
  zvt bf16 M=256 N=128 K=256, operands in SRAM, no DMA: C as expected, 169501 cycles, 99.0 ops/cycle
  zvt bf16 M=256 N=256 K=512, operands in SRAM, no DMA: C as expected, 633885 cycles, 105.9 ops/cycle
