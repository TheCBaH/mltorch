Zvt BF16 (vtfmm.alt, BF16 inputs, FP32 accumulators) on the 1 MiB-DTCM matrix
SoC, which computes it since the v0.0.5 release. First the reference: the same
kernel with A, B and C all in DTCM (peak), against which the DMA-fed multiply
is measured, and FP32's at the same shape. BF16 moves two bytes an input
element, so its peak is twice FP32's tile rate, close to the 173.6 ops/cycle
of the peak suite's tgemm_bf16 bursts.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem peak zvt-f32 8 128 256 peak zvt-bf16 8 128 256 peak zvt-bf16 8 256 512
  zvt f32 M=256 N=128 K=256, operands in DTCM, no DMA: C as expected, 210351 cycles, 79.8 ops/cycle
  zvt bf16 M=256 N=128 K=256, operands in DTCM, no DMA: C as expected, 112047 cycles, 149.7 ops/cycle
  zvt bf16 M=256 N=256 K=512, operands in DTCM, no DMA: C as expected, 420655 cycles, 159.5 ops/cycle

Then B resident in DTCM and A streamed a block of rows at a time, as for FP32
and INT8 (gemm_zvt_f32.t, gemm_zvt_i8.t), with A, B and C in SRAM. At K = 256
the bytes and so the cycles are INT8's at K = 512: the bus, not the engine,
bounds the multiply, and the core's share of B's prologue and C's stores
straight to SRAM leave the DMA least to do.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem gemm zvt-bf16 b-by-dma c-by-dma 8 128 256 gemm zvt-bf16 b-shared c-by-core 8 128 256 gemm zvt-bf16 b-shared c-by-dma 8 128 256 gemm zvt-bf16 b-shared c-split-8 8 128 256 gemm zvt-bf16 b-shared c-by-core 8 256 512
  zvt bf16 resident-b b-by-dma c-by-dma M=256 N=128 K=256: C as expected, 312123 cycles, 53.8 ops/cycle; 71700 waiting for B and the first block, 128048 for the rest
  zvt bf16 resident-b b-shared c-by-core M=256 N=128 K=256: C as expected, 183697 cycles, 91.3 ops/cycle; 9891 waiting for B and the first block, 784 for the rest
  zvt bf16 resident-b b-shared c-by-dma M=256 N=128 K=256: C as expected, 270644 cycles, 62.0 ops/cycle; 9891 waiting for B and the first block, 128048 for the rest
  zvt bf16 resident-b b-shared c-split-8 M=256 N=128 K=256: C as expected, 242844 cycles, 69.1 ops/cycle; 9891 waiting for B and the first block, 90044 for the rest
  zvt bf16 resident-b b-shared c-by-core M=256 N=256 K=512: C as expected, 595153 cycles, 112.8 ops/cycle; 10787 waiting for B and the first block, 784 for the rest

The control, A, B and C left in SRAM, no DMA.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem direct zvt-bf16 8 128 256 direct zvt-bf16 8 256 512
  zvt bf16 M=256 N=128 K=256, operands in SRAM, no DMA: C as expected, 416831 cycles, 40.2 ops/cycle
  zvt bf16 M=256 N=256 K=512, operands in SRAM, no DMA: C as expected, 1554591 cycles, 43.2 ops/cycle
