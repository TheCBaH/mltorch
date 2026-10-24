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
  zvt i8 resident-b b-by-dma c-by-dma M=256 N=128 K=512: C as expected, 312123 cycles, 107.5 ops/cycle; 71700 waiting for B and the first block, 128048 for the rest
  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem gemm zvt-i8 b-by-dma c-by-core 8 128 512
  zvt i8 resident-b b-by-dma c-by-core M=256 N=128 K=512: C as expected, 225176 cycles, 149.0 ops/cycle; 71700 waiting for B and the first block, 784 for the rest
  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem gemm zvt-i8 b-shared c-by-core 8 128 512
  zvt i8 resident-b b-shared c-by-core M=256 N=128 K=512: C as expected, 183697 cycles, 182.7 ops/cycle; 9891 waiting for B and the first block, 784 for the rest
  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem gemm zvt-i8 b-shared c-by-core 8 128 1024
  zvt i8 resident-b b-shared c-by-core M=256 N=128 K=1024: C as expected, 331417 cycles, 202.5 ops/cycle; 19747 waiting for B and the first block, 19992 for the rest

Splitting C's rows: the core stores the first 20 of each block's 32 straight
to SRAM, and the DMA copies the other 12 from the C ring, so neither stream
waits on the other. At N = 256 that is 256 ops a cycle, against 244.5 with the
core storing all of C.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem gemm zvt-i8 b-shared c-split-20 16 256 1024
  zvt i8 resident-b b-shared c-split-20 M=512 N=256 K=1024: C as expected, 1047424 cycles, 256.3 ops/cycle; 10787 waiting for B and the first block, 11284 for the rest

The control, A, B and C left in SRAM, no DMA: the core's loads of every slice
from SRAM, at about three bytes a cycle, against DTCM's twelve.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem direct zvt-i8 8 128 512
  zvt i8 M=256 N=128 K=512, operands in SRAM, no DMA: C as expected, 416831 cycles, 80.5 ops/cycle
