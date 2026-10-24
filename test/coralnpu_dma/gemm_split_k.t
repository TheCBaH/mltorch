Matrix multiplies with K too deep for B to stay in DTCM: B in chunks of K
rows, two chunks in DTCM, the next arriving a share a step beside the current
chunk's steps, so no step waits for a whole chunk. A streams a block of a
chunk's columns a step; each block's partial sums come back from SRAM through
the C ring before its step (zeros for the first chunk) and go out after it.
The RVV block starts its accumulators from them by vector loads, the Zvt one
by tile loads (vtle32). The cycles are the whole call's, prologue included;
the waits the cycles the core spent polling the DMA.

RVV INT8 -> INT16 on the 32 KiB-DTCM SoC, K = 1024: B is 64 KiB, twice the
DTCM. Chunks of 128 rows keep the blocks at 92% of the peak suite's rate with
every operand in DTCM (31.3): a block's partial sums cost the DMA 512 bytes
more a step, which it hides.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal gemm rvv-i8 split-k/128 b-shared c-by-dma 64 64 1024
  rvv i8 split-k/128 b-shared c-by-dma M=128 N=64 K=1024: C as expected, 584634 cycles, 28.7 ops/cycle; 35 waiting for B's first chunk and the first block, 10484 for the rest

Zvt INT8 -> INT32 on the same SoC: the DMA carries A, B and the partial sums
twice, four bytes a result each way, and bounds every step.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal gemm zvt-i8 split-k/128 b-shared c-by-dma 8 32 1024
  zvt i8 split-k/128 b-shared c-by-dma M=256 N=32 K=1024: C as expected, 741512 cycles, 22.6 ops/cycle; 6923 waiting for B's first chunk and the first block, 630218 for the rest
