Resident B with A row-major in SRAM, as an activation arrives: a block's rows
are contiguous, so one descriptor still brings them, but the block reads A
packed (a k step's rows together), so the core packs each block in DTCM
first, a row at a time by a vector load and a strided store (vsse) VLMAX
elements long. The cycles are the whole call's, prologue included; the waits
the cycles the core spent polling the DMA.

RVV INT8 -> INT16 on the 32 KiB-DTCM SoC: two rows a block, and the packing
costs 5% beside the packed kernel's 279456 cycles (gemm_rvv.t).

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal gemm rvv-i8 row-major-a b-shared c-by-dma 128 64 256
  rvv i8 row-major-a b-shared c-by-dma M=256 N=64 K=256: C as expected, 294312 cycles, 28.5 ops/cycle; 35 waiting for B and the first block, 2804 for the rest

Zvt INT8 -> INT32 on the 1 MiB SoC: 32 rows a block, packed at about 0.9
bytes a cycle, as long again as the block's tile multiplies; against 183697
cycles packed (gemm_zvt_i8.t). For Zvt, A is worth packing where it is
produced.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-highmem gemm zvt-i8 row-major-a b-shared c-by-core 8 128 512
  zvt i8 row-major-a b-shared c-by-core M=256 N=128 K=512: C as expected, 328054 cycles, 102.3 ops/cycle; 9891 waiting for B and the first block, 784 for the rest
