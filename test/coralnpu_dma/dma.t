Raw transfer speed on the subsystem (subsystem-matrix-normal; the other three
variants give the same cycles): the DMA engine against the core's own loads and
stores. Every case is timed inside its kernel by mcycle, around the transfer
(DMA: from the descriptor address written to the done bit seen); a sweep is one
simulator run, its timings read back over the SPI bridge and its destination
checked by hash. Bytes a cycle are the case's bytes over its cycles.

The DMA engine at 16-byte beats over each path, from one beat up. Its start
(the write of DESC_ADDR and CTRL to the done bit seen) costs about 57 cycles;
past that, a beat takes about 14 cycles SRAM to DTCM or back, 9 SRAM to SRAM
and 19 DTCM to DTCM, the engine moving one beat at a time, read then written.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal sram-dtcm dma16 16,64,256,1024,4096,16384 dtcm-sram dma16 16,256,4096,16384 sram-sram dma16 16,256,4096,16384 dtcm-dtcm dma16 16,256,4096,8192
  sram->dtcm dma/16 16: 71 cycles, 0.23 B/cycle
  sram->dtcm dma/16 64: 113 cycles, 0.57 B/cycle
  sram->dtcm dma/16 256: 281 cycles, 0.91 B/cycle
  sram->dtcm dma/16 1024: 953 cycles, 1.07 B/cycle
  sram->dtcm dma/16 4096: 3641 cycles, 1.12 B/cycle
  sram->dtcm dma/16 16384: 14393 cycles, 1.14 B/cycle
  dtcm->sram dma/16 16: 71 cycles, 0.23 B/cycle
  dtcm->sram dma/16 256: 281 cycles, 0.91 B/cycle
  dtcm->sram dma/16 4096: 3641 cycles, 1.12 B/cycle
  dtcm->sram dma/16 16384: 14393 cycles, 1.14 B/cycle
  sram->sram dma/16 16: 57 cycles, 0.28 B/cycle
  sram->sram dma/16 256: 197 cycles, 1.30 B/cycle
  sram->sram dma/16 4096: 2353 cycles, 1.74 B/cycle
  sram->sram dma/16 16384: 9269 cycles, 1.77 B/cycle
  dtcm->dtcm dma/16 16: 71 cycles, 0.23 B/cycle
  dtcm->dtcm dma/16 256: 351 cycles, 0.73 B/cycle
  dtcm->dtcm dma/16 4096: 4915 cycles, 0.83 B/cycle
  dtcm->dtcm dma/16 8192: 9773 cycles, 0.84 B/cycle

A narrower beat takes as long as a 16-byte one, so the bytes a cycle fall with
it.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal sram-dtcm dma8 4096 sram-dtcm dma4 4096 sram-dtcm dma1 1024
  sram->dtcm dma/8 4096: 7225 cycles, 0.57 B/cycle
  sram->dtcm dma/4 4096: 14393 cycles, 0.28 B/cycle
  sram->dtcm dma/1 1024: 14393 cycles, 0.07 B/cycle

The same 4096 bytes as a chain of shorter descriptors: each descriptor after
the first adds about 8 cycles (its two 16-byte fetches), so even a chain of
row-sized pieces keeps most of the rate.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal sram-dtcm dma16 4096/4096,4096/1024,4096/256,4096/64,4096/16
  sram->dtcm dma/16 4096: 3641 cycles, 1.12 B/cycle
  sram->dtcm dma/16 4096/1024: 3655 cycles, 1.12 B/cycle
  sram->dtcm dma/16 4096/256: 3753 cycles, 1.09 B/cycle
  sram->dtcm dma/16 4096/64: 4145 cycles, 0.99 B/cycle
  sram->dtcm dma/16 4096/16: 5671 cycles, 0.72 B/cycle

The core's own copy with RVV (vle8.v then vse8.v at e8, two register groups a
trip) moves more than twice the DMA's bytes a cycle between SRAM and DTCM;
LMUL=8 is needed for it. DTCM to DTCM it is seven times the DMA.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal sram-dtcm rvv-copy-m1 4096 sram-dtcm rvv-copy-m2 4096 sram-dtcm rvv-copy-m4 4096 sram-dtcm rvv-copy-m8 256,1024,4096,16384 dtcm-sram rvv-copy-m8 256,1024,4096,16384 sram-sram rvv-copy-m8 256,4096,16384 dtcm-dtcm rvv-copy-m8 256,1024,4096,8192
  sram->dtcm rvv-copy/m1 4096: 2817 cycles, 1.45 B/cycle
  sram->dtcm rvv-copy/m2 4096: 2175 cycles, 1.88 B/cycle
  sram->dtcm rvv-copy/m4 4096: 1851 cycles, 2.21 B/cycle
  sram->dtcm rvv-copy/m8 256: 112 cycles, 2.29 B/cycle
  sram->dtcm rvv-copy/m8 1024: 411 cycles, 2.49 B/cycle
  sram->dtcm rvv-copy/m8 4096: 1683 cycles, 2.43 B/cycle
  sram->dtcm rvv-copy/m8 16384: 6771 cycles, 2.42 B/cycle
  dtcm->sram rvv-copy/m8 256: 128 cycles, 2.00 B/cycle
  dtcm->sram rvv-copy/m8 1024: 395 cycles, 2.59 B/cycle
  dtcm->sram rvv-copy/m8 4096: 1859 cycles, 2.20 B/cycle
  dtcm->sram rvv-copy/m8 16384: 7715 cycles, 2.12 B/cycle
  sram->sram rvv-copy/m8 256: 192 cycles, 1.33 B/cycle
  sram->sram rvv-copy/m8 4096: 2883 cycles, 1.42 B/cycle
  sram->sram rvv-copy/m8 16384: 11811 cycles, 1.39 B/cycle
  dtcm->dtcm rvv-copy/m8 256: 48 cycles, 5.33 B/cycle
  dtcm->dtcm rvv-copy/m8 1024: 155 cycles, 6.61 B/cycle
  dtcm->dtcm rvv-copy/m8 4096: 659 cycles, 6.22 B/cycle
  dtcm->dtcm rvv-copy/m8 8192: 1331 cycles, 6.15 B/cycle

Reads and writes alone: the core reads SRAM at about 3 bytes a cycle and DTCM
at 12.6, and writes either at about 12 until its store buffer fills (2.6 to
SRAM in the long run).

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal sram-dtcm rvv-read-m8 256,4096,16384 dtcm-dtcm rvv-read-m8 256,4096,8192 dtcm-sram rvv-write-m8 256,1024,4096,16384 dtcm-dtcm rvv-write-m8 256,1024,4096,8192
  sram->dtcm rvv-read/m8 256: 92 cycles, 2.78 B/cycle
  sram->dtcm rvv-read/m8 4096: 1352 cycles, 3.03 B/cycle
  sram->dtcm rvv-read/m8 16384: 5384 cycles, 3.04 B/cycle
  dtcm->dtcm rvv-read/m8 256: 28 cycles, 9.14 B/cycle
  dtcm->dtcm rvv-read/m8 4096: 328 cycles, 12.49 B/cycle
  dtcm->dtcm rvv-read/m8 8192: 648 cycles, 12.64 B/cycle
  dtcm->sram rvv-write/m8 256: 111 cycles, 2.31 B/cycle
  dtcm->sram rvv-write/m8 1024: 220 cycles, 4.65 B/cycle
  dtcm->sram rvv-write/m8 4096: 1444 cycles, 2.84 B/cycle
  dtcm->sram rvv-write/m8 16384: 6340 cycles, 2.58 B/cycle
  dtcm->dtcm rvv-write/m8 256: 31 cycles, 8.26 B/cycle
  dtcm->dtcm rvv-write/m8 1024: 60 cycles, 17.07 B/cycle
  dtcm->dtcm rvv-write/m8 4096: 324 cycles, 12.64 B/cycle
  dtcm->dtcm rvv-write/m8 8192: 676 cycles, 12.12 B/cycle

Scalar lw/sw, four words a trip: a fifth of RVV's rate or less.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal sram-dtcm scalar-copy 256,4096,16384 dtcm-sram scalar-copy 256,4096,16384 sram-sram scalar-copy 4096 dtcm-dtcm scalar-copy 256,4096,8192
  sram->dtcm scalar-copy 256: 582 cycles, 0.44 B/cycle
  sram->dtcm scalar-copy 4096: 9222 cycles, 0.44 B/cycle
  sram->dtcm scalar-copy 16384: 36870 cycles, 0.44 B/cycle
  dtcm->sram scalar-copy 256: 646 cycles, 0.40 B/cycle
  dtcm->sram scalar-copy 4096: 10246 cycles, 0.40 B/cycle
  dtcm->sram scalar-copy 16384: 40966 cycles, 0.40 B/cycle
  sram->sram scalar-copy 4096: 14342 cycles, 0.29 B/cycle
  dtcm->dtcm scalar-copy 256: 326 cycles, 0.79 B/cycle
  dtcm->dtcm scalar-copy 4096: 5126 cycles, 0.80 B/cycle
  dtcm->dtcm scalar-copy 8192: 10246 cycles, 0.80 B/cycle

The DMA beside the core's own work, an 8 KiB chain from SRAM into DTCM against
about as many cycles of work. ALU work and the core's SRAM reads leave both at
full speed. While the core streams DTCM at its full rate, the core is not
slowed at all, but the DMA's writes into DTCM wait: it advances at about 70%
of its own rate until the core stops.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal overlap alu 8192 1800 overlap sram-read 8192 80 overlap dtcm-read 8192 360 overlap dtcm-write 8192 360
  overlap alu 8192 x1800: work 5413 alone, 5426 beside the DMA; DMA 7219 alone, 7214 in all; 100% of the shorter hidden
  overlap sram-read 8192 x80: work 6732 alone, 6749 beside the DMA; DMA 7219 alone, 7221 in all; 100% of the shorter hidden
  overlap dtcm-read 8192 x360: work 7212 alone, 7229 beside the DMA; DMA 7219 alone, 9381 in all; 70% of the shorter hidden
  overlap dtcm-write 8192 x360: work 7896 alone, 7910 beside the DMA; DMA 7219 alone, 10093 in all; 70% of the shorter hidden
