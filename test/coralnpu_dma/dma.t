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
  sram->dtcm dma/16 16: 66 cycles, 0.24 B/cycle
  sram->dtcm dma/16 64: 105 cycles, 0.61 B/cycle
  sram->dtcm dma/16 256: 261 cycles, 0.98 B/cycle
  sram->dtcm dma/16 1024: 885 cycles, 1.16 B/cycle
  sram->dtcm dma/16 4096: 3381 cycles, 1.21 B/cycle
  sram->dtcm dma/16 16384: 13365 cycles, 1.23 B/cycle
  dtcm->sram dma/16 16: 66 cycles, 0.24 B/cycle
  dtcm->sram dma/16 256: 235 cycles, 1.09 B/cycle
  dtcm->sram dma/16 4096: 3121 cycles, 1.31 B/cycle
  dtcm->sram dma/16 16384: 12338 cycles, 1.33 B/cycle
  sram->sram dma/16 16: 53 cycles, 0.30 B/cycle
  sram->sram dma/16 256: 196 cycles, 1.31 B/cycle
  sram->sram dma/16 4096: 2354 cycles, 1.74 B/cycle
  sram->sram dma/16 16384: 9270 cycles, 1.77 B/cycle
  dtcm->dtcm dma/16 16: 66 cycles, 0.24 B/cycle
  dtcm->dtcm dma/16 256: 300 cycles, 0.85 B/cycle
  dtcm->dtcm dma/16 4096: 4148 cycles, 0.99 B/cycle
  dtcm->dtcm dma/16 8192: 8243 cycles, 0.99 B/cycle

A narrower beat takes as long as a 16-byte one, so the bytes a cycle fall with
it.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal sram-dtcm dma8 4096 sram-dtcm dma4 4096 sram-dtcm dma1 1024
  sram->dtcm dma/8 4096: 6709 cycles, 0.61 B/cycle
  sram->dtcm dma/4 4096: 13365 cycles, 0.31 B/cycle
  sram->dtcm dma/1 1024: 13365 cycles, 0.08 B/cycle

The same 4096 bytes as a chain of shorter descriptors: each descriptor after
the first adds about 8 cycles (its two 16-byte fetches), so even a chain of
row-sized pieces keeps most of the rate.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal sram-dtcm dma16 4096/4096,4096/1024,4096/256,4096/64,4096/16
  sram->dtcm dma/16 4096: 3381 cycles, 1.21 B/cycle
  sram->dtcm dma/16 4096/1024: 3394 cycles, 1.21 B/cycle
  sram->dtcm dma/16 4096/256: 3498 cycles, 1.17 B/cycle
  sram->dtcm dma/16 4096/64: 3875 cycles, 1.06 B/cycle
  sram->dtcm dma/16 4096/16: 5422 cycles, 0.76 B/cycle

The core's own copy with RVV (vle8.v then vse8.v at e8, two register groups a
trip) moves more than twice the DMA's bytes a cycle between SRAM and DTCM;
LMUL=8 is needed for it. DTCM to DTCM it is seven times the DMA.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal sram-dtcm rvv-copy-m1 4096 sram-dtcm rvv-copy-m2 4096 sram-dtcm rvv-copy-m4 4096 sram-dtcm rvv-copy-m8 256,1024,4096,16384 dtcm-sram rvv-copy-m8 256,1024,4096,16384 sram-sram rvv-copy-m8 256,4096,16384 dtcm-dtcm rvv-copy-m8 256,1024,4096,8192
  sram->dtcm rvv-copy/m1 4096: 2561 cycles, 1.60 B/cycle
  sram->dtcm rvv-copy/m2 4096: 1535 cycles, 2.67 B/cycle
  sram->dtcm rvv-copy/m4 4096: 1019 cycles, 4.02 B/cycle
  sram->dtcm rvv-copy/m8 256: 54 cycles, 4.74 B/cycle
  sram->dtcm rvv-copy/m8 1024: 179 cycles, 5.72 B/cycle
  sram->dtcm rvv-copy/m8 4096: 755 cycles, 5.43 B/cycle
  sram->dtcm rvv-copy/m8 16384: 3059 cycles, 5.36 B/cycle
  dtcm->sram rvv-copy/m8 256: 56 cycles, 4.57 B/cycle
  dtcm->sram rvv-copy/m8 1024: 179 cycles, 5.72 B/cycle
  dtcm->sram rvv-copy/m8 4096: 779 cycles, 5.26 B/cycle
  dtcm->sram rvv-copy/m8 16384: 3179 cycles, 5.15 B/cycle
  sram->sram rvv-copy/m8 256: 62 cycles, 4.13 B/cycle
  sram->sram rvv-copy/m8 4096: 875 cycles, 4.68 B/cycle
  sram->sram rvv-copy/m8 16384: 3563 cycles, 4.60 B/cycle
  dtcm->dtcm rvv-copy/m8 256: 48 cycles, 5.33 B/cycle
  dtcm->dtcm rvv-copy/m8 1024: 155 cycles, 6.61 B/cycle
  dtcm->dtcm rvv-copy/m8 4096: 659 cycles, 6.22 B/cycle
  dtcm->dtcm rvv-copy/m8 8192: 1331 cycles, 6.15 B/cycle

Reads and writes alone: the core reads SRAM at about 3 bytes a cycle and DTCM
at 12.6, and writes either at about 12 until its store buffer fills (2.6 to
SRAM in the long run).

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal sram-dtcm rvv-read-m8 256,4096,16384 dtcm-dtcm rvv-read-m8 256,4096,8192 dtcm-sram rvv-write-m8 256,1024,4096,16384 dtcm-dtcm rvv-write-m8 256,1024,4096,8192
  sram->dtcm rvv-read/m8 256: 34 cycles, 7.53 B/cycle
  sram->dtcm rvv-read/m8 4096: 424 cycles, 9.66 B/cycle
  sram->dtcm rvv-read/m8 16384: 1672 cycles, 9.80 B/cycle
  dtcm->dtcm rvv-read/m8 256: 28 cycles, 9.14 B/cycle
  dtcm->dtcm rvv-read/m8 4096: 328 cycles, 12.49 B/cycle
  dtcm->dtcm rvv-read/m8 8192: 648 cycles, 12.64 B/cycle
  dtcm->sram rvv-write/m8 256: 39 cycles, 6.56 B/cycle
  dtcm->sram rvv-write/m8 1024: 76 cycles, 13.47 B/cycle
  dtcm->sram rvv-write/m8 4096: 436 cycles, 9.39 B/cycle
  dtcm->sram rvv-write/m8 16384: 1876 cycles, 8.73 B/cycle
  dtcm->dtcm rvv-write/m8 256: 31 cycles, 8.26 B/cycle
  dtcm->dtcm rvv-write/m8 1024: 60 cycles, 17.07 B/cycle
  dtcm->dtcm rvv-write/m8 4096: 324 cycles, 12.64 B/cycle
  dtcm->dtcm rvv-write/m8 8192: 676 cycles, 12.12 B/cycle

Scalar lw/sw, four words a trip: a fifth of RVV's rate or less.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal sram-dtcm scalar-copy 256,4096,16384 dtcm-sram scalar-copy 256,4096,16384 sram-sram scalar-copy 4096 dtcm-dtcm scalar-copy 256,4096,8192
  sram->dtcm scalar-copy 256: 518 cycles, 0.49 B/cycle
  sram->dtcm scalar-copy 4096: 8198 cycles, 0.50 B/cycle
  sram->dtcm scalar-copy 16384: 32774 cycles, 0.50 B/cycle
  dtcm->sram scalar-copy 256: 582 cycles, 0.44 B/cycle
  dtcm->sram scalar-copy 4096: 9222 cycles, 0.44 B/cycle
  dtcm->sram scalar-copy 16384: 36870 cycles, 0.44 B/cycle
  sram->sram scalar-copy 4096: 12294 cycles, 0.33 B/cycle
  dtcm->dtcm scalar-copy 256: 326 cycles, 0.79 B/cycle
  dtcm->dtcm scalar-copy 4096: 5126 cycles, 0.80 B/cycle
  dtcm->dtcm scalar-copy 8192: 10246 cycles, 0.80 B/cycle

The DMA beside the core's own work, an 8 KiB chain from SRAM into DTCM against
about as many cycles of work. ALU work and the core's SRAM reads leave both at
full speed. While the core streams DTCM at its full rate, the core is not
slowed at all, but the DMA's writes into DTCM wait: it advances at about 70%
of its own rate until the core stops.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal overlap alu 8192 1800 overlap sram-read 8192 80 overlap dtcm-read 8192 360 overlap dtcm-write 8192 360
  overlap alu 8192 x1800: work 5413 alone, 5424 beside the DMA; DMA 6704 alone, 6708 in all; 100% of the shorter hidden
  overlap sram-read 8192 x80: work 2092 alone, 2107 beside the DMA; DMA 6704 alone, 6719 in all; 99% of the shorter hidden
  overlap dtcm-read 8192 x360: work 7212 alone, 7625 beside the DMA; DMA 6704 alone, 7648 in all; 93% of the shorter hidden
  overlap dtcm-write 8192 x360: work 7896 alone, 8306 beside the DMA; DMA 6704 alone, 8360 in all; 93% of the shorter hidden

Descriptor polling: with poll_en the engine reads a word before every beat and
waits until it matches. A one-beat descriptor polling a flag holds the chain
(CUR_DESC stays on it through the delay) and lets it go about as fast as a
fresh start once the core writes the flag. Polling every beat costs about 4
cycles a beat on an SRAM word, 9 on a DTCM one. While the DMA spins on a flag,
in SRAM or DTCM, the core's DTCM stream loses 1%. A handshake through a flag
and a token written back (51 cycles) is slower than starting a one-beat chain
and polling STATUS (35), so a step's chain is best started by the core.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal poll
  copy alone: stream 25, flag to done 3323, CUR_DESC 20000000, STATUS 2
  gated copy, released after a delay: stream 25, flag to done 3377, CUR_DESC 20000020, STATUS 2
  copy, every beat polling SRAM: stream 25, flag to done 4337, CUR_DESC 20000060, STATUS 2
  copy, every beat polling DTCM: stream 25, flag to done 5104, CUR_DESC 20000080, STATUS 2
  core stream alone: stream 1292, flag to done 21, CUR_DESC 20000080, STATUS 1
  core stream, the DMA polling an SRAM flag: stream 1307, flag to done 3377, CUR_DESC 200000a0, STATUS 2
  core stream, the DMA polling a DTCM flag: stream 1459, flag to done 3373, CUR_DESC 200000e0, STATUS 2
  handshake: flag in SRAM, token back in DTCM: stream 25, flag to done 50, CUR_DESC 20000120, STATUS 2
  start and STATUS poll of a 16-byte copy: stream 25, flag to done 34, CUR_DESC 20000160, STATUS 2
