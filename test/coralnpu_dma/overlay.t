Code streamed into the instruction memory by the DMA, then run, on
subsystem-matrix-normal. A resident loader owns a 2048-byte slot of zero bytes
in its text (a zero word is an illegal instruction, so a beat that did not land
traps rather than computes) and loads four overlays into it from SRAM, each
linked on its own at the slot's address: A returns 42, B calls its own body
through an address `la` makes and sums to 66, C fills the whole slot with 511
increments, and A again goes over C's longer image. A load is the cycles from
the start of the chain to the end of `fence.i`: the start costs about 57, then
a beat about 14, the rate of SRAM to DTCM.

`fence.i` is in the loader because the ISA requires it, but this core runs
without it: the same results, four cycles fewer a load, even with the wait loop
right before the slot. No run here shows its omission, so the loader's encoded
form is what pins it.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal overlay
  overlay: slot 0x2c0, 2048 bytes; images 16, 48, 2048 bytes
  load 0 (A): 42, load 74 cycles
  load 1 (B): 66, load 102 cycles
  load 2 (C): 511, load 1852 cycles
  load 3 (A): 42, load 74 cycles

The same overlays through the image builder: a caller that only `jal`s their
names, and the builder adds the slot, a stub for each name, the manager and an
SRAM blob of chains, table and images. A call costs its load (the chain, the
wait, the bss zeroing, `fence.i`) plus the overlay itself, 36 cycles when the
overlay is already resident. D counts in a bss word: a reload starts it from
zero again (the last call of it), a resident call continues it (the second). E
calls back into the resident image, a helper in the caller's module that adds
7 to its 50.

  $ ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal overlay-calls
  call 0 (ovl_a): 42, 153 cycles
  call 1 (ovl_b): 66, 231 cycles
  call 2 (ovl_c): 511, 2949 cycles
  call 3 (ovl_a): 42, 153 cycles
  call 4 (ovl_a): 42, 36 cycles
  call 5 (ovl_d): 1, 190 cycles
  call 6 (ovl_d): 2, 45 cycles
  call 7 (ovl_a): 42, 153 cycles
  call 8 (ovl_d): 1, 190 cycles
  call 9 (ovl_e): 57, 185 cycles

A run can leave its waveform: `DMA_BENCH_TRACE=FILE` has the run end the
simulator with an interrupt, which is when it writes the trace out, rather than
kill it (a killed one leaves an empty file). `coralview dma
--platform=subsystem-matrix-normal FILE` then shows each load's beats.

  $ DMA_BENCH_TRACE=trace.fst ./dma_bench.exe "$CORALNPU_SUBSYSTEM_DATA" subsystem-matrix-normal overlay > /dev/null
  $ test -s trace.fst && echo written
  written
