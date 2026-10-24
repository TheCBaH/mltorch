The peak suite's kernels, unchanged, on the subsystem's normal-TCM cores, with
every operand in DTCM as on the standalone cores. The cycles in the kernel
match the standalone core's within a few cycles of code placement, so the
peak suite's rates stand for the subsystem's cores.

BF16 matrix multiplies (vtfmm.alt) compute the expected results at the
standalone core's cycles since the v0.0.5 release, whose subsystem builds set
the -DZVTBF16FMM_ON define the engine's BF16 tile multiplies need (v0.0.4's
did not, and gave wrong results faster than 256 tile multiplies can run).
RVV's BF16 widening multiply-add works in both.

  $ ./soc_peak.exe "$CORALNPU_SUBSYSTEM_DATA"
  subsystem-vector-normal gemm_bf16: as expected, 5953 cycles (standalone 5953)
  subsystem-vector-normal gemm_f32: as expected, 3861 cycles (standalone 3857)
  subsystem-vector-normal gemm_i8: as expected, 4283 cycles (standalone 4283)
  subsystem-matrix-normal gemm_bf16: as expected, 5953 cycles (standalone 5953)
  subsystem-matrix-normal gemm_f32: as expected, 3861 cycles (standalone 3857)
  subsystem-matrix-normal gemm_i8: as expected, 4283 cycles (standalone 4283)
  subsystem-matrix-normal tile_bf16: as expected, 1470 cycles (standalone 1470)
  subsystem-matrix-normal tile_f32: as expected, 1469 cycles (standalone 1469)
  subsystem-matrix-normal tile_i8s: as expected, 1472 cycles (standalone 1473)
  subsystem-matrix-normal tgemm_bf16: as expected, 2439 cycles (standalone 2440)
  subsystem-matrix-normal tgemm_f32: as expected, 2439 cycles (standalone 2440)
  subsystem-matrix-normal tgemm_i8: as expected, 2439 cycles (standalone 2440)
