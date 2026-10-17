Typed binary32 kernels through the Coral runtime on each baseline simulator,
then guests that fail one way each.

  $ ./coral_kernels.exe "$CORALNPU_DATA"
  core-mini-axi 8 floats: dtcm dtcm dtcm
  core-mini-axi 8: status 7, negated, minstret 133, mcycle counted
  core-mini-axi 16384 floats: testbench-ram(staged) testbench-ram(staged) testbench-ram(staged)
  core-mini-axi 16384: status 7, negated, minstret 213021, mcycle counted
  core-mini-highmem-axi 8 floats: dtcm dtcm dtcm
  core-mini-highmem-axi 8: status 7, negated, minstret 133, mcycle counted
  core-mini-highmem-axi 16384 floats: dtcm dtcm dtcm
  core-mini-highmem-axi 16384: status 7, negated, minstret 213021, mcycle counted
  mpact-v2 8 floats: dtcm dtcm dtcm
  mpact-v2 8: status 7, negated, minstret 133, mcycle counted
  mpact-v2 16384 floats: extmem(staged) extmem(staged) extmem(staged)
  mpact-v2 16384: status 7, negated, minstret 213021, mcycle counted
  core-mini-axi 8 floats in place: dtcm
  core-mini-axi 8 in place: status 7, negated, minstret 72, mcycle counted
  core-mini-axi 16384 floats in place: testbench-ram(staged)
  core-mini-axi 16384 in place: status 7, negated, minstret 98328, mcycle counted
  core-mini-highmem-axi 8 floats in place: dtcm
  core-mini-highmem-axi 8 in place: status 7, negated, minstret 72, mcycle counted
  core-mini-highmem-axi 16384 floats in place: dtcm
  core-mini-highmem-axi 16384 in place: status 7, negated, minstret 98328, mcycle counted
  mpact-v2 8 floats in place: dtcm
  mpact-v2 8 in place: status 7, negated, minstret 72, mcycle counted
  mpact-v2 16384 floats in place: extmem(staged)
  mpact-v2 16384 in place: status 7, negated, minstret 98328, mcycle counted
  core-mini-axi illegal instruction: guest trap: mcause 2 mepc 0x000005e0 mtval 0x00000000
  core-mini-axi store to ITCM: guest trap: mcause 7 mepc 0x000005e0 mtval 0x00000010
  core-mini-axi spin: deadline passed
  mpact-v2 illegal instruction: guest trap: mcause 2 mepc 0x000005e0 mtval 0x00000000
  mpact-v2 store to ITCM: guest trap: mcause 7 mepc 0x000005e0 mtval 0x00000100
  mpact-v2 spin: deadline passed
  core-mini-axi cycle limit: the simulator reached its cycle limit
  mpact-v2 wrong digest: the guest refused its input: layout digest
  missing simulator: no simulator at /nonexistent/bin/coralnpu_v2_sim
