Typed binary32 kernels through the mailbox runtime on each subsystem simulator,
slots in DTCM and in SRAM, checked by the hashes the guest leaves; then guests
that fail one way each.

  $ ./soc_kernels.exe "$CORALNPU_SUBSYSTEM_DATA"
  subsystem-matrix-highmem 8 floats: dtcm dtcm dtcm
  subsystem-matrix-highmem 8 any: status 7, negated, minstret 133, mcycle 202
  subsystem-matrix-highmem 8 floats: sram sram sram
  subsystem-matrix-highmem 8 ram: status 7, negated, minstret 133, mcycle 322
  subsystem-matrix-normal 8 floats: dtcm dtcm dtcm
  subsystem-matrix-normal 8 any: status 7, negated, minstret 133, mcycle 202
  subsystem-matrix-normal 8 floats: sram sram sram
  subsystem-matrix-normal 8 ram: status 7, negated, minstret 133, mcycle 322
  subsystem-vector-highmem 8 floats: dtcm dtcm dtcm
  subsystem-vector-highmem 8 any: status 7, negated, minstret 133, mcycle 202
  subsystem-vector-highmem 8 floats: sram sram sram
  subsystem-vector-highmem 8 ram: status 7, negated, minstret 133, mcycle 322
  subsystem-vector-normal 8 floats: dtcm dtcm dtcm
  subsystem-vector-normal 8 any: status 7, negated, minstret 133, mcycle 202
  subsystem-vector-normal 8 floats: sram sram sram
  subsystem-vector-normal 8 ram: status 7, negated, minstret 133, mcycle 322
  4096 floats: sram sram sram
  4096 ram: status 7, negated, minstret 53277, mcycle 139314
  8 against its input: status 7, WRONG, minstret 133, mcycle 202
  illegal instruction: guest trap: mcause 2, mepc 0x00000220, mtval 0x00000000
  store to ITCM: guest trap: mcause 7, mepc 0x00000220, mtval 0x00000100
  spin: the guest was not done by the deadline
  wrong digest: the guest reports layout digest d97daf51, not the image's
  missing simulator: no simulator at /nonexistent/bin/Vchip_vector_normal
