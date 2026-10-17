The pinned emulators release decodes; each baseline simulator matches its
manifest hash.

  $ ./coral_manifest_check.exe "$CORALNPU_DATA"
  10 configurations, every ID recomputed
  core-mini-axi rtl coralnpu:core-mini-axi:0ab498c0b2c3bcb59e0f
  f,i,m,zbb,zfbfmin,zicsr,zifencei
  itcm           itcm 0x00000000     8192 loadable
  dtcm           dtcm 0x00010000    32768 writable loadable htif
  core-csr       csr  0x00030000     4096
  testbench-ram  ram  0x20000000  4194304 writable
  testbench-uart mmio 0x54000000        4 writable
  $DIR/bin/core_mini_axi_sim --binary=guest.elf
  core-mini-highmem-axi rtl coralnpu:core-mini-highmem-axi:32e8cd735a35ffc9a9b1
  f,i,m,zbb,zfbfmin,zicsr,zifencei
  itcm           itcm 0x00000000  1048576 loadable
  dtcm           dtcm 0x00100000  1048576 writable loadable htif
  core-csr       csr  0x00200000     4096
  testbench-ram  ram  0x20000000  4194304 writable
  testbench-uart mmio 0x54000000        4 writable
  $DIR/bin/core_mini_highmem_axi_sim --binary=guest.elf
  mpact-v2 mpact coralnpu:mpact-v2:d66941535b7130e0ccc5
  f,i,m,zbb,zicsr,zifencei,zve32x
  itcm           itcm 0x00000000     8192 loadable
  dtcm           dtcm 0x00010000    32768 writable loadable htif
  extmem         ram  0x20000000  4194304 writable loadable
  $DIR/bin/coralnpu_v2_sim --semihost_htif --exit_on_ebreak guest.elf
