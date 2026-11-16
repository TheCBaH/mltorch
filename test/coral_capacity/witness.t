Both witness images placed by the real layout on each baseline configuration.

  $ ./witness_placed.exe "$CORALNPU_DATA"
  core-mini-axi
    mobilenetv2_050 pointwise 8 -> 48 at 112x112
      input    0x20000000    401408 testbench-ram
      output   0x20062000   2408448 testbench-ram
      scratch  0x202ae000     43008 testbench-ram
    efficientnet_b0 expand 16 -> 96, depthwise 3x3 stride 2
      input    0x20000000    802816 testbench-ram
      output   0x200c4000   1204224 testbench-ram
      band     0x201ea000    172032 testbench-ram
      scratch  0x20214000     86016 testbench-ram
  core-mini-highmem-axi
    mobilenetv2_050 pointwise 8 -> 48 at 112x112
      input    0x001011c0    401408 dtcm
      output   0x20000000   2408448 testbench-ram
      scratch  0x001631c0     43008 dtcm
    efficientnet_b0 expand 16 -> 96, depthwise 3x3 stride 2
      input    0x001011c0    802816 dtcm
      output   0x20000000   1204224 testbench-ram
      band     0x001c51c0    172032 dtcm
      scratch  0x20126000     86016 testbench-ram
  mpact-v2
    mobilenetv2_050 pointwise 8 -> 48 at 112x112
      input    0x20000000    401408 extmem
      output   0x20062000   2408448 extmem
      scratch  0x202ae000     43008 extmem
    efficientnet_b0 expand 16 -> 96, depthwise 3x3 stride 2
      input    0x20000000    802816 extmem
      output   0x200c4000   1204224 extmem
      band     0x201ea000    172032 extmem
      scratch  0x20214000     86016 extmem
