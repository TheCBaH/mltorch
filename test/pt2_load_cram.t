Load a functional-release external input map. Gated on PT2_DATA so the default
suite needs no download; run via `make pt2.runtest` after `make pt2.download`.

  $ ./pt2_tensor_map_load.exe "$PT2_DATA/mobilenetv2_050/inputs.pt" | head -1
  000000000009.jpg: float32[3; 224; 224] contiguous=true
