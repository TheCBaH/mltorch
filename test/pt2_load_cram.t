Load real resnet18 weights and a sample image from the v0.0.4 release zip, and
bridge both into ATen tensors. Gated on PT2_DATA so the default suite needs no
download; run via `make pt2.runtest` after `make pt2.download`.

  $ ./pt2_load.exe "$PT2_DATA/resnet18/resnet18.pt2" \
  >   "$PT2_DATA/resnet18/images/000000000149.pt"
  weights: 122
  conv1.weight: float32[64; 3; 7; 7]
  conv1.weight (aten): [64; 3; 7; 7] contiguous=true
  image: float32[1; 3; 224; 224] contiguous=false
  image (aten): [1; 3; 224; 224] contiguous=false
