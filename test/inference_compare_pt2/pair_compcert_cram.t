The embedded CompCert as a variant (`VIEW:compcert`). Gated on PT2_DATA, an
AArch64 host and a build with MLTORCH_COMPCERT=aarch64. CompCert against the
Rivet image route, whose assembler, image planner and loader they share:

  $ m="$PT2_SAFETENSORS_MODELS/test_convnext2"; i="$PT2_DATA/test_convnext2/inputs.pt"
  $ TOL=0 ./fixture_pair_run.exe safetensors "$m" "$i" 0 native.canonical:compcert native.canonical:rivet 2>/dev/null | grep -E "^execution|^comparisons"
  execution=complete numerical=agreement executed=49 units
  comparisons: 49 pass, 0 mismatch, 0 other
  $ TOL=0 ./fixture_pair_run.exe safetensors "$m" "$i" 0 native native.canonical:compcert 2>/dev/null | grep -E "^execution|^comparisons"
  execution=complete numerical=agreement executed=13 units
  comparisons: 13 pass, 0 mismatch, 0 other
