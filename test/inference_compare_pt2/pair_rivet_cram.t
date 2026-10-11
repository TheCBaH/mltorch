The Rivet AArch64 routes as variants (`VIEW:rivet` the image route,
`rivet-table` the table host, each with `-scanned` for the scanned allocator),
on a small model. Gated on PT2_DATA and an AArch64 host with Rivet installed.

Native against the image route, bit for bit, then the two mechanisms against
each other, then the four-axis evaluator against a table host with the scanned
allocator (the dialects are related by the composed maps; no adapter is needed):

  $ m="$PT2_SAFETENSORS_MODELS/test_convnext2"; i="$PT2_DATA/test_convnext2/inputs.pt"
  $ for p in "native native.canonical:rivet" "native.canonical:rivet native.canonical:rivet-table" "native4d native.canonical:rivet-table-scanned"; do
  >   TOL=0 ./fixture_pair_run.exe safetensors "$m" "$i" 0 $p 2>/dev/null | grep -E "^execution|^comparisons"
  > done
  execution=complete numerical=agreement executed=13 units
  comparisons: 13 pass, 0 mismatch, 0 other
  execution=complete numerical=agreement executed=49 units
  comparisons: 49 pass, 0 mismatch, 0 other
  execution=complete numerical=agreement executed=40 units
  comparisons: 40 pass, 0 mismatch, 0 other

A view that cannot carry the engine is invalid input (exit 2):

  $ ./fixture_pair_run.exe safetensors "$m" "$i" 0 native native4d:rivet 2>&1 | head -1
  variant "native4d:rivet": no configuration "rivet"
