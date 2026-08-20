The functional release ships external torch-save tensor maps.  The map loader
sorts sample keys lexically and retains the original tensor layouts.

  $ ./pt2_tensor_map_load.exe "$PT2_DATA/mobilenetv2_050/inputs.pt" | head -2
  000000000009.jpg: float32[3; 224; 224] contiguous=true
  000000000025.jpg: float32[3; 224; 224] contiguous=true

The interpreter adds the producer's missing batch view and verifies the top-5
oracle for every shipped input. Timings are deliberately stderr-only.

  $ for m in test_convnext2 mobilenetv2_050 regnetx_002 efficientnet_b0 fastvit_sa12; do
  >   ./interp_run.exe "$PT2_DATA/$m/$m.pt2" "$PT2_DATA/$m/inputs.pt" "$PT2_DATA/$m/expected.json" "$PT2_DATA/$m/outputs.pt" --cram --strict >/dev/null 2>/dev/null || exit $?
  > done
