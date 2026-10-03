Weights read from the pinned Hub checkpoint are byte-equal to the .pt2's, for a
CNN and for the model with Region-authored ops. Gated on PT2_DATA (and a
checkpoint in the huggingface_hub cache, fetched on first use); run via
`make pt2.runtest`.

  $ ./pt2_safetensors_equal.exe "$PT2_DATA/mobilenetv2_050/mobilenetv2_050.pt2" "$PT2_SAFETENSORS_MODELS/mobilenetv2_050"
  314 captured tensors byte-equal

  $ ./pt2_safetensors_equal.exe "$PT2_DATA/fastvit_sa12/fastvit_sa12.pt2" "$PT2_SAFETENSORS_MODELS/fastvit_sa12"
  504 captured tensors byte-equal
