Transform ResNet-18's imported graph with the rewriting framework and execute
the result, resolving PT2 provenance for the transformed graph through the
composed source-to-destination map rather than by rebuilding the sidecar.
See .ai/native_transform_design.md §10. Gated on PT2_DATA; run with
`make pt2.runtest` after `make pt2.download-cram`.

Structural only. Nothing is loaded up front, so every constant the graph reads
is fetched from the archive by resolving its destination id back to a captured
source. The node count falls because the relayout lowering emits an inverse
permute pair at each op boundary and the permute passes cancel them — one of the
three cases the design was written for.

  $ ../bin/native_graph.exe transform --pt2 "$PT2_DATA/resnet18/resnet18.pt2" --input "$PT2_DATA/resnet18/images/000000000149.pt"
  nodes: 174 -> 133
  constants: 0 computed, 102 from the archive
  derived constants: 0
  output[0] = tensor f32 [C=1000] {-0.433519, -1.53545, -1.13174, -3.36286, -2.4336, -1.20894, -0.974103, -0.616052, ...}

With `--fold`, the weights are bound up front and constant folding hoists the
permuted conv weights to load time — the motivating case of the whole framework,
which today re-permutes every OIHW weight on every inference. Those 21 weights
now have no archive path at all, because the captured bytes are the *unpermuted*
layout and handing them back would be corruption; their PT2 identity is still
recoverable, separately, through provenance.

  $ ../bin/native_graph.exe transform --fold --pt2 "$PT2_DATA/resnet18/resnet18.pt2" --input "$PT2_DATA/resnet18/images/000000000149.pt"
  nodes: 174 -> 112
  constants: 102 computed, 0 from the archive
  derived constants: 21
    t124 <- [p_conv1_weight]
    t136 <- [p_layer1_0_conv1_weight]
    t144 <- [p_layer1_0_conv2_weight]
  output[0] = tensor f32 [C=1000] {-0.433519, -1.53545, -1.13174, -3.36286, -2.4336, -1.20894, -0.974103, -0.616052, ...}

Both agree with the untransformed interpreter, which is the check that matters:
62 nodes removed and 21 weights hoisted, same numbers out.

  $ ../bin/native_graph.exe eval --pt2 "$PT2_DATA/resnet18/resnet18.pt2" --input "$PT2_DATA/resnet18/images/000000000149.pt"
  output[0] = tensor f32 [C=1000] {-0.433519, -1.53545, -1.13174, -3.36286, -2.4336, -1.20894, -0.974103, -0.616052, ...}
