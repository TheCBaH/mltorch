Symbolically verify ResNet-18's structural rewrite: every pass's mapping is
checked against the state it came from, WITHOUT payloads for the graph inputs,
so a proved cluster is a statement about every input rather than about the one
tensor a numeric run happens to use. See .ai/native_transform_verify.md. Gated
on PT2_DATA; run with `make pt2.runtest` after `make pt2.download-cram`.

This is the companion to native_transform_cram.t, which pins the resulting
graph. Here the graph is dropped and only the verification is kept — the two
would otherwise duplicate a 300-line dump.

Results are per pass and per group, because a flat count over ~1600 clusters
says nothing about which part of the model was covered. The groups are the PT2
call sites the importer recorded, so they name what a reader recognises.

Read `unproved (too large)` as "not looked at": a whole activation tensor is
refused by the coordinate budget before any expansion, which is what stops the
verifier being pointed at a real model and hanging. The clusters that DO get
checked here are the layout-shaped ones the permute passes actually rewrite,
which is exactly where these passes could be wrong.

  $ ../bin/native_graph.exe transform --verify-symbolic quick \
  >   --pt2 "$PT2_DATA/resnet18/resnet18.pt2" | sed '/^graph$/,$d'
  nodes: 174 -> 91
  constants: 102, of which 0 folded
  symbolic verification: reshape_to_permute
    (root)
         83  proved (structural) [sampled 4]
         48  unproved (too large)
         20  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         60  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
          1  unproved (too large)
    torch.ops.aten.convolution.default
         80  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: chain_permute
    (root)
         83  proved (structural) [sampled 4]
         68  unproved (too large)
         21  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         40  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: trim_permute
    (root)
         83  proved (structural) [sampled 4]
         48  unproved (too large)
    torch.ops.aten._native_batch_norm_legit_no_training.default
         40  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: sink_permute
    (root)
         83  proved (structural) [sampled 4]
         48  unproved (too large)
         27  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         25  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: sink_permute
    (root)
         83  proved (structural) [sampled 4]
         57  unproved (too large)
          6  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         25  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: reuse_permute
    (root)
         83  proved (structural) [sampled 4]
         60  unproved (too large)
         10  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: sink_permute
    (root)
         83  proved (structural) [sampled 4]
         60  unproved (too large)
         10  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         60  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          4  unproved (too large)
  symbolic verification: bypass_permute
    (root)
         83  proved (structural) [sampled 4]
         49  unproved (too large)
         17  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         41  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          2  unproved (too large)
  symbolic verification: sink_permute_mean
    (root)
         83  proved (structural) [sampled 4]
         48  unproved (too large)
          2  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         41  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          2  unproved (too large)
  symbolic verification: bypass_permute
    (root)
         81  proved (structural) [sampled 4]
          1  unproved (over max_rounds) [sampled 4]
         48  unproved (too large)
          1  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  unproved (over max_nodes) [sampled 4]
    torch.ops.aten.convolution.default
         41  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          2  unproved (too large)
  symbolic verification: total
      837  proved (structural) [sampled 4]
        1  unproved (over max_nodes) [sampled 4]
        1  unproved (over max_rounds) [sampled 4]
     1422  unproved (too large)
      114  vacuous
