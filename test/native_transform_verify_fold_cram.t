The same symbolic verification with `--fold`, so constant folding and batch-norm
folding also run. Gated on PT2_DATA; run with `make pt2.runtest` after
`make pt2.download-cram`.

Folding is what makes the constant-shaped clusters checkable, and the contrast
with the structural run is the point of having both: with payloads bound, a fold
can be compared against the number the pass computed, and those clusters come
back `proved (for these constants)` — a weaker claim than `proved (structural)`
and deliberately labelled apart. It holds for every input, but only for the
weights this model carries, where a structural proof holds for any payload at
all.

`tested` is not a proof either. A batch-norm fold re-associates, so the two
sides agree as polynomials but not bit for bit; the honest verdict is that their
coefficients agree within a tolerance. Only `Identical` claims can be refuted
outright, which is why a disagreement here would be reported as evidence rather
than as a counterexample.

  $ ../bin/native_graph.exe transform --fold --verify-symbolic quick \
  >   --pt2 "$PT2_DATA/resnet18/resnet18.pt2" | sed '/^graph$/,$d'
  nodes: 174 -> 50
  constants: 42, of which 41 folded
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
  symbolic verification: fold_const
    (root)
         82  proved (structural) [sampled 4]
         47  unproved (too large)
         21  vacuous
    torch.ops.aten._native_batch_norm_legit_no_training.default
         20  unproved (too large)
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
         21  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          2  unproved (too large)
  symbolic verification: fold_batch_norm
    (root)
         82  proved (structural) [sampled 4]
         67  unproved (too large)
        220  vacuous
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
          1  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          1  unproved (too large)
          1  unproved (unsupported relation)
  symbolic verification: fold_const
    (root)
         40  proved (for these constants) [sampled 4]
        142  proved (structural) [sampled 4]
         87  unproved (too large)
         80  vacuous
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
          1  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          2  unproved (too large)
  symbolic verification: fold_const
    (root)
         20  proved (for these constants) [sampled 4]
        142  proved (structural) [sampled 4]
         87  unproved (too large)
         20  vacuous
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
          1  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          2  unproved (too large)
  symbolic verification: fold_const
    (root)
         20  proved (for these constants) [sampled 4]
        102  proved (structural) [sampled 4]
         87  unproved (too large)
         40  vacuous
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
          1  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          2  unproved (too large)
  symbolic verification: fold_const
    (root)
         40  proved (for these constants) [sampled 4]
         42  proved (structural) [sampled 4]
         87  unproved (too large)
         40  vacuous
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
          1  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          2  unproved (too large)
  symbolic verification: fold_const
    (root)
         20  proved (for these constants) [sampled 4]
          2  proved (structural) [sampled 4]
         67  unproved (too large)
         80  vacuous
    torch.ops.aten.addmm.default
          1  proved (structural) [sampled 4]
    torch.ops.aten.convolution.default
          1  unproved (too large)
    torch.ops.aten.max_pool2d_with_indices.default
          2  unproved (too large)
  symbolic verification: total
      140  proved (for these constants) [sampled 4]
     1438  proved (structural) [sampled 4]
        1  unproved (over max_nodes) [sampled 4]
        1  unproved (over max_rounds) [sampled 4]
     2011  unproved (too large)
        1  unproved (unsupported relation)
      615  vacuous
