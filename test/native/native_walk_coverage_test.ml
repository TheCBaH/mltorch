(* Random-walk equivalence for native ops (1/2): split out of
   native_walk_test.ml (T7.1), then split AGAIN into two halves (this
   session's own walks pushed the merged coverage file past the 750-line
   new-file cap) -- both purely file-organization splits, not semantic
   ones: [Native_op_walk.all_walks] and its seed-by-index contract are
   unchanged, so every step here is identical to what ran in the merged
   file. This half covers walks [0, 36) of [all_walks]; see
   native_walk_coverage_test2.ml for [36, length). Pure OCaml (no
   libtorch) -- the oracle is Direct vs Symbolic. *)

let capture f = print_string (Core.Pretty.capture_to_string f)

module Pcg = Walk_core.Pcg

let%expect_test "native walk coverage (1/2)" =
  (* [assert], not [ignore]: a mismatch also truncates the printed walk, so the
     golden would catch it -- but only by way of a diff that reads like a
     rewritten expectation. Asserting says which of the two happened. *)
  capture (fun ppf ->
      List.iteri
        (fun i m ->
          assert (
            Native_op_walk.run m ~ppf
              ~pcg:(Pcg.seed ~seed:(Int64.of_int i) ~seq:1L)
              ~steps:5))
        (List.filteri (fun i _ -> i < 36) Native_op_walk.all_walks));
  [%expect
    {|
    step 0: {shape=[1,4,8,8] output_size=[4,4]}
    [native] adaptive_avg_pool2d: direct==symbolic
    step 1 [n]: {shape=[2,4,8,8] output_size=[4,4]}
    [native] adaptive_avg_pool2d: direct==symbolic
    step 2 [input_h]: {shape=[2,4,9,8] output_size=[4,4]}
    [native] adaptive_avg_pool2d: direct==symbolic
    step 3 [c]: {shape=[2,6,9,8] output_size=[4,4]}
    [native] adaptive_avg_pool2d: direct==symbolic
    step 4 [input_w]: {shape=[2,6,9,10] output_size=[4,4]}
    [native] adaptive_avg_pool2d: direct==symbolic
    step 5 [out_w]: {shape=[2,6,9,10] output_size=[4,3]}
    [native] adaptive_avg_pool2d: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] add: direct==symbolic
    step 1 [input]: [n=1 c=8 h=4 w=4]
    [native] add: direct==symbolic
    step 2 [input]: [n=1 c=8 h=10 w=4]
    [native] add: direct==symbolic
    step 3 [input]: [n=2 c=8 h=10 w=4]
    [native] add: direct==symbolic
    step 4 [input]: [n=2 c=8 h=15 w=4]
    [native] add: direct==symbolic
    step 5 [input]: [n=1 c=8 h=15 w=4]
    [native] add: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] add_i64: direct==symbolic
    step 1 [input]: [n=1 c=5 h=4 w=4]
    [native] add_i64: direct==symbolic
    step 2 [input]: [n=1 c=5 h=4 w=4]
    [native] add_i64: direct==symbolic
    step 3 [input]: [n=1 c=4 h=4 w=4]
    [native] add_i64: direct==symbolic
    step 4 [input]: [n=1 c=4 h=4 w=11]
    [native] add_i64: direct==symbolic
    step 5 [input]: [n=1 c=9 h=4 w=11]
    [native] add_i64: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] scalar=3
    [native] add_scalar: direct==symbolic
    step 1 [scalar]: [n=1 c=3 h=4 w=4] scalar=-2
    [native] add_scalar: direct==symbolic
    step 2 [scalar]: [n=1 c=3 h=4 w=4] scalar=0.5
    [native] add_scalar: direct==symbolic
    step 3 [input]: [n=1 c=3 h=5 w=4] scalar=0.5
    [native] add_scalar: direct==symbolic
    step 4 [scalar]: [n=1 c=3 h=5 w=4] scalar=0.5
    [native] add_scalar: direct==symbolic
    step 5 [scalar]: [n=1 c=3 h=5 w=4] scalar=0.1
    [native] add_scalar: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=false}
    [native] amax: direct==symbolic
    step 1 [input]: {shape=[n=2 c=4 h=1 w=4] dims=[H,W] keepdim=false}
    [native] amax: direct==symbolic
    step 2 [input]: {shape=[n=2 c=28 h=1 w=4] dims=[H,W] keepdim=false}
    [native] amax: direct==symbolic
    step 3 [keepdim]: {shape=[n=2 c=28 h=1 w=4] dims=[H,W] keepdim=false}
    [native] amax: direct==symbolic
    step 4 [input]: {shape=[n=2 c=28 h=15 w=4] dims=[H,W] keepdim=false}
    [native] amax: direct==symbolic
    step 5 [input]: {shape=[n=2 c=28 h=15 w=13] dims=[H,W] keepdim=false}
    [native] amax: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0} ceil_mode=false count_include_pad=true
    [native] avg_pool2d: direct==symbolic
    step 1 [kernel]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=2x2 pad=0x0} ceil_mode=false count_include_pad=true
    [native] avg_pool2d: direct==symbolic
    step 2 [input]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=2x2 pad=0x0} ceil_mode=false count_include_pad=true
    [native] avg_pool2d: direct==symbolic
    step 3 [stride]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=2x1 pad=0x0} ceil_mode=false count_include_pad=true
    [native] avg_pool2d: direct==symbolic
    step 4 [ceil_mode]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=2x1 pad=0x0} ceil_mode=true count_include_pad=true
    [native] avg_pool2d: direct==symbolic
    step 5 [pad]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=2x1 pad=1x0} ceil_mode=true count_include_pad=true
    [native] avg_pool2d: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] eps=1e-05}
    [native] batch_norm: direct==symbolic
    step 1 [input]: {shape=[n=2 c=4 h=4 w=5] eps=1e-05}
    [native] batch_norm: direct==symbolic
    step 2 [eps]: {shape=[n=2 c=4 h=4 w=5] eps=0.001}
    [native] batch_norm: direct==symbolic
    step 3 [input]: {shape=[n=2 c=29 h=4 w=5] eps=0.001}
    [native] batch_norm: direct==symbolic
    step 4 [input]: {shape=[n=2 c=29 h=4 w=6] eps=0.001}
    [native] batch_norm: direct==symbolic
    step 5 [eps]: {shape=[n=2 c=29 h=4 w=6] eps=0}
    [native] batch_norm: direct==symbolic
    step 0: {d=1 heads=2 n=2 m=3 p=4}
    [native] batched_matmul: direct==symbolic
    step 1 [d]: {d=2 heads=2 n=2 m=3 p=4}
    [native] batched_matmul: direct==symbolic
    step 2 [p]: {d=2 heads=2 n=2 m=3 p=8}
    [native] batched_matmul: direct==symbolic
    step 3 [d]: {d=1 heads=2 n=2 m=3 p=8}
    [native] batched_matmul: direct==symbolic
    step 4 [d]: {d=2 heads=2 n=2 m=3 p=8}
    [native] batched_matmul: direct==symbolic
    step 5 [heads]: {d=2 heads=2 n=2 m=3 p=8}
    [native] batched_matmul: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] bitwise_not: direct==symbolic
    step 1 [input]: [n=2 c=3 h=4 w=4]
    [native] bitwise_not: direct==symbolic
    step 2 [input]: [n=2 c=3 h=4 w=4]
    [native] bitwise_not: direct==symbolic
    step 3 [input]: [n=1 c=3 h=4 w=4]
    [native] bitwise_not: direct==symbolic
    step 4 [input]: [n=1 c=3 h=4 w=6]
    [native] bitwise_not: direct==symbolic
    step 5 [input]: [n=1 c=27 h=4 w=6]
    [native] bitwise_not: direct==symbolic
    step 0: {batch=1 n=2 m=3 p=4}
    [native] bmm: direct==symbolic
    step 1 [batch]: {batch=1 n=2 m=3 p=4}
    [native] bmm: direct==symbolic
    step 2 [m]: {batch=1 n=2 m=6 p=4}
    [native] bmm: direct==symbolic
    step 3 [p]: {batch=1 n=2 m=6 p=4}
    [native] bmm: direct==symbolic
    step 4 [m]: {batch=1 n=2 m=1 p=4}
    [native] bmm: direct==symbolic
    step 5 [n]: {batch=1 n=8 m=1 p=4}
    [native] bmm: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] {min=0; max=6}
    [native] clamp: direct==symbolic
    step 1 [bounds]: [n=1 c=3 h=4 w=4] {min=-1; max=1}
    [native] clamp: direct==symbolic
    step 2 [bounds]: [n=1 c=3 h=4 w=4] {min=0; max=nan}
    [native] clamp: direct==symbolic
    step 3 [input]: [n=1 c=3 h=2 w=4] {min=0; max=nan}
    [native] clamp: direct==symbolic
    step 4 [bounds]: [n=1 c=3 h=2 w=4] {min=-1; max=1}
    [native] clamp: direct==symbolic
    step 5 [input]: [n=1 c=3 h=7 w=4] {min=-1; max=1}
    [native] clamp: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] clone: direct==symbolic
    step 1 [input]: [n=1 c=3 h=15 w=4]
    [native] clone: direct==symbolic
    step 2 [input]: [n=1 c=3 h=6 w=4]
    [native] clone: direct==symbolic
    step 3 [input]: [n=1 c=3 h=15 w=4]
    [native] clone: direct==symbolic
    step 4 [input]: [n=1 c=3 h=15 w=4]
    [native] clone: direct==symbolic
    step 5 [input]: [n=1 c=3 h=15 w=5]
    [native] clone: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=8}
    [native] conv2d: direct==symbolic
    step 1 [pad]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 pad=0x2 dilation=1x1 groups=1 out_c=8}
    [native] conv2d: direct==symbolic
    step 2 [kernel]: {shape=[n=1 c=4 h=8 w=8] kernel=1x4 stride=1x1 pad=0x2 dilation=1x1 groups=1 out_c=8}
    [native] conv2d: direct==symbolic
    step 3 [groups]: {shape=[n=1 c=4 h=8 w=8] kernel=1x4 stride=1x1 pad=0x2 dilation=1x1 groups=4 out_c=8}
    [native] conv2d: direct==symbolic
    step 4 [input]: {shape=[n=1 c=4 h=11 w=8] kernel=1x4 stride=1x1 pad=0x2 dilation=1x1 groups=4 out_c=8}
    [native] conv2d: direct==symbolic
    step 5 [dilation]: {shape=[n=1 c=4 h=11 w=8] kernel=1x4 stride=1x1 pad=0x2 dilation=2x1 groups=4 out_c=8}
    [native] conv2d: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=1x1 groups=1 out_c=8 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 1 [dilation]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=3x2 groups=1 out_c=8 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 2 [stride]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=3x2 groups=1 out_c=8 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 3 [stride]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=3x2 groups=1 out_c=8 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 4 [dilation]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=1x2 groups=1 out_c=8 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 5 [groups]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=1x2 groups=1 out_c=8 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=8}
    [native] convolution: direct==symbolic
    step 1 [pad]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 pad=0x2 dilation=1x1 groups=1 out_c=8}
    [native] convolution: direct==symbolic
    step 2 [dilation]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 pad=0x2 dilation=2x3 groups=1 out_c=8}
    [native] convolution: direct==symbolic
    step 3 [input]: {shape=[n=2 c=4 h=8 w=8] kernel=3x3 stride=1x1 pad=0x2 dilation=2x3 groups=1 out_c=8}
    [native] convolution: direct==symbolic
    step 4 [input]: {shape=[n=2 c=4 h=16 w=8] kernel=3x3 stride=1x1 pad=0x2 dilation=2x3 groups=1 out_c=8}
    [native] convolution: direct==symbolic
    step 5 [input]: {shape=[n=2 c=17 h=16 w=8] kernel=3x3 stride=1x1 pad=0x2 dilation=2x3 groups=1 out_c=8}
    [native] convolution: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] axis=C}
    [native] cumsum: direct==symbolic
    step 1 [input]: {shape=[n=2 c=4 h=4 w=4] axis=C}
    [native] cumsum: direct==symbolic
    step 2 [axis]: {shape=[n=2 c=4 h=4 w=4] axis=D}
    [native] cumsum: direct==symbolic
    step 3 [input]: {shape=[n=2 c=4 h=4 w=9] axis=D}
    [native] cumsum: direct==symbolic
    step 4 [axis]: {shape=[n=2 c=4 h=4 w=9] axis=C}
    [native] cumsum: direct==symbolic
    step 5 [axis]: {shape=[n=2 c=4 h=4 w=9] axis=H}
    [native] cumsum: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] div: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=4]
    [native] div: direct==symbolic
    step 2 [input]: [n=1 c=3 h=4 w=5]
    [native] div: direct==symbolic
    step 3 [input]: [n=1 c=3 h=11 w=5]
    [native] div: direct==symbolic
    step 4 [input]: [n=1 c=3 h=11 w=5]
    [native] div: direct==symbolic
    step 5 [input]: [n=1 c=3 h=11 w=1]
    [native] div: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] scalar=3
    [native] div_scalar: direct==symbolic
    step 1 [scalar]: [n=1 c=3 h=4 w=4] scalar=3
    [native] div_scalar: direct==symbolic
    step 2 [input]: [n=1 c=3 h=15 w=4] scalar=3
    [native] div_scalar: direct==symbolic
    step 3 [scalar]: [n=1 c=3 h=15 w=4] scalar=0.5
    [native] div_scalar: direct==symbolic
    step 4 [scalar]: [n=1 c=3 h=15 w=4] scalar=3
    [native] div_scalar: direct==symbolic
    step 5 [scalar]: [n=1 c=3 h=15 w=4] scalar=3
    [native] div_scalar: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] scalar=3
    [native] eq_scalar: direct==symbolic
    step 1 [scalar]: [n=1 c=3 h=4 w=4] scalar=6
    [native] eq_scalar: direct==symbolic
    step 2 [scalar]: [n=1 c=3 h=4 w=4] scalar=0.5
    [native] eq_scalar: direct==symbolic
    step 3 [input]: [n=1 c=3 h=4 w=15] scalar=0.5
    [native] eq_scalar: direct==symbolic
    step 4 [input]: [n=1 c=3 h=16 w=15] scalar=0.5
    [native] eq_scalar: direct==symbolic
    step 5 [input]: [n=2 c=3 h=16 w=15] scalar=0.5
    [native] eq_scalar: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] eq_tensor: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=4]
    [native] eq_tensor: direct==symbolic
    step 2 [input]: [n=2 c=3 h=4 w=4]
    [native] eq_tensor: direct==symbolic
    step 3 [input]: [n=2 c=3 h=6 w=4]
    [native] eq_tensor: direct==symbolic
    step 4 [input]: [n=2 c=3 h=6 w=8]
    [native] eq_tensor: direct==symbolic
    step 5 [input]: [n=2 c=3 h=6 w=15]
    [native] eq_tensor: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] expand: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=4]
    [native] expand: direct==symbolic
    step 2 [input]: [n=1 c=3 h=4 w=11]
    [native] expand: direct==symbolic
    step 3 [input]: [n=1 c=3 h=4 w=11]
    [native] expand: direct==symbolic
    step 4 [input]: [n=1 c=3 h=11 w=11]
    [native] expand: direct==symbolic
    step 5 [input]: [n=1 c=28 h=11 w=11]
    [native] expand: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] none
    [native] gelu: direct==symbolic
    step 1 [input]: [n=1 c=17 h=4 w=4] none
    [native] gelu: direct==symbolic
    step 2 [approximate]: [n=1 c=17 h=4 w=4] none
    [native] gelu: direct==symbolic
    step 3 [input]: [n=1 c=17 h=8 w=4] none
    [native] gelu: direct==symbolic
    step 4 [approximate]: [n=1 c=17 h=8 w=4] tanh
    [native] gelu: direct==symbolic
    step 5 [input]: [n=1 c=17 h=7 w=4] tanh
    [native] gelu: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] scalar=3
    [native] gt_scalar: direct==symbolic
    step 1 [input]: [n=2 c=3 h=4 w=4] scalar=3
    [native] gt_scalar: direct==symbolic
    step 2 [scalar]: [n=2 c=3 h=4 w=4] scalar=3
    [native] gt_scalar: direct==symbolic
    step 3 [scalar]: [n=2 c=3 h=4 w=4] scalar=0.5
    [native] gt_scalar: direct==symbolic
    step 4 [input]: [n=1 c=3 h=4 w=4] scalar=0.5
    [native] gt_scalar: direct==symbolic
    step 5 [scalar]: [n=1 c=3 h=4 w=4] scalar=3
    [native] gt_scalar: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] hardsigmoid: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=5]
    [native] hardsigmoid: direct==symbolic
    step 2 [input]: [n=1 c=3 h=4 w=6]
    [native] hardsigmoid: direct==symbolic
    step 3 [input]: [n=1 c=3 h=4 w=4]
    [native] hardsigmoid: direct==symbolic
    step 4 [input]: [n=2 c=3 h=4 w=4]
    [native] hardsigmoid: direct==symbolic
    step 5 [input]: [n=2 c=3 h=4 w=4]
    [native] hardsigmoid: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] hardswish: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=4]
    [native] hardswish: direct==symbolic
    step 2 [input]: [n=1 c=4 h=4 w=4]
    [native] hardswish: direct==symbolic
    step 3 [input]: [n=2 c=4 h=4 w=4]
    [native] hardswish: direct==symbolic
    step 4 [input]: [n=2 c=4 h=14 w=4]
    [native] hardswish: direct==symbolic
    step 5 [input]: [n=2 c=4 h=14 w=14]
    [native] hardswish: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] {min_val=0; max_val=6}
    [native] hardtanh: direct==symbolic
    step 1 [bounds]: [n=1 c=3 h=4 w=4] {min_val=-1; max_val=1}
    [native] hardtanh: direct==symbolic
    step 2 [bounds]: [n=1 c=3 h=4 w=4] {min_val=0; max_val=6}
    [native] hardtanh: direct==symbolic
    step 3 [bounds]: [n=1 c=3 h=4 w=4] {min_val=1; max_val=-1}
    [native] hardtanh: direct==symbolic
    step 4 [bounds]: [n=1 c=3 h=4 w=4] {min_val=1; max_val=-1}
    [native] hardtanh: direct==symbolic
    step 5 [bounds]: [n=1 c=3 h=4 w=4] {min_val=-1; max_val=1}
    [native] hardtanh: direct==symbolic
    step 0: {n=1 h=2 w=3 c=4 axis=W l=2}
    [native] index_tensor: direct==symbolic
    step 1 [w]: {n=1 h=2 w=4 c=4 axis=W l=2}
    [native] index_tensor: direct==symbolic
    step 2 [c]: {n=1 h=2 w=4 c=1 axis=W l=2}
    [native] index_tensor: direct==symbolic
    step 3 [w]: {n=1 h=2 w=2 c=1 axis=W l=2}
    [native] index_tensor: direct==symbolic
    step 4 [n]: {n=1 h=2 w=2 c=1 axis=W l=2}
    [native] index_tensor: direct==symbolic
    step 5 [l]: {n=1 h=2 w=2 c=1 axis=W l=4}
    [native] index_tensor: direct==symbolic
    step 0: {shape=[n=1 c=5 h=4 w=3] k=1 eps=1e-05 weight=true bias=true}
    [native] layer_norm: direct==symbolic
    step 1 [input]: {shape=[n=1 c=5 h=4 w=3] k=1 eps=1e-05 weight=true bias=true}
    [native] layer_norm: direct==symbolic
    step 2 [input]: {shape=[n=2 c=5 h=4 w=3] k=1 eps=1e-05 weight=true bias=true}
    [native] layer_norm: direct==symbolic
    step 3 [weight]: {shape=[n=2 c=5 h=4 w=3] k=1 eps=1e-05 weight=false bias=true}
    [native] layer_norm: direct==symbolic
    step 4 [weight]: {shape=[n=2 c=5 h=4 w=3] k=1 eps=1e-05 weight=true bias=true}
    [native] layer_norm: direct==symbolic
    step 5 [input]: {shape=[n=2 c=5 h=3 w=3] k=1 eps=1e-05 weight=true bias=true}
    [native] layer_norm: direct==symbolic
    step 0: {shape=[n=1 c=8 h=1 w=4] out_features=6 bias=true}
    [native] linear: direct==symbolic
    step 1 [out_features]: {shape=[n=1 c=8 h=1 w=4] out_features=17 bias=true}
    [native] linear: direct==symbolic
    step 2 [bias]: {shape=[n=1 c=8 h=1 w=4] out_features=17 bias=false}
    [native] linear: direct==symbolic
    step 3 [bias]: {shape=[n=1 c=8 h=1 w=4] out_features=17 bias=true}
    [native] linear: direct==symbolic
    step 4 [out_features]: {shape=[n=1 c=8 h=1 w=4] out_features=3 bias=true}
    [native] linear: direct==symbolic
    step 5 [bias]: {shape=[n=1 c=8 h=1 w=4] out_features=3 bias=true}
    [native] linear: direct==symbolic
    step 0: {hidden_size=3 input_size=2 seq=4 batch=2 num_layers=1 bidirectional=false bias=true batch_first=false}
    [native] lstm: direct==symbolic
    step 1 [num_layers]: {hidden_size=3 input_size=2 seq=4 batch=2 num_layers=3 bidirectional=false bias=true batch_first=false}
    [native] lstm: direct==symbolic
    step 2 [bidirectional]: {hidden_size=3 input_size=2 seq=4 batch=2 num_layers=1 bidirectional=true bias=true batch_first=false}
    [native] lstm: direct==symbolic
    step 3 [batch]: {hidden_size=3 input_size=2 seq=4 batch=2 num_layers=1 bidirectional=true bias=true batch_first=false}
    [native] lstm: direct==symbolic
    step 4 [batch]: {hidden_size=3 input_size=2 seq=4 batch=1 num_layers=1 bidirectional=true bias=true batch_first=false}
    [native] lstm: direct==symbolic
    step 5 [hidden_size]: {hidden_size=31 input_size=2 seq=4 batch=1 num_layers=1 bidirectional=true bias=true batch_first=false}
    [native] lstm: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] axis=C keepdim=false}
    [native] max_dim: direct==symbolic
    step 1 [input]: {shape=[n=2 c=20 h=4 w=4] axis=C keepdim=false}
    [native] max_dim: direct==symbolic
    step 2 [axis]: {shape=[n=2 c=20 h=4 w=4] axis=T keepdim=false}
    [native] max_dim: direct==symbolic
    step 3 [keepdim]: {shape=[n=2 c=20 h=4 w=4] axis=T keepdim=true}
    [native] max_dim: direct==symbolic
    step 4 [axis]: {shape=[n=2 c=20 h=4 w=4] axis=W keepdim=true}
    [native] max_dim: direct==symbolic
    step 5 [axis]: {shape=[n=2 c=20 h=4 w=4] axis=D keepdim=true}
    [native] max_dim: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0} ceil_mode=false
    [native] max_pool2d: direct==symbolic
    step 1 [ceil_mode]: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0} ceil_mode=true
    [native] max_pool2d: direct==symbolic
    step 2 [input]: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0} ceil_mode=true
    [native] max_pool2d: direct==symbolic
    step 3 [input]: {shape=[n=1 c=4 h=5 w=8] kernel=2x2 stride=2x2 pad=0x0} ceil_mode=true
    [native] max_pool2d: direct==symbolic
    step 4 [stride]: {shape=[n=1 c=4 h=5 w=8] kernel=2x2 stride=3x3 pad=0x0} ceil_mode=true
    [native] max_pool2d: direct==symbolic
    step 5 [pad]: {shape=[n=1 c=4 h=5 w=8] kernel=2x2 stride=3x3 pad=1x1} ceil_mode=true
    [native] max_pool2d: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0} ceil_mode=false
    [native] max_pool2d_with_indices: direct==symbolic
    step 1 [stride]: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x3 pad=0x0} ceil_mode=false
    [native] max_pool2d_with_indices: direct==symbolic
    step 2 [pad]: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x3 pad=0x1} ceil_mode=false
    [native] max_pool2d_with_indices: direct==symbolic
    step 3 [ceil_mode]: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x3 pad=0x1} ceil_mode=false
    [native] max_pool2d_with_indices: direct==symbolic
    step 4 [kernel]: {shape=[n=1 c=4 h=8 w=8] kernel=5x2 stride=2x3 pad=0x1} ceil_mode=false
    [native] max_pool2d_with_indices: direct==symbolic
    step 5 [kernel]: {shape=[n=1 c=4 h=8 w=8] kernel=4x1 stride=2x3 pad=0x0} ceil_mode=false
    [native] max_pool2d_with_indices: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=false}
    [native] mean: direct==symbolic
    step 1 [input]: {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=false}
    [native] mean: direct==symbolic
    step 2 [input]: {shape=[n=2 c=16 h=4 w=4] dims=[H,W] keepdim=false}
    [native] mean: direct==symbolic
    step 3 [input]: {shape=[n=2 c=31 h=4 w=4] dims=[H,W] keepdim=false}
    [native] mean: direct==symbolic
    step 4 [dims]: {shape=[n=2 c=31 h=4 w=4] dims=[N,H,W] keepdim=false}
    [native] mean: direct==symbolic
    step 5 [keepdim]: {shape=[n=2 c=31 h=4 w=4] dims=[N,H,W] keepdim=true}
    [native] mean: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] mul: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=4]
    [native] mul: direct==symbolic
    step 2 [input]: [n=1 c=3 h=7 w=4]
    [native] mul: direct==symbolic
    step 3 [input]: [n=1 c=3 h=7 w=4]
    [native] mul: direct==symbolic
    step 4 [input]: [n=1 c=3 h=5 w=4]
    [native] mul: direct==symbolic
    step 5 [input]: [n=1 c=3 h=5 w=6]
    [native] mul: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] mul_i64: direct==symbolic
    step 1 [input]: [n=1 c=3 h=6 w=4]
    [native] mul_i64: direct==symbolic
    step 2 [input]: [n=1 c=7 h=6 w=4]
    [native] mul_i64: direct==symbolic
    step 3 [input]: [n=1 c=7 h=6 w=4]
    [native] mul_i64: direct==symbolic
    step 4 [input]: [n=1 c=7 h=6 w=4]
    [native] mul_i64: direct==symbolic
    step 5 [input]: [n=1 c=16 h=6 w=4]
    [native] mul_i64: direct==symbolic
    |}]
