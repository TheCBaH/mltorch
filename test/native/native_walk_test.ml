(* Random-walk equivalence for native ops: walk each op's own config space and
   assert the Direct and Symbolic backends agree at every step. Pure OCaml (no
   libtorch) — the oracle is Direct vs Symbolic. *)

module Pcg = Walk_core.Pcg

let capture f = print_string (Core.Pretty.capture_to_string f)

let%expect_test "native walk coverage" =
  capture (fun ppf ->
      List.iteri
        (fun i m ->
          Native_op_walk.run m ~ppf
            ~pcg:(Pcg.seed ~seed:(Int64.of_int i) ~seq:1L)
            ~steps:5)
        Native_op_walk.all_walks);
  [%expect
    {|
    step 0: [n=1 c=3 h=4 w=4]
    [native] relu: direct==symbolic
    step 1 [input]: [n=1 c=3 h=13 w=4]
    [native] relu: direct==symbolic
    step 2 [input]: [n=2 c=3 h=13 w=4]
    [native] relu: direct==symbolic
    step 3 [input]: [n=2 c=3 h=1 w=4]
    [native] relu: direct==symbolic
    step 4 [input]: [n=1 c=3 h=1 w=4]
    [native] relu: direct==symbolic
    step 5 [input]: [n=2 c=3 h=1 w=4]
    [native] relu: direct==symbolic
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
    [native] mul: direct==symbolic
    step 1 [input]: [n=1 c=5 h=4 w=4]
    [native] mul: direct==symbolic
    step 2 [input]: [n=1 c=5 h=4 w=4]
    [native] mul: direct==symbolic
    step 3 [input]: [n=1 c=4 h=4 w=4]
    [native] mul: direct==symbolic
    step 4 [input]: [n=1 c=4 h=4 w=11]
    [native] mul: direct==symbolic
    step 5 [input]: [n=1 c=9 h=4 w=11]
    [native] mul: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=false}
    [native] mean: direct==symbolic
    step 1 [input]: {shape=[n=2 c=4 h=12 w=4] dims=[H,W] keepdim=false}
    [native] mean: direct==symbolic
    step 2 [input]: {shape=[n=2 c=4 h=12 w=4] dims=[H,W] keepdim=false}
    [native] mean: direct==symbolic
    step 3 [keepdim]: {shape=[n=2 c=4 h=12 w=4] dims=[H,W] keepdim=false}
    [native] mean: direct==symbolic
    step 4 [keepdim]: {shape=[n=2 c=4 h=12 w=4] dims=[H,W] keepdim=true}
    [native] mean: direct==symbolic
    step 5 [dims]: {shape=[n=2 c=4 h=12 w=4] dims=[W] keepdim=true}
    [native] mean: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0}
    [native] max_pool2d: direct==symbolic
    step 1 [input]: {shape=[n=1 c=4 h=11 w=8] kernel=2x2 stride=2x2 pad=0x0}
    [native] max_pool2d: direct==symbolic
    step 2 [pad]: {shape=[n=1 c=4 h=11 w=8] kernel=2x2 stride=2x2 pad=1x0}
    [native] max_pool2d: direct==symbolic
    step 3 [kernel]: {shape=[n=1 c=4 h=11 w=8] kernel=3x1 stride=2x2 pad=1x0}
    [native] max_pool2d: direct==symbolic
    step 4 [kernel]: {shape=[n=1 c=4 h=11 w=8] kernel=3x4 stride=2x2 pad=1x0}
    [native] max_pool2d: direct==symbolic
    step 5 [kernel]: {shape=[n=1 c=4 h=11 w=8] kernel=5x5 stride=2x2 pad=1x0}
    [native] max_pool2d: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0}
    [native] avg_pool2d: direct==symbolic
    step 1 [kernel]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=2x2 pad=0x0}
    [native] avg_pool2d: direct==symbolic
    step 2 [stride]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=1x1 pad=0x0}
    [native] avg_pool2d: direct==symbolic
    step 3 [input]: {shape=[n=1 c=4 h=3 w=8] kernel=3x4 stride=1x1 pad=0x0}
    [native] avg_pool2d: direct==symbolic
    step 4 [stride]: {shape=[n=1 c=4 h=3 w=8] kernel=3x4 stride=2x3 pad=0x0}
    [native] avg_pool2d: direct==symbolic
    step 5 [stride]: {shape=[n=1 c=4 h=3 w=8] kernel=3x4 stride=3x3 pad=0x0}
    [native] avg_pool2d: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=8}
    [native] conv2d: direct==symbolic
    step 1 [input]: {shape=[n=1 c=4 h=8 w=15] kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=8}
    [native] conv2d: direct==symbolic
    step 2 [dilation]: {shape=[n=1 c=4 h=8 w=15] kernel=3x3 stride=1x1 pad=1x1 dilation=1x3 groups=1 out_c=8}
    [native] conv2d: direct==symbolic
    step 3 [stride]: {shape=[n=1 c=4 h=8 w=15] kernel=3x3 stride=3x2 pad=1x1 dilation=1x3 groups=1 out_c=8}
    [native] conv2d: direct==symbolic
    step 4 [kernel]: {shape=[n=1 c=4 h=8 w=15] kernel=5x2 stride=3x2 pad=1x1 dilation=1x3 groups=1 out_c=8}
    [native] conv2d: direct==symbolic
    step 5 [out_channels]: {shape=[n=1 c=4 h=8 w=15] kernel=5x2 stride=3x2 pad=1x1 dilation=1x3 groups=1 out_c=3}
    [native] conv2d: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=8}
    [native] convolution: direct==symbolic
    step 1 [kernel]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=8}
    [native] convolution: direct==symbolic
    step 2 [pad]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=1x1 pad=2x0 dilation=1x1 groups=1 out_c=8}
    [native] convolution: direct==symbolic
    step 3 [out_channels]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=1x1 pad=2x0 dilation=1x1 groups=1 out_c=21}
    [native] convolution: direct==symbolic
    step 4 [input]: {shape=[n=1 c=4 h=8 w=6] kernel=3x4 stride=1x1 pad=2x0 dilation=1x1 groups=1 out_c=21}
    [native] convolution: direct==symbolic
    step 5 [groups]: {shape=[n=1 c=4 h=8 w=6] kernel=3x4 stride=1x1 pad=2x0 dilation=1x1 groups=2 out_c=22}
    [native] convolution: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=1x1 groups=1 out_c=8 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 1 [out_channels]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=1x1 groups=1 out_c=16 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 2 [kernel]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=1x1 groups=1 out_c=16 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 3 [out_channels]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=1x1 groups=1 out_c=25 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 4 [out_channels]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=1x1 groups=1 out_c=6 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 5 [groups]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=1x1 groups=2 out_c=6 padding=same}
    [native] conv2d_padding: direct==symbolic |}]
