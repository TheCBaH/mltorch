(* Random-walk equivalence for native ops: walk each op's own config space and
   assert the Direct and Symbolic backends agree at every step. Pure OCaml (no
   libtorch) — the oracle is Direct vs Symbolic. *)

module Pcg = Walk_core.Pcg

let capture f = print_string (Core.Pretty.capture_to_string f)

(* The coverage sweep runs every walk at 5 steps with seed = index, and on
   linear's seed the [bias] axis is never drawn -- so the sweep's golden shows
   `bias=true` throughout and says nothing about the other state. That is the
   shape of evidence this file exists to avoid: an axis that exists, is never
   exercised, and reads as coverage.

   [Graph_ir]'s [Linear] carries the bias as an OPTION and [Eval_op] synthesizes
   a zero one when it is absent, so the two states are different graphs through
   different code and both need the Direct-vs-Symbolic check. rms_norm's own
   optional operand happens to be drawn by the sweep; this one is not. *)
let%expect_test "native walk: linear reaches both bias states" =
  capture (fun ppf ->
      match Native_op_walk.find "linear" with
      | None -> Format.fprintf ppf "no linear walk registered@."
      | Some m ->
          assert (
            Native_op_walk.run m ~ppf ~pcg:(Pcg.seed ~seed:3L ~seq:1L) ~steps:8));
  [%expect
    {|
    step 0: {shape=[n=1 c=8 h=1 w=4] out_features=6 bias=true}
    [native] linear: direct==symbolic
    step 1 [bias]: {shape=[n=1 c=8 h=1 w=4] out_features=6 bias=true}
    [native] linear: direct==symbolic
    step 2 [bias]: {shape=[n=1 c=8 h=1 w=4] out_features=6 bias=true}
    [native] linear: direct==symbolic
    step 3 [out_features]: {shape=[n=1 c=8 h=1 w=4] out_features=13 bias=true}
    [native] linear: direct==symbolic
    step 4 [out_features]: {shape=[n=1 c=8 h=1 w=4] out_features=19 bias=true}
    [native] linear: direct==symbolic
    step 5 [bias]: {shape=[n=1 c=8 h=1 w=4] out_features=19 bias=false}
    [native] linear: direct==symbolic
    step 6 [input]: {shape=[n=1 c=27 h=1 w=4] out_features=19 bias=false}
    [native] linear: direct==symbolic
    step 7 [out_features]: {shape=[n=1 c=27 h=1 w=4] out_features=16 bias=false}
    [native] linear: direct==symbolic
    step 8 [input]: {shape=[n=1 c=27 h=5 w=4] out_features=16 bias=false}
    [native] linear: direct==symbolic |}]

(* The coverage sweep runs pad at 5 steps on its own index as the seed, and that
   walk never draws an extent of 1 -- so [Pad.Walk]'s config space could emit a
   configuration the op REFUSES, and the sweep stayed green. [Pad_nwalk] builds
   through [Err.or_raise], so an invalid config is not a mismatch, it is an
   uncaught exception that aborts the whole sweep.

   Seed 9 at 12 steps draws H=1 and pattern=reflect_hw, which is the pair that
   used to raise ("reflect pad of axis H by (1, 1) needs each side below the
   extent 1"). Pinned here rather than left to the sweep, because the sweep's
   seed is its INDEX in [all_walks] -- so inserting any walk ahead of pad
   silently re-rolls its dice, and that is how this surfaced. *)
let%expect_test "native walk: pad stays inside its own domain at extent 1" =
  capture (fun ppf ->
      match Native_op_walk.find "pad" with
      | None -> Format.fprintf ppf "no pad walk registered@."
      | Some m ->
          assert (
            Native_op_walk.run m ~ppf ~pcg:(Pcg.seed ~seed:9L ~seq:1L) ~steps:12));
  [%expect
    {|
    step 0: {shape=[n=1 c=3 h=6 w=6] pattern=pad_hw}
    [native] pad: direct==symbolic
    step 1 [pattern]: {shape=[n=1 c=3 h=6 w=6] pattern=pad_asym_w}
    [native] pad: direct==symbolic
    step 2 [input]: {shape=[n=1 c=3 h=6 w=1] pattern=pad_asym_w}
    [native] pad: direct==symbolic
    step 3 [pattern]: {shape=[n=1 c=3 h=6 w=1] pattern=reflect_hw}
    [native] pad: direct==symbolic
    step 4 [input]: {shape=[n=1 c=3 h=6 w=10] pattern=reflect_hw}
    [native] pad: direct==symbolic
    step 5 [pattern]: {shape=[n=1 c=3 h=6 w=10] pattern=reflect_asym_w}
    [native] pad: direct==symbolic
    step 6 [pattern]: {shape=[n=1 c=3 h=6 w=10] pattern=reflect_hw}
    [native] pad: direct==symbolic
    step 7 [input]: {shape=[n=1 c=22 h=6 w=10] pattern=reflect_hw}
    [native] pad: direct==symbolic
    step 8 [pattern]: {shape=[n=1 c=22 h=6 w=10] pattern=reflect_hw}
    [native] pad: direct==symbolic
    step 9 [input]: {shape=[n=1 c=22 h=6 w=10] pattern=reflect_hw}
    [native] pad: direct==symbolic
    step 10 [pattern]: {shape=[n=1 c=22 h=6 w=10] pattern=reflect_asym_w}
    [native] pad: direct==symbolic
    step 11 [input]: {shape=[n=1 c=22 h=6 w=9] pattern=reflect_asym_w}
    [native] pad: direct==symbolic
    step 12 [pattern]: {shape=[n=1 c=22 h=6 w=9] pattern=pad_hw}
    [native] pad: direct==symbolic |}]

let%expect_test "native walk coverage" =
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
    [native] sub: direct==symbolic
    step 1 [input]: [n=1 c=5 h=4 w=4]
    [native] sub: direct==symbolic
    step 2 [input]: [n=1 c=5 h=4 w=4]
    [native] sub: direct==symbolic
    step 3 [input]: [n=1 c=4 h=4 w=4]
    [native] sub: direct==symbolic
    step 4 [input]: [n=1 c=4 h=4 w=11]
    [native] sub: direct==symbolic
    step 5 [input]: [n=1 c=9 h=4 w=11]
    [native] sub: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] mul: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=14]
    [native] mul: direct==symbolic
    step 2 [input]: [n=1 c=3 h=4 w=14]
    [native] mul: direct==symbolic
    step 3 [input]: [n=1 c=1 h=4 w=14]
    [native] mul: direct==symbolic
    step 4 [input]: [n=1 c=1 h=11 w=14]
    [native] mul: direct==symbolic
    step 5 [input]: [n=1 c=1 h=11 w=4]
    [native] mul: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=false}
    [native] mean: direct==symbolic
    step 1 [input]: {shape=[n=2 c=4 h=1 w=4] dims=[H,W] keepdim=false}
    [native] mean: direct==symbolic
    step 2 [input]: {shape=[n=2 c=28 h=1 w=4] dims=[H,W] keepdim=false}
    [native] mean: direct==symbolic
    step 3 [keepdim]: {shape=[n=2 c=28 h=1 w=4] dims=[H,W] keepdim=false}
    [native] mean: direct==symbolic
    step 4 [input]: {shape=[n=2 c=28 h=15 w=4] dims=[H,W] keepdim=false}
    [native] mean: direct==symbolic
    step 5 [input]: {shape=[n=2 c=28 h=15 w=13] dims=[H,W] keepdim=false}
    [native] mean: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=false}
    [native] amax: direct==symbolic
    step 1 [keepdim]: {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=true}
    [native] amax: direct==symbolic
    step 2 [dims]: {shape=[n=2 c=4 h=4 w=4] dims=[C] keepdim=true}
    [native] amax: direct==symbolic
    step 3 [input]: {shape=[n=2 c=4 h=4 w=1] dims=[C] keepdim=true}
    [native] amax: direct==symbolic
    step 4 [dims]: {shape=[n=2 c=4 h=4 w=1] dims=[C] keepdim=true}
    [native] amax: direct==symbolic
    step 5 [input]: {shape=[n=2 c=4 h=10 w=1] dims=[C] keepdim=true}
    [native] amax: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] dims=[H,W] keepdim=false}
    [native] vector_norm: direct==symbolic
    step 1 [input]: {shape=[n=2 c=7 h=4 w=4] dims=[H,W] keepdim=false}
    [native] vector_norm: direct==symbolic
    step 2 [input]: {shape=[n=2 c=19 h=4 w=4] dims=[H,W] keepdim=false}
    [native] vector_norm: direct==symbolic
    step 3 [keepdim]: {shape=[n=2 c=19 h=4 w=4] dims=[H,W] keepdim=false}
    [native] vector_norm: direct==symbolic
    step 4 [keepdim]: {shape=[n=2 c=19 h=4 w=4] dims=[H,W] keepdim=false}
    [native] vector_norm: direct==symbolic
    step 5 [input]: {shape=[n=2 c=19 h=4 w=4] dims=[H,W] keepdim=false}
    [native] vector_norm: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0} ceil_mode=false
    [native] max_pool2d: direct==symbolic
    step 1 [ceil_mode]: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0} ceil_mode=false
    [native] max_pool2d: direct==symbolic
    step 2 [input]: {shape=[n=1 c=15 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0} ceil_mode=false
    [native] max_pool2d: direct==symbolic
    step 3 [pad]: {shape=[n=1 c=15 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0} ceil_mode=false
    [native] max_pool2d: direct==symbolic
    step 4 [stride]: {shape=[n=1 c=15 h=8 w=8] kernel=2x2 stride=3x2 pad=0x0} ceil_mode=false
    [native] max_pool2d: direct==symbolic
    step 5 [stride]: {shape=[n=1 c=15 h=8 w=8] kernel=2x2 stride=1x1 pad=0x0} ceil_mode=false
    [native] max_pool2d: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0}
    [native] avg_pool2d: direct==symbolic
    step 1 [pad]: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=1x0}
    [native] avg_pool2d: direct==symbolic
    step 2 [input]: {shape=[n=1 c=4 h=8 w=16] kernel=2x2 stride=2x2 pad=1x0}
    [native] avg_pool2d: direct==symbolic
    step 3 [kernel]: {shape=[n=1 c=4 h=8 w=16] kernel=3x3 stride=2x2 pad=1x0}
    [native] avg_pool2d: direct==symbolic
    step 4 [input]: {shape=[n=1 c=4 h=8 w=4] kernel=3x3 stride=2x2 pad=1x0}
    [native] avg_pool2d: direct==symbolic
    step 5 [pad]: {shape=[n=1 c=4 h=8 w=4] kernel=3x3 stride=2x2 pad=1x1}
    [native] avg_pool2d: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=8}
    [native] conv2d: direct==symbolic
    step 1 [out_channels]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=30}
    [native] conv2d: direct==symbolic
    step 2 [input]: {shape=[n=1 c=4 h=15 w=8] kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=30}
    [native] conv2d: direct==symbolic
    step 3 [stride]: {shape=[n=1 c=4 h=15 w=8] kernel=3x3 stride=3x2 pad=1x1 dilation=1x1 groups=1 out_c=30}
    [native] conv2d: direct==symbolic
    step 4 [input]: {shape=[n=1 c=4 h=15 w=8] kernel=3x3 stride=3x2 pad=1x1 dilation=1x1 groups=1 out_c=30}
    [native] conv2d: direct==symbolic
    step 5 [input]: {shape=[n=1 c=4 h=15 w=16] kernel=3x3 stride=3x2 pad=1x1 dilation=1x1 groups=1 out_c=30}
    [native] conv2d: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=8}
    [native] convolution: direct==symbolic
    step 1 [out_channels]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=3}
    [native] convolution: direct==symbolic
    step 2 [input]: {shape=[n=1 c=4 h=8 w=13] kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=3}
    [native] convolution: direct==symbolic
    step 3 [kernel]: {shape=[n=1 c=4 h=8 w=13] kernel=2x2 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=3}
    [native] convolution: direct==symbolic
    step 4 [pad]: {shape=[n=1 c=4 h=8 w=13] kernel=2x2 stride=1x1 pad=0x1 dilation=1x1 groups=1 out_c=3}
    [native] convolution: direct==symbolic
    step 5 [dilation]: {shape=[n=1 c=4 h=8 w=13] kernel=2x2 stride=1x1 pad=0x1 dilation=3x2 groups=1 out_c=3}
    [native] convolution: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=1x1 groups=1 out_c=8 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 1 [padding]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=1x1 groups=1 out_c=8 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 2 [out_channels]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=1x1 groups=1 out_c=28 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 3 [out_channels]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=1x1 groups=1 out_c=16 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 4 [kernel]: {shape=[n=1 c=4 h=8 w=8] kernel=5x5 stride=1x1 dilation=1x1 groups=1 out_c=16 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 5 [dilation]: {shape=[n=1 c=4 h=8 w=13] kernel=5x5 stride=1x1 dilation=1x3 groups=1 out_c=16 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 0: {shape=[n=1 c=8 h=1 w=4] out_features=6 bias=true}
    [native] linear: direct==symbolic
    step 1 [out_features]: {shape=[n=1 c=8 h=1 w=4] out_features=2 bias=true}
    [native] linear: direct==symbolic
    step 2 [out_features]: {shape=[n=1 c=8 h=1 w=4] out_features=22 bias=true}
    [native] linear: direct==symbolic
    step 3 [bias]: {shape=[n=1 c=8 h=1 w=4] out_features=22 bias=true}
    [native] linear: direct==symbolic
    step 4 [out_features]: {shape=[n=1 c=8 h=1 w=4] out_features=10 bias=true}
    [native] linear: direct==symbolic
    step 5 [bias]: {shape=[n=1 c=8 h=1 w=4] out_features=10 bias=false}
    [native] linear: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] eps=1e-05}
    [native] batch_norm: direct==symbolic
    step 1 [eps]: {shape=[n=2 c=4 h=4 w=4] eps=1e-05}
    [native] batch_norm: direct==symbolic
    step 2 [eps]: {shape=[n=2 c=4 h=4 w=4] eps=1e-05}
    [native] batch_norm: direct==symbolic
    step 3 [input]: {shape=[n=2 c=4 h=3 w=4] eps=1e-05}
    [native] batch_norm: direct==symbolic
    step 4 [eps]: {shape=[n=2 c=4 h=3 w=4] eps=1e-05}
    [native] batch_norm: direct==symbolic
    step 5 [input]: {shape=[n=2 c=31 h=3 w=4] eps=1e-05}
    [native] batch_norm: direct==symbolic
    step 0: {shape=[n=1 c=5 h=4 w=3] k=1 eps=1e-05 weight=true}
    [native] rms_norm: direct==symbolic
    step 1 [eps]: {shape=[n=1 c=5 h=4 w=3] k=1 eps=0.001 weight=true}
    [native] rms_norm: direct==symbolic
    step 2 [input]: {shape=[n=2 c=5 h=4 w=3] k=1 eps=0.001 weight=true}
    [native] rms_norm: direct==symbolic
    step 3 [eps]: {shape=[n=2 c=5 h=4 w=3] k=1 eps=0 weight=true}
    [native] rms_norm: direct==symbolic
    step 4 [input]: {shape=[n=2 c=5 h=4 w=3] k=1 eps=0 weight=true}
    [native] rms_norm: direct==symbolic
    step 5 [weight]: {shape=[n=2 c=5 h=4 w=3] k=1 eps=0 weight=true}
    [native] rms_norm: direct==symbolic
    step 0: {shape=[n=1 c=5 h=4 w=3] k=1 eps=1e-05 weight=true bias=true}
    [native] layer_norm: direct==symbolic
    step 1 [input]: {shape=[n=1 c=5 h=4 w=3] k=1 eps=1e-05 weight=true bias=true}
    [native] layer_norm: direct==symbolic
    step 2 [k]: {shape=[n=1 c=5 h=4 w=3] k=1 eps=1e-05 weight=true bias=true}
    [native] layer_norm: direct==symbolic
    step 3 [weight]: {shape=[n=1 c=5 h=4 w=3] k=1 eps=1e-05 weight=false bias=true}
    [native] layer_norm: direct==symbolic
    step 4 [eps]: {shape=[n=1 c=5 h=4 w=3] k=1 eps=0 weight=false bias=true}
    [native] layer_norm: direct==symbolic
    step 5 [k]: {shape=[n=1 c=5 h=4 w=3] k=1 eps=0 weight=false bias=true}
    [native] layer_norm: direct==symbolic
    step 0: {batch=1 heads=2 wq=3 wk=4 e=5 mask=present scale=default}
    [native] sdpa: direct==symbolic
    step 1 [mask]: {batch=1 heads=2 wq=3 wk=4 e=5 mask=absent scale=default}
    [native] sdpa: direct==symbolic
    step 2 [wk]: {batch=1 heads=2 wq=3 wk=6 e=5 mask=absent scale=default}
    [native] sdpa: direct==symbolic
    step 3 [wk]: {batch=1 heads=2 wq=3 wk=6 e=5 mask=absent scale=default}
    [native] sdpa: direct==symbolic
    step 4 [e]: {batch=1 heads=2 wq=3 wk=6 e=1 mask=absent scale=default}
    [native] sdpa: direct==symbolic
    step 5 [heads]: {batch=1 heads=2 wq=3 wk=6 e=1 mask=absent scale=default}
    [native] sdpa: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0} ceil_mode=false
    [native] max_pool2d_with_indices: direct==symbolic
    step 1 [kernel]: {shape=[n=1 c=4 h=8 w=8] kernel=1x2 stride=2x2 pad=0x0} ceil_mode=false
    [native] max_pool2d_with_indices: direct==symbolic
    step 2 [stride]: {shape=[n=1 c=4 h=8 w=8] kernel=1x2 stride=3x2 pad=0x0} ceil_mode=false
    [native] max_pool2d_with_indices: direct==symbolic
    step 3 [stride]: {shape=[n=1 c=4 h=8 w=8] kernel=1x2 stride=1x3 pad=0x0} ceil_mode=false
    [native] max_pool2d_with_indices: direct==symbolic
    step 4 [ceil_mode]: {shape=[n=1 c=4 h=8 w=8] kernel=1x2 stride=1x3 pad=0x0} ceil_mode=false
    [native] max_pool2d_with_indices: direct==symbolic
    step 5 [pad]: {shape=[n=1 c=4 h=8 w=8] kernel=1x2 stride=1x3 pad=0x1} ceil_mode=false
    [native] max_pool2d_with_indices: direct==symbolic
    step 0: {shape=[n=1 c=4 h=4 w=4] -> flat}
    [native] reshape: direct==symbolic
    step 1 [input]: {shape=[n=1 c=4 h=4 w=15] -> flat}
    [native] reshape: direct==symbolic
    step 2 [input]: {shape=[n=1 c=4 h=10 w=15] -> flat}
    [native] reshape: direct==symbolic
    step 3 [input]: {shape=[n=1 c=4 h=10 w=15] -> flat}
    [native] reshape: direct==symbolic
    step 4 [input]: {shape=[n=1 c=4 h=10 w=6] -> flat}
    [native] reshape: direct==symbolic
    step 5 [input]: {shape=[n=1 c=4 h=12 w=6] -> flat}
    [native] reshape: direct==symbolic
    step 0: {shape=[n=1 c=4 h=4 w=4] perm=[H<-W, W<-H]}
    [native] permute: direct==symbolic
    step 1 [input]: {shape=[n=1 c=4 h=5 w=4] perm=[H<-W, W<-H]}
    [native] permute: direct==symbolic
    step 2 [input]: {shape=[n=1 c=3 h=5 w=4] perm=[H<-W, W<-H]}
    [native] permute: direct==symbolic
    step 3 [perm]: {shape=[n=1 c=3 h=5 w=4] perm=[W<-C, C<-W]}
    [native] permute: direct==symbolic
    step 4 [input]: {shape=[n=1 c=3 h=5 w=4] perm=[W<-C, C<-W]}
    [native] permute: direct==symbolic
    step 5 [perm]: {shape=[n=1 c=3 h=5 w=4] perm=[H<-W, W<-C, C<-H]}
    [native] permute: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] {min=0; max=6}
    [native] clamp: direct==symbolic
    step 1 [input]: [n=1 c=6 h=4 w=4] {min=0; max=6}
    [native] clamp: direct==symbolic
    step 2 [input]: [n=1 c=12 h=4 w=4] {min=0; max=6}
    [native] clamp: direct==symbolic
    step 3 [input]: [n=1 c=12 h=4 w=5] {min=0; max=6}
    [native] clamp: direct==symbolic
    step 4 [bounds]: [n=1 c=12 h=4 w=5] {min=-1; max=1}
    [native] clamp: direct==symbolic
    step 5 [bounds]: [n=1 c=12 h=4 w=5] {min=0; max=nan}
    [native] clamp: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] clone: direct==symbolic
    step 1 [input]: [n=1 c=17 h=4 w=4]
    [native] clone: direct==symbolic
    step 2 [input]: [n=1 c=17 h=4 w=14]
    [native] clone: direct==symbolic
    step 3 [input]: [n=1 c=17 h=4 w=14]
    [native] clone: direct==symbolic
    step 4 [input]: [n=1 c=17 h=1 w=14]
    [native] clone: direct==symbolic
    step 5 [input]: [n=1 c=17 h=1 w=14]
    [native] clone: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] {min_val=0; max_val=6}
    [native] hardtanh: direct==symbolic
    step 1 [input]: [n=2 c=3 h=4 w=4] {min_val=0; max_val=6}
    [native] hardtanh: direct==symbolic
    step 2 [bounds]: [n=2 c=3 h=4 w=4] {min_val=-1; max_val=1}
    [native] hardtanh: direct==symbolic
    step 3 [bounds]: [n=2 c=3 h=4 w=4] {min_val=-1; max_val=1}
    [native] hardtanh: direct==symbolic
    step 4 [input]: [n=1 c=3 h=4 w=4] {min_val=-1; max_val=1}
    [native] hardtanh: direct==symbolic
    step 5 [bounds]: [n=1 c=3 h=4 w=4] {min_val=-1; max_val=1}
    [native] hardtanh: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] scalar=3
    [native] add_scalar: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=5] scalar=3
    [native] add_scalar: direct==symbolic
    step 2 [input]: [n=1 c=3 h=4 w=6] scalar=3
    [native] add_scalar: direct==symbolic
    step 3 [input]: [n=1 c=3 h=4 w=4] scalar=3
    [native] add_scalar: direct==symbolic
    step 4 [scalar]: [n=1 c=3 h=4 w=4] scalar=0.5
    [native] add_scalar: direct==symbolic
    step 5 [input]: [n=1 c=3 h=4 w=14] scalar=0.5
    [native] add_scalar: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] scalar=3
    [native] div_scalar: direct==symbolic
    step 1 [scalar]: [n=1 c=3 h=4 w=4] scalar=0.5
    [native] div_scalar: direct==symbolic
    step 2 [input]: [n=1 c=3 h=4 w=8] scalar=0.5
    [native] div_scalar: direct==symbolic
    step 3 [input]: [n=2 c=3 h=4 w=8] scalar=0.5
    [native] div_scalar: direct==symbolic
    step 4 [scalar]: [n=2 c=3 h=4 w=8] scalar=3
    [native] div_scalar: direct==symbolic
    step 5 [input]: [n=2 c=10 h=4 w=8] scalar=3
    [native] div_scalar: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] axis=H}
    [native] unbind: direct==symbolic
    step 1 [input]: {shape=[n=2 c=4 h=4 w=1] axis=H}
    [native] unbind: direct==symbolic
    step 2 [input]: {shape=[n=2 c=4 h=4 w=8] axis=H}
    [native] unbind: direct==symbolic
    step 3 [axis]: {shape=[n=2 c=4 h=4 w=8] axis=N}
    [native] unbind: direct==symbolic
    step 4 [input]: {shape=[n=2 c=4 h=6 w=8] axis=N}
    [native] unbind: direct==symbolic
    step 5 [input]: {shape=[n=2 c=4 h=10 w=8] axis=N}
    [native] unbind: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] silu: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=16]
    [native] silu: direct==symbolic
    step 2 [input]: [n=1 c=3 h=3 w=16]
    [native] silu: direct==symbolic
    step 3 [input]: [n=1 c=3 h=9 w=16]
    [native] silu: direct==symbolic
    step 4 [input]: [n=1 c=26 h=9 w=16]
    [native] silu: direct==symbolic
    step 5 [input]: [n=2 c=26 h=9 w=16]
    [native] silu: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] sigmoid: direct==symbolic
    step 1 [input]: [n=1 c=3 h=1 w=4]
    [native] sigmoid: direct==symbolic
    step 2 [input]: [n=2 c=3 h=1 w=4]
    [native] sigmoid: direct==symbolic
    step 3 [input]: [n=2 c=25 h=1 w=4]
    [native] sigmoid: direct==symbolic
    step 4 [input]: [n=2 c=25 h=1 w=12]
    [native] sigmoid: direct==symbolic
    step 5 [input]: [n=2 c=25 h=10 w=12]
    [native] sigmoid: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] none
    [native] gelu: direct==symbolic
    step 1 [approximate]: [n=1 c=3 h=4 w=4] none
    [native] gelu: direct==symbolic
    step 2 [input]: [n=1 c=10 h=4 w=4] none
    [native] gelu: direct==symbolic
    step 3 [approximate]: [n=1 c=10 h=4 w=4] none
    [native] gelu: direct==symbolic
    step 4 [input]: [n=1 c=17 h=4 w=4] none
    [native] gelu: direct==symbolic
    step 5 [input]: [n=1 c=17 h=8 w=4] none
    [native] gelu: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] scalar=3
    [native] mul_scalar: direct==symbolic
    step 1 [input]: [n=2 c=3 h=4 w=4] scalar=3
    [native] mul_scalar: direct==symbolic
    step 2 [scalar]: [n=2 c=3 h=4 w=4] scalar=6
    [native] mul_scalar: direct==symbolic
    step 3 [scalar]: [n=2 c=3 h=4 w=4] scalar=3
    [native] mul_scalar: direct==symbolic
    step 4 [scalar]: [n=2 c=3 h=4 w=4] scalar=0.5
    [native] mul_scalar: direct==symbolic
    step 5 [scalar]: [n=2 c=3 h=4 w=4] scalar=6
    [native] mul_scalar: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] hardsigmoid: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=4]
    [native] hardsigmoid: direct==symbolic
    step 2 [input]: [n=1 c=13 h=4 w=4]
    [native] hardsigmoid: direct==symbolic
    step 3 [input]: [n=1 c=13 h=10 w=4]
    [native] hardsigmoid: direct==symbolic
    step 4 [input]: [n=1 c=13 h=10 w=4]
    [native] hardsigmoid: direct==symbolic
    step 5 [input]: [n=1 c=13 h=10 w=2]
    [native] hardsigmoid: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] hardswish: direct==symbolic
    step 1 [input]: [n=2 c=3 h=4 w=4]
    [native] hardswish: direct==symbolic
    step 2 [input]: [n=2 c=3 h=13 w=4]
    [native] hardswish: direct==symbolic
    step 3 [input]: [n=2 c=3 h=13 w=4]
    [native] hardswish: direct==symbolic
    step 4 [input]: [n=1 c=3 h=13 w=4]
    [native] hardswish: direct==symbolic
    step 5 [input]: [n=1 c=4 h=13 w=4]
    [native] hardswish: direct==symbolic
    step 0: {shape=[n=1 c=3 h=6 w=6] pattern=pad_hw}
    [native] pad: direct==symbolic
    step 1 [pattern]: {shape=[n=1 c=3 h=6 w=6] pattern=pad_hw}
    [native] pad: direct==symbolic
    step 2 [input]: {shape=[n=1 c=3 h=6 w=3] pattern=pad_hw}
    [native] pad: direct==symbolic
    step 3 [input]: {shape=[n=2 c=3 h=6 w=3] pattern=pad_hw}
    [native] pad: direct==symbolic
    step 4 [pattern]: {shape=[n=2 c=3 h=6 w=3] pattern=reflect_asym_w}
    [native] pad: direct==symbolic
    step 5 [input]: {shape=[n=2 c=28 h=6 w=3] pattern=reflect_asym_w}
    [native] pad: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] axis=H start=0 stop=2 step=1}
    [native] slice: direct==symbolic
    step 1 [input]: {shape=[n=2 c=4 h=4 w=4] axis=H start=0 stop=2 step=1}
    [native] slice: direct==symbolic
    step 2 [input]: {shape=[n=2 c=16 h=4 w=4] axis=H start=0 stop=2 step=1}
    [native] slice: direct==symbolic
    step 3 [input]: {shape=[n=2 c=31 h=4 w=4] axis=H start=0 stop=2 step=1}
    [native] slice: direct==symbolic
    step 4 [axis]: {shape=[n=2 c=31 h=4 w=4] axis=H start=0 stop=2 step=1}
    [native] slice: direct==symbolic
    step 5 [pattern]: {shape=[n=2 c=31 h=4 w=4] axis=H start=0 stop=4 step=2}
    [native] slice: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] div: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=4]
    [native] div: direct==symbolic
    step 2 [input]: [n=1 c=3 h=4 w=5]
    [native] div: direct==symbolic
    step 3 [input]: [n=1 c=3 h=8 w=5]
    [native] div: direct==symbolic
    step 4 [input]: [n=1 c=3 h=16 w=5]
    [native] div: direct==symbolic
    step 5 [input]: [n=1 c=3 h=16 w=6]
    [native] div: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] sqrt: direct==symbolic
    step 1 [input]: [n=1 c=1 h=4 w=4]
    [native] sqrt: direct==symbolic
    step 2 [input]: [n=1 c=1 h=4 w=1]
    [native] sqrt: direct==symbolic
    step 3 [input]: [n=1 c=1 h=9 w=1]
    [native] sqrt: direct==symbolic
    step 4 [input]: [n=1 c=6 h=9 w=1]
    [native] sqrt: direct==symbolic
    step 5 [input]: [n=1 c=6 h=8 w=1]
    [native] sqrt: direct==symbolic
    step 0: {batch=1 n=2 m=3 p=4}
    [native] bmm: direct==symbolic
    step 1 [batch]: {batch=2 n=2 m=3 p=4}
    [native] bmm: direct==symbolic
    step 2 [p]: {batch=2 n=2 m=3 p=5}
    [native] bmm: direct==symbolic
    step 3 [p]: {batch=2 n=2 m=3 p=8}
    [native] bmm: direct==symbolic
    step 4 [m]: {batch=2 n=2 m=4 p=8}
    [native] bmm: direct==symbolic
    step 5 [p]: {batch=2 n=2 m=4 p=3}
    [native] bmm: direct==symbolic
    step 0: {shape=[1,4,8,8] output_size=[4,4]}
    [native] adaptive_avg_pool2d: direct==symbolic
    step 1 [c]: {shape=[1,2,8,8] output_size=[4,4]}
    [native] adaptive_avg_pool2d: direct==symbolic
    step 2 [out_h]: {shape=[1,2,8,8] output_size=[5,4]}
    [native] adaptive_avg_pool2d: direct==symbolic
    step 3 [input_w]: {shape=[1,2,8,5] output_size=[5,4]}
    [native] adaptive_avg_pool2d: direct==symbolic
    step 4 [c]: {shape=[1,6,8,5] output_size=[5,4]}
    [native] adaptive_avg_pool2d: direct==symbolic
    step 5 [input_w]: {shape=[1,6,8,3] output_size=[5,4]}
    [native] adaptive_avg_pool2d: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] scalar=3
    [native] pow: direct==symbolic
    step 1 [scalar]: [n=1 c=3 h=4 w=4] scalar=-2
    [native] pow: direct==symbolic
    step 2 [input]: [n=1 c=3 h=3 w=4] scalar=-2
    [native] pow: direct==symbolic
    step 3 [input]: [n=1 c=3 h=3 w=4] scalar=-2
    [native] pow: direct==symbolic
    step 4 [scalar]: [n=1 c=3 h=3 w=4] scalar=6
    [native] pow: direct==symbolic
    step 5 [input]: [n=1 c=3 h=5 w=4] scalar=6
    [native] pow: direct==symbolic |}]
