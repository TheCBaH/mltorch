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
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0}
    [native] max_pool2d: direct==symbolic
    step 1 [kernel]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=2x2 pad=0x0}
    [native] max_pool2d: direct==symbolic
    step 2 [stride]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=1x1 pad=0x0}
    [native] max_pool2d: direct==symbolic
    step 3 [input]: {shape=[n=1 c=4 h=3 w=8] kernel=3x4 stride=1x1 pad=0x0}
    [native] max_pool2d: direct==symbolic
    step 4 [stride]: {shape=[n=1 c=4 h=3 w=8] kernel=3x4 stride=2x3 pad=0x0}
    [native] max_pool2d: direct==symbolic
    step 5 [stride]: {shape=[n=1 c=4 h=3 w=8] kernel=3x4 stride=3x3 pad=0x0}
    [native] max_pool2d: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0}
    [native] avg_pool2d: direct==symbolic
    step 1 [stride]: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x1 pad=0x0}
    [native] avg_pool2d: direct==symbolic
    step 2 [pad]: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x1 pad=1x1}
    [native] avg_pool2d: direct==symbolic
    step 3 [input]: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x1 pad=1x1}
    [native] avg_pool2d: direct==symbolic
    step 4 [stride]: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=3x3 pad=1x1}
    [native] avg_pool2d: direct==symbolic
    step 5 [stride]: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=3x3 pad=1x1}
    [native] avg_pool2d: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=8}
    [native] conv2d: direct==symbolic
    step 1 [kernel]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=8}
    [native] conv2d: direct==symbolic
    step 2 [pad]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=1x1 pad=2x0 dilation=1x1 groups=1 out_c=8}
    [native] conv2d: direct==symbolic
    step 3 [out_channels]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=1x1 pad=2x0 dilation=1x1 groups=1 out_c=21}
    [native] conv2d: direct==symbolic
    step 4 [input]: {shape=[n=1 c=4 h=8 w=6] kernel=3x4 stride=1x1 pad=2x0 dilation=1x1 groups=1 out_c=21}
    [native] conv2d: direct==symbolic
    step 5 [groups]: {shape=[n=1 c=4 h=8 w=6] kernel=3x4 stride=1x1 pad=2x0 dilation=1x1 groups=2 out_c=22}
    [native] conv2d: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=8}
    [native] convolution: direct==symbolic
    step 1 [out_channels]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=16}
    [native] convolution: direct==symbolic
    step 2 [kernel]: {shape=[n=1 c=4 h=8 w=8] kernel=4x2 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=16}
    [native] convolution: direct==symbolic
    step 3 [out_channels]: {shape=[n=1 c=4 h=8 w=8] kernel=4x2 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=3}
    [native] convolution: direct==symbolic
    step 4 [input]: {shape=[n=2 c=4 h=8 w=8] kernel=4x2 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=3}
    [native] convolution: direct==symbolic
    step 5 [out_channels]: {shape=[n=2 c=4 h=8 w=8] kernel=4x2 stride=1x1 pad=1x1 dilation=1x1 groups=1 out_c=21}
    [native] convolution: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=1x1 groups=1 out_c=8 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 1 [out_channels]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=1x1 dilation=1x1 groups=1 out_c=30 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 2 [input]: {shape=[n=1 c=4 h=15 w=8] kernel=3x3 stride=1x1 dilation=1x1 groups=1 out_c=30 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 3 [stride]: {shape=[n=1 c=4 h=15 w=8] kernel=3x3 stride=1x1 dilation=1x1 groups=1 out_c=30 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 4 [input]: {shape=[n=1 c=4 h=15 w=8] kernel=3x3 stride=1x1 dilation=1x1 groups=1 out_c=30 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 5 [input]: {shape=[n=1 c=4 h=15 w=16] kernel=3x3 stride=1x1 dilation=1x1 groups=1 out_c=30 padding=same}
    [native] conv2d_padding: direct==symbolic
    step 0: {shape=[n=1 c=8 h=1 w=4] out_features=6 bias=true}
    [native] linear: direct==symbolic
    step 1 [bias]: {shape=[n=1 c=8 h=1 w=4] out_features=6 bias=true}
    [native] linear: direct==symbolic
    step 2 [bias]: {shape=[n=1 c=8 h=1 w=4] out_features=6 bias=false}
    [native] linear: direct==symbolic
    step 3 [bias]: {shape=[n=1 c=8 h=1 w=4] out_features=6 bias=true}
    [native] linear: direct==symbolic
    step 4 [bias]: {shape=[n=1 c=8 h=1 w=4] out_features=6 bias=false}
    [native] linear: direct==symbolic
    step 5 [input]: {shape=[n=1 c=8 h=1 w=4] out_features=6 bias=false}
    [native] linear: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] eps=1e-05}
    [native] batch_norm: direct==symbolic
    step 1 [eps]: {shape=[n=2 c=4 h=4 w=4] eps=0.001}
    [native] batch_norm: direct==symbolic
    step 2 [eps]: {shape=[n=2 c=4 h=4 w=4] eps=0}
    [native] batch_norm: direct==symbolic
    step 3 [eps]: {shape=[n=2 c=4 h=4 w=4] eps=0.001}
    [native] batch_norm: direct==symbolic
    step 4 [input]: {shape=[n=2 c=4 h=4 w=4] eps=0.001}
    [native] batch_norm: direct==symbolic
    step 5 [eps]: {shape=[n=2 c=4 h=4 w=4] eps=0.001}
    [native] batch_norm: direct==symbolic
    step 0: {shape=[n=1 c=5 h=4 w=3] k=1 eps=1e-05 weight=true}
    [native] rms_norm: direct==symbolic
    step 1 [k]: {shape=[n=1 c=5 h=4 w=3] k=2 eps=1e-05 weight=true}
    [native] rms_norm: direct==symbolic
    step 2 [eps]: {shape=[n=1 c=5 h=4 w=3] k=2 eps=0 weight=true}
    [native] rms_norm: direct==symbolic
    step 3 [k]: {shape=[n=1 c=5 h=4 w=3] k=1 eps=0 weight=true}
    [native] rms_norm: direct==symbolic
    step 4 [k]: {shape=[n=1 c=5 h=4 w=3] k=3 eps=0 weight=true}
    [native] rms_norm: direct==symbolic
    step 5 [input]: {shape=[n=1 c=5 h=15 w=3] k=3 eps=0 weight=true}
    [native] rms_norm: direct==symbolic
    step 0: {shape=[n=1 c=4 h=8 w=8] kernel=2x2 stride=2x2 pad=0x0}
    [native] max_pool2d_with_indices: direct==symbolic
    step 1 [kernel]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=2x2 pad=0x0}
    [native] max_pool2d_with_indices: direct==symbolic
    step 2 [pad]: {shape=[n=1 c=4 h=8 w=8] kernel=3x4 stride=2x2 pad=1x1}
    [native] max_pool2d_with_indices: direct==symbolic
    step 3 [kernel]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=2x2 pad=1x1}
    [native] max_pool2d_with_indices: direct==symbolic
    step 4 [kernel]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=2x2 pad=1x1}
    [native] max_pool2d_with_indices: direct==symbolic
    step 5 [pad]: {shape=[n=1 c=4 h=8 w=8] kernel=3x3 stride=2x2 pad=0x1}
    [native] max_pool2d_with_indices: direct==symbolic
    step 0: {shape=[n=1 c=4 h=4 w=4] -> flat}
    [native] reshape: direct==symbolic
    step 1 [input]: {shape=[n=1 c=4 h=4 w=2] -> flat}
    [native] reshape: direct==symbolic
    step 2 [input]: {shape=[n=1 c=5 h=4 w=2] -> flat}
    [native] reshape: direct==symbolic
    step 3 [input]: {shape=[n=1 c=5 h=4 w=2] -> flat}
    [native] reshape: direct==symbolic
    step 4 [input]: {shape=[n=2 c=5 h=4 w=2] -> flat}
    [native] reshape: direct==symbolic
    step 5 [input]: {shape=[n=2 c=5 h=8 w=2] -> flat}
    [native] reshape: direct==symbolic
    step 0: {shape=[n=1 c=4 h=4 w=4] perm=[H<-W, W<-H]}
    [native] permute: direct==symbolic
    step 1 [input]: {shape=[n=1 c=4 h=1 w=4] perm=[H<-W, W<-H]}
    [native] permute: direct==symbolic
    step 2 [input]: {shape=[n=1 c=4 h=3 w=4] perm=[H<-W, W<-H]}
    [native] permute: direct==symbolic
    step 3 [input]: {shape=[n=1 c=1 h=3 w=4] perm=[H<-W, W<-H]}
    [native] permute: direct==symbolic
    step 4 [input]: {shape=[n=1 c=1 h=15 w=4] perm=[H<-W, W<-H]}
    [native] permute: direct==symbolic
    step 5 [perm]: {shape=[n=1 c=1 h=15 w=4] perm=[N<-C, C<-N]}
    [native] permute: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] {min=0; max=6}
    [native] clamp: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=6] {min=0; max=6}
    [native] clamp: direct==symbolic
    step 2 [bounds]: [n=1 c=3 h=4 w=6] {min=1; max=-1}
    [native] clamp: direct==symbolic
    step 3 [bounds]: [n=1 c=3 h=4 w=6] {min=0; max=6}
    [native] clamp: direct==symbolic
    step 4 [input]: [n=1 c=3 h=4 w=6] {min=0; max=6}
    [native] clamp: direct==symbolic
    step 5 [bounds]: [n=1 c=3 h=4 w=6] {min=0; max=nan}
    [native] clamp: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] clone: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=8]
    [native] clone: direct==symbolic
    step 2 [input]: [n=1 c=15 h=4 w=8]
    [native] clone: direct==symbolic
    step 3 [input]: [n=1 c=21 h=4 w=8]
    [native] clone: direct==symbolic
    step 4 [input]: [n=1 c=21 h=6 w=8]
    [native] clone: direct==symbolic
    step 5 [input]: [n=1 c=21 h=6 w=8]
    [native] clone: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] {min_val=0; max_val=6}
    [native] hardtanh: direct==symbolic
    step 1 [bounds]: [n=1 c=3 h=4 w=4] {min_val=-1; max_val=1}
    [native] hardtanh: direct==symbolic
    step 2 [bounds]: [n=1 c=3 h=4 w=4] {min_val=1; max_val=-1}
    [native] hardtanh: direct==symbolic
    step 3 [input]: [n=1 c=3 h=4 w=15] {min_val=1; max_val=-1}
    [native] hardtanh: direct==symbolic
    step 4 [input]: [n=1 c=3 h=16 w=15] {min_val=1; max_val=-1}
    [native] hardtanh: direct==symbolic
    step 5 [input]: [n=2 c=3 h=16 w=15] {min_val=1; max_val=-1}
    [native] hardtanh: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] scalar=3
    [native] add_scalar: direct==symbolic
    step 1 [input]: [n=1 c=3 h=4 w=16] scalar=3
    [native] add_scalar: direct==symbolic
    step 2 [input]: [n=1 c=3 h=4 w=1] scalar=3
    [native] add_scalar: direct==symbolic
    step 3 [input]: [n=1 c=3 h=4 w=14] scalar=3
    [native] add_scalar: direct==symbolic
    step 4 [scalar]: [n=1 c=3 h=4 w=14] scalar=3
    [native] add_scalar: direct==symbolic
    step 5 [input]: [n=1 c=12 h=4 w=14] scalar=3
    [native] add_scalar: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4] scalar=3
    [native] div_scalar: direct==symbolic
    step 1 [input]: [n=1 c=6 h=4 w=4] scalar=3
    [native] div_scalar: direct==symbolic
    step 2 [input]: [n=1 c=12 h=4 w=4] scalar=3
    [native] div_scalar: direct==symbolic
    step 3 [input]: [n=1 c=12 h=4 w=5] scalar=3
    [native] div_scalar: direct==symbolic
    step 4 [scalar]: [n=1 c=12 h=4 w=5] scalar=-2
    [native] div_scalar: direct==symbolic
    step 5 [scalar]: [n=1 c=12 h=4 w=5] scalar=0.5
    [native] div_scalar: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] axis=H}
    [native] unbind: direct==symbolic
    step 1 [input]: {shape=[n=2 c=4 h=4 w=13] axis=H}
    [native] unbind: direct==symbolic
    step 2 [axis]: {shape=[n=2 c=4 h=4 w=13] axis=W}
    [native] unbind: direct==symbolic
    step 3 [axis]: {shape=[n=2 c=4 h=4 w=13] axis=N}
    [native] unbind: direct==symbolic
    step 4 [axis]: {shape=[n=2 c=4 h=4 w=13] axis=H}
    [native] unbind: direct==symbolic
    step 5 [axis]: {shape=[n=2 c=4 h=4 w=13] axis=N}
    [native] unbind: direct==symbolic
    step 0: [n=1 c=3 h=4 w=4]
    [native] silu: direct==symbolic
    step 1 [input]: [n=2 c=3 h=4 w=4]
    [native] silu: direct==symbolic
    step 2 [input]: [n=2 c=3 h=4 w=7]
    [native] silu: direct==symbolic
    step 3 [input]: [n=2 c=3 h=4 w=6]
    [native] silu: direct==symbolic
    step 4 [input]: [n=2 c=3 h=15 w=6]
    [native] silu: direct==symbolic
    step 5 [input]: [n=1 c=3 h=15 w=6]
    [native] silu: direct==symbolic
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
    step 0: {shape=[n=1 c=3 h=6 w=6] pattern=pad_hw}
    [native] pad: direct==symbolic
    step 1 [pattern]: {shape=[n=1 c=3 h=6 w=6] pattern=reflect_asym_w}
    [native] pad: direct==symbolic
    step 2 [pattern]: {shape=[n=1 c=3 h=6 w=6] pattern=pad_hw}
    [native] pad: direct==symbolic
    step 3 [pattern]: {shape=[n=1 c=3 h=6 w=6] pattern=reflect_hw}
    [native] pad: direct==symbolic
    step 4 [input]: {shape=[n=1 c=3 h=6 w=6] pattern=reflect_hw}
    [native] pad: direct==symbolic
    step 5 [pattern]: {shape=[n=1 c=3 h=6 w=6] pattern=pad_hw}
    [native] pad: direct==symbolic
    step 0: {shape=[n=2 c=4 h=4 w=4] axis=H start=0 stop=2 step=1}
    [native] slice: direct==symbolic
    step 1 [input]: {shape=[n=2 c=4 h=4 w=13] axis=H start=0 stop=2 step=1}
    [native] slice: direct==symbolic
    step 2 [pattern]: {shape=[n=2 c=4 h=4 w=13] axis=H start=2 stop=4 step=1}
    [native] slice: direct==symbolic
    step 3 [input]: {shape=[n=2 c=4 h=4 w=12] axis=H start=2 stop=4 step=1}
    [native] slice: direct==symbolic
    step 4 [axis]: {shape=[n=2 c=4 h=4 w=12] axis=W start=6 stop=12 step=1}
    [native] slice: direct==symbolic
    step 5 [input]: {shape=[n=2 c=14 h=4 w=12] axis=W start=6 stop=12 step=1}
    [native] slice: direct==symbolic |}]
