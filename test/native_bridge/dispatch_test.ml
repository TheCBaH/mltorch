(* Op_bridge.dispatch native compute, evaluated directly. Split from the former native_bridge_test.ml; promote with [dune promote test/native_bridge/dispatch_test.ml]. *)

open Helpers

let%expect_test "dispatch: div.Tensor elementwise" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = float_tensor [ 2; 3 ] [ 2.; 4.; 0.5; 1.; 10.; 3. ] in
  dispatch_print ~target:"torch.ops.aten.div.Tensor"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=3] {0.5, 0.5, 6, 4, 0.5, 2} |}]

let%expect_test "dispatch: mul.Tensor elementwise" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = float_tensor [ 2; 3 ] [ 0.; 10.; 100.; 1.; 2.; 3. ] in
  dispatch_print ~target:"torch.ops.aten.mul.Tensor"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=3] {0, 20, 300, 4, 10, 18} |}]

let%expect_test "dispatch: bmm 1x2x2 @ 1x2x2" =
  let a = float_tensor [ 1; 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  let b = float_tensor [ 1; 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  dispatch_print ~target:"torch.ops.aten.bmm.default"
    ~bindings:[ ("self", a); ("mat2", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "mat2" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=2] {7, 10, 15, 22} |}]

let%expect_test "dispatch: sqrt.default elementwise" =
  let a = float_tensor [ 2; 2 ] [ 0.; 1.; 4.; 2.25 ] in
  dispatch_print ~target:"torch.ops.aten.sqrt.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=2] {0, 1, 2, 1.5} |}]

let%expect_test "dispatch: sub.Tensor elementwise" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = float_tensor [ 2; 3 ] [ 0.; 10.; 100.; 1.; 2.; 3. ] in
  dispatch_print ~target:"torch.ops.aten.sub.Tensor"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=3] {1, -8, -97, 3, 3, 3} |}]

let%expect_test "dispatch: addmm.default relayouts [In,Out] weight" =
  let bias = float_tensor [ 2 ] [ 10.; 100. ] in
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let w = float_tensor [ 3; 2 ] [ 1.; 0.; 0.; 1.; 0.; 1. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.addmm.default"
    ~bindings:[ ("self", bias); ("mat1", x); ("mat2", w) ]
    ~inputs:[ in_tensor "self"; in_tensor "mat1"; in_tensor "mat2" ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs:
      [t0 f32 [C=2] ->[n1], t1 f32 [W=2 C=3] ->[n1], t2 f32 [W=3 C=2] ->[n0]]
    nodes:
      n0: [t3 f32 [N=2 T=1 D=1 H=1 W=1 C=3] ->[n1]] =
        permute x=t2 perm=[N<-C, W<-N, C<-W]
      n1: [t4 f32 [W=2 C=2]] =
        linear x=t1 weight=t3 <-n0 bias=t0 params={in_features=3}
    outputs: [t4 f32 [W=2 C=2] <-n1]
    tensor f32 [W=2 C=2] {11, 105, 14, 111} |}]

let%expect_test "dispatch: _native_batch_norm_legit_no_training per-channel" =
  (* NCHW [1,2,1,2]: c0 = [1,3], c1 = [5,7]. mean=[1,5], var=[4,4] (inv=0.5),
     weight=[2,10], bias=[1,-1], eps=0 -> y = (x-mean)*0.5*w+b: c0=[1,3],
     c1=[-1,9]. Only out0 is produced (the size-0 save_* outputs are dropped). *)
  let x = float_tensor [ 1; 2; 1; 2 ] [ 1.; 3.; 5.; 7. ] in
  let w = float_tensor [ 2 ] [ 2.; 10. ] in
  let b = float_tensor [ 2 ] [ 1.; -1. ] in
  let rm = float_tensor [ 2 ] [ 1.; 5. ] in
  let rv = float_tensor [ 2 ] [ 4.; 4. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten._native_batch_norm_legit_no_training.default"
    ~bindings:
      [
        ("input", x);
        ("weight", w);
        ("bias", b);
        ("running_mean", rm);
        ("running_var", rv);
      ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_tensor "bias";
        in_tensor "running_mean";
        in_tensor "running_var";
        in_float "momentum" 0.1;
        in_float "eps" 0.;
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs:
      [t0 f32 [H=2 W=1 C=2] ->[n0], t1 f32 [C=2] ->[n1], t2 f32 [C=2] ->[n1],
       t3 f32 [C=2] ->[n1], t4 f32 [C=2] ->[n1]]
    nodes:
      n0: [t5 f32 [W=2 C=2] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t6 f32 [W=2 C=2] ->[n2]] =
        batch_norm
          x=t5 <-n0
          weight=t1
          bias=t2
          running_mean=t3
          running_var=t4
          params={channel=C; eps=0}
      n2: [t7 f32 [H=2 W=1 C=2]] = permute x=t6 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t7 f32 [H=2 W=1 C=2] <-n2]
    tensor f32 [H=2 W=1 C=2] {1, 3, -1, 9} |}]

let%expect_test "dispatch: conv2d.default relayouts NCHW/OIHW with bias" =
  let x = float_tensor [ 1; 1; 3; 3 ] (List.init 9 float_of_int) in
  let w = float_tensor [ 1; 1; 2; 2 ] [ 1.; 1.; 1.; 1. ] in
  let bias = float_tensor [ 1 ] [ 10. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.conv2d.default"
    ~bindings:[ ("input", x); ("weight", w); ("bias", bias) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_tensor "bias";
        in_ints "stride" [ 1; 1 ];
        in_ints "padding" [ 0; 0 ];
        in_ints "dilation" [ 1; 1 ];
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs:
      [t0 f32 [W=3 C=3] ->[n0], t1 f32 [W=2 C=2] ->[n1], t2 f32 [C=1] ->[n2]]
    nodes:
      n0: [t3 f32 [H=3 W=3 C=1] ->[n2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t4 f32 [H=2 W=2 C=1] ->[n2]] =
        permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2: [t5 f32 [H=2 W=2 C=1] ->[n3]] =
        conv2d
          x=t3 <-n0
          weight=t4 <-n1
          bias=t2
          params={h={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=1;
                 groups=1}
      n3: [t6 f32 [W=2 C=2]] = permute x=t5 <-n2 perm=[H<-C, W<-H, C<-W]
    outputs: [t6 f32 [W=2 C=2] <-n3]
    tensor f32 [W=2 C=2] {18, 22, 30, 34} |}]

let%expect_test "dispatch: conv2d.padding same uses distinct native op" =
  let x = float_tensor [ 1; 1; 3; 3 ] (List.init 9 float_of_int) in
  let w = float_tensor [ 1; 1; 2; 2 ] [ 1.; 1.; 1.; 1. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.conv2d.padding"
    ~bindings:[ ("input", x); ("weight", w) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_none "bias";
        in_ints "stride" [ 1; 1 ];
        PT.NamedArgument.make "padding" (PT.Argument.String "same") None;
        in_ints "dilation" [ 1; 1 ];
        in_int "groups" 1;
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=3 C=3] ->[n0], t1 f32 [W=2 C=2] ->[n1]]
    nodes:
      n0: [t2 f32 [H=3 W=3 C=1] ->[n2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t3 f32 [H=2 W=2 C=1] ->[n2]] =
        permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2: [t4 f32 [H=3 W=3 C=1] ->[n3]] =
        conv2d_padding
          x=t2 <-n0
          weight=t3 <-n1
          bias=none
          params={stride={h=1; w=1}; padding=same; dilation={h=1; w=1}; groups=1}
      n3: [t5 f32 [W=3 C=3]] = permute x=t4 <-n2 perm=[H<-C, W<-H, C<-W]
    outputs: [t5 f32 [W=3 C=3] <-n3]
    tensor f32 [W=3 C=3] {8, 12, 7, 20, 24, 13, 13, 15, ...} |}]

let%expect_test "dispatch: conv2d.padding invalid weight rank is typed" =
  let x = float_tensor [ 1; 1; 3; 3 ] (List.init 9 float_of_int) in
  let w = float_tensor [ 1; 4 ] [ 1.; 2.; 3.; 4. ] in
  dispatch_print ~target:"torch.ops.aten.conv2d.padding"
    ~bindings:[ ("input", x); ("weight", w) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_none "bias";
        in_ints "stride" [ 1; 1 ];
        PT.NamedArgument.make "padding" (PT.Argument.String "same") None;
        in_ints "dilation" [ 1; 1 ];
        in_int "groups" 1;
      ]
    ~noutputs:1;
  [%expect {| error: conv2d.padding: weight must be rank-4, got shape [1, 4] |}]

let%expect_test "dispatch: convolution.default uses distinct native op" =
  let x = float_tensor [ 1; 1; 3; 3 ] (List.init 9 float_of_int) in
  let w = float_tensor [ 1; 1; 2; 2 ] [ 1.; 1.; 1.; 1. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.convolution.default"
    ~bindings:[ ("input", x); ("weight", w) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_none "bias";
        in_ints "stride" [ 1; 1 ];
        in_ints "padding" [ 0; 0 ];
        in_ints "dilation" [ 1; 1 ];
        in_bool "transposed" false;
        in_ints "output_padding" [ 0; 0 ];
        in_int "groups" 1;
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=3 C=3] ->[n0], t1 f32 [W=2 C=2] ->[n1]]
    nodes:
      n0: [t2 f32 [H=3 W=3 C=1] ->[n2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t3 f32 [H=2 W=2 C=1] ->[n2]] =
        permute x=t1 perm=[N<-D, D<-N, H<-W, W<-C, C<-H]
      n2: [t4 f32 [H=2 W=2 C=1] ->[n3]] =
        convolution
          x=t2 <-n0
          weight=t3 <-n1
          bias=none
          params={stride={h=1; w=1};
                 padding={h=0; w=0};
                 dilation={h=1; w=1};
                 transposed=false;
                 output_padding={h=0; w=0};
                 groups=1}
      n3: [t5 f32 [W=2 C=2]] = permute x=t4 <-n2 perm=[H<-C, W<-H, C<-W]
    outputs: [t5 f32 [W=2 C=2] <-n3]
    tensor f32 [W=2 C=2] {8, 12, 20, 24} |}]

let%expect_test "dispatch: convolution.default grouped conv2d" =
  let x = float_tensor [ 1; 4; 1; 1 ] [ 1.; 2.; 10.; 20. ] in
  let w = float_tensor [ 4; 2; 1; 1 ] [ 1.; 1.; 10.; 0.; 1.; 1.; 0.; 2. ] in
  let bias = float_tensor [ 4 ] [ 0.; 100.; 1000.; 10000. ] in
  dispatch_print ~target:"torch.ops.aten.convolution.default"
    ~bindings:[ ("input", x); ("weight", w); ("bias", bias) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_tensor "bias";
        in_ints "stride" [ 1; 1 ];
        in_ints "padding" [ 0; 0 ];
        in_ints "dilation" [ 1; 1 ];
        in_bool "transposed" false;
        in_ints "output_padding" [ 0; 0 ];
        in_int "groups" 2;
      ]
    ~noutputs:1;
  [%expect {| tensor f32 [H=4 W=1 C=1] {3, 110, 1030, 10040} |}]

let%expect_test "dispatch: convolution.default transposed" =
  let x = float_tensor [ 1; 1; 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  let w = float_tensor [ 1; 1; 2; 2 ] [ 1.; 1.; 1.; 1. ] in
  dispatch_print ~target:"torch.ops.aten.convolution.default"
    ~bindings:[ ("input", x); ("weight", w) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_none "bias";
        in_ints "stride" [ 1; 1 ];
        in_ints "padding" [ 0; 0 ];
        in_ints "dilation" [ 1; 1 ];
        in_bool "transposed" true;
        in_ints "output_padding" [ 0; 0 ];
        in_int "groups" 1;
      ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=3 C=3] {1, 3, 2, 4, 10, 6, 3, 7, ...} |}]

let%expect_test "dispatch: conv2d.default dilated spatial window" =
  let x = float_tensor [ 1; 1; 1; 5 ] [ 0.; 1.; 2.; 3.; 4. ] in
  let w = float_tensor [ 1; 1; 1; 3 ] [ 1.; 1.; 1. ] in
  dispatch_print ~target:"torch.ops.aten.conv2d.default"
    ~bindings:[ ("input", x); ("weight", w) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_none "bias";
        in_ints "stride" [ 1; 1 ];
        in_ints "padding" [ 0; 1 ];
        in_ints "dilation" [ 1; 2 ];
      ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=3] {4, 6, 4} |}]

let%expect_test
    "dispatch: conv2d.default combines bias stride padding dilation and groups"
    =
  let x =
    float_tensor [ 1; 2; 1; 7 ]
      [ 0.; 1.; 2.; 3.; 4.; 5.; 6.; 10.; 11.; 12.; 13.; 14.; 15.; 16. ]
  in
  let w = float_tensor [ 2; 1; 1; 2 ] [ 1.; 1.; 2.; 3. ] in
  let bias = float_tensor [ 2 ] [ 10.; 100. ] in
  dispatch_print ~target:"torch.ops.aten.conv2d.default"
    ~bindings:[ ("input", x); ("weight", w); ("bias", bias) ]
    ~inputs:
      [
        in_tensor "input";
        in_tensor "weight";
        in_tensor "bias";
        in_ints "stride" [ 1; 2 ];
        in_ints "padding" [ 0; 1 ];
        in_ints "dilation" [ 1; 2 ];
        in_int "groups" 2;
      ]
    ~noutputs:1;
  [%expect {| tensor f32 [H=2 W=1 C=4] {11, 14, 18, 15, 133, 161, 171, 130} |}]

let%expect_test "dispatch: linear.default relayouts [Out,In] weight with bias" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let w = float_tensor [ 2; 3 ] [ 1.; 0.; 0.; 0.; 1.; 1. ] in
  let bias = float_tensor [ 2 ] [ 10.; 100. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.linear.default"
    ~bindings:[ ("input", x); ("weight", w); ("bias", bias) ]
    ~inputs:[ in_tensor "input"; in_tensor "weight"; in_tensor "bias" ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs:
      [t0 f32 [W=2 C=3] ->[n1], t1 f32 [W=2 C=3] ->[n0], t2 f32 [C=2] ->[n1]]
    nodes:
      n0: [t3 f32 [N=2 T=1 D=1 H=1 W=1 C=3] ->[n1]] =
        permute x=t1 perm=[N<-W, W<-N]
      n1: [t4 f32 [W=2 C=2]] =
        linear x=t0 weight=t3 <-n0 bias=t2 params={in_features=3}
    outputs: [t4 f32 [W=2 C=2] <-n1]
    tensor f32 [W=2 C=2] {11, 105, 14, 111} |}]

let%expect_test "dispatch: linear.default accepts explicit None bias" =
  let x = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let w = float_tensor [ 2; 3 ] [ 1.; 0.; 0.; 0.; 1.; 1. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.linear.default"
    ~bindings:[ ("input", x); ("weight", w) ]
    ~inputs:[ in_tensor "input"; in_tensor "weight"; in_none "bias" ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n1], t1 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t2 f32 [N=2 T=1 D=1 H=1 W=1 C=3] ->[n1]] =
        permute x=t1 perm=[N<-W, W<-N]
      n1: [t3 f32 [W=2 C=2]] =
        linear x=t0 weight=t2 <-n0 bias=none params={in_features=3}
    outputs: [t3 f32 [W=2 C=2] <-n1]
    tensor f32 [W=2 C=2] {1, 5, 4, 11} |}]

let%expect_test "dispatch: max_pool2d.default relayouts NCHW input and output" =
  let x =
    float_tensor [ 1; 1; 3; 3 ] (List.init 9 (fun i -> float_of_int (-(i + 1))))
  in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.max_pool2d.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [
        in_tensor "self";
        in_ints "kernel_size" [ 2; 2 ];
        in_ints "stride" [ 1; 1 ];
        in_ints "padding" [ 1; 1 ];
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=3 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [H=3 W=3 C=1] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t2 f32 [H=4 W=4 C=1] ->[n2]] =
        max_pool2d
          x=t1 <-n0
          params={kernel={h=2; w=2};
                 stride={h=1; w=1};
                 pad={h=1; w=1};
                 ceil_mode=false}
      n2: [t3 f32 [W=4 C=4]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [W=4 C=4] <-n2]
    tensor f32 [W=4 C=4] {-1, -1, -2, -3, -1, -1, -2, -3, ...} |}]

let%expect_test
    "dispatch: adaptive_avg_pool2d.default retains output_size through relayout"
    =
  let x =
    float_tensor [ 1; 1; 4; 4 ] (List.init 16 (fun i -> float_of_int (i + 1)))
  in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.adaptive_avg_pool2d.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "output_size" [ 1; 1 ] ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=4 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=4 W=4 C=1] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t2 f32 [C=1] ->[n2]] =
        adaptive_avg_pool2d x=t1 <-n0 params={output_size={h=1; w=1}}
      n2: [t3 f32 [C=1]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [C=1] <-n2]
    tensor f32 [C=1] {8.5} |}]

let%expect_test
    "dispatch: upsample_bilinear2d.vec explicit output_size, align_corners=true"
    =
  let x = float_tensor [ 1; 1; 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.upsample_bilinear2d.vec"
    ~bindings:[ ("input", x) ]
    ~inputs:
      [
        in_tensor "input";
        in_ints "output_size" [ 3; 3 ];
        in_bool "align_corners" true;
        in_none "scale_factors";
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=2] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=2 C=1] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t2 f32 [H=3 W=3 C=1] ->[n2]] =
        upsample_bilinear2d
          x=t1 <-n0
          params={output_size={h=3; w=3};
          align_corners=true}
      n2: [t3 f32 [W=3 C=3]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [W=3 C=3] <-n2]
    tensor f32 [W=3 C=3] {1, 1.5, 2, 2, 2.5, 3, 3, 3.5, ...} |}]

(* Same [2x2 -> 3x3] geometry, but resolved from [scale_factors] rather than
   an explicit [output_size] -- the import-time resolution [Op_bridge] and
   [Native_interp] both perform (ATen's own `floor(input_size *
   scale_factor)`), so the Native op sees the identical params either way.
   [align_corners=false] here, so this also exercises the other
   [Bilinear_axis.endpoints] branch. *)
let%expect_test "dispatch: upsample_bilinear2d.vec via scale_factors" =
  let x = float_tensor [ 1; 1; 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.upsample_bilinear2d.vec"
    ~bindings:[ ("input", x) ]
    ~inputs:
      [
        in_tensor "input";
        in_none "output_size";
        in_bool "align_corners" false;
        in_floats "scale_factors" [ 1.5; 1.5 ];
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=2] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=2 C=1] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t2 f32 [H=3 W=3 C=1] ->[n2]] =
        upsample_bilinear2d
          x=t1 <-n0
          params={output_size={h=3; w=3};
          align_corners=false}
      n2: [t3 f32 [W=3 C=3]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [W=3 C=3] <-n2]
    tensor f32 [W=3 C=3] {1, 1.5, 2, 2, 2.5, 3, 3, 3.5, ...} |}]

let%expect_test "dispatch: upsample_bilinear2d.vec rejects neither size arg" =
  let x = float_tensor [ 1; 1; 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  dispatch_print ~target:"torch.ops.aten.upsample_bilinear2d.vec"
    ~bindings:[ ("input", x) ]
    ~inputs:
      [
        in_tensor "input";
        in_none "output_size";
        in_bool "align_corners" true;
        in_none "scale_factors";
      ]
    ~noutputs:1;
  [%expect
    {|
    error: upsample_bilinear2d.vec: exactly one of output_size or scale_factors must be given |}]

let%expect_test "dispatch: upsample_bilinear2d.vec rejects both size args" =
  let x = float_tensor [ 1; 1; 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  dispatch_print ~target:"torch.ops.aten.upsample_bilinear2d.vec"
    ~bindings:[ ("input", x) ]
    ~inputs:
      [
        in_tensor "input";
        in_ints "output_size" [ 3; 3 ];
        in_bool "align_corners" true;
        in_floats "scale_factors" [ 1.5; 1.5 ];
      ]
    ~noutputs:1;
  [%expect
    {|
    error: upsample_bilinear2d.vec: output_size and scale_factors are mutually exclusive |}]

let%expect_test "dispatch: max_pool2d_with_indices.default discards indices" =
  (* NCHW [1,1,4,4], value(h,w)=h*4+w. 2x2/stride-2 windows: max is each
     window's bottom-right; the graph output is the relayout'd values, and the
     dead indices edge is routed into a Discard node. *)
  let x = float_tensor [ 1; 1; 4; 4 ] (List.init 16 float_of_int) in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.max_pool2d_with_indices.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [
        in_tensor "self";
        in_ints "kernel_size" [ 2; 2 ];
        in_ints "stride" [ 2; 2 ];
        in_ints "padding" [ 0; 0 ];
      ]
    ~noutputs:2;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=4 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=4 W=4 C=1] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t2 f32 [H=2 W=2 C=1] ->[n3], t3 f32 [H=2 W=2 C=1] ->[n2]] =
        max_pool2d_with_indices
          x=t1 <-n0
          params={kernel={h=2; w=2};
                 stride={h=2; w=2};
                 pad={h=0; w=0};
                 ceil_mode=false}
      n2: [] = discard x=t3 <-n1
      n3: [t4 f32 [W=2 C=2]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t4 f32 [W=2 C=2] <-n3]
    tensor f32 [W=2 C=2] {5, 7, 13, 15} |}]

let%expect_test "dispatch: mean.dim dim=[1] keepdim=true" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.mean.dim"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ]; in_bool "keepdim" true ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=1] {1, 4} |}]

let%expect_test "dispatch: mean.dim dim=[1] keepdim=false" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.mean.dim"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ]; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=2] {1, 4} |}]

let%expect_test "dispatch: mean.dim dim=[] reduces over all dims" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.mean.dim"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" []; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=1] {2.5} |}]

let%expect_test "dispatch: mean.dim omitted dim reduces over all dims" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.mean.dim"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=1] {2.5} |}]

(* [Aten_shape.axis_of_dim] asserts its range and raises; before commit 0 this
   escaped [Op_bridge.dispatch] as an uncaught [Invalid_argument] rather than
   the typed row every other bad-argument arm returns. *)
let%expect_test "dispatch: mean.dim rejects an out-of-range dim" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  List.iter
    (fun d ->
      dispatch_print ~target:"torch.ops.aten.mean.dim"
        ~bindings:[ ("self", x) ]
        ~inputs:
          [ in_tensor "self"; in_ints "dim" [ d ]; in_bool "keepdim" false ]
        ~noutputs:1)
    [ 7; -3 ];
  [%expect
    {|
    error: mean.dim: invalid dimension 7 for rank 2
    error: mean.dim: invalid dimension -3 for rank 2 |}]

let%expect_test "dispatch: amax.default dim=[1] keepdim=true" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.amax.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ]; in_bool "keepdim" true ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=1] {2, 5} |}]

let%expect_test "dispatch: amax.default dim=[1] keepdim=false" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.amax.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ]; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=2] {2, 5} |}]

let%expect_test "dispatch: amax.default dim=[] reduces over all dims" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.amax.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" []; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=1] {5} |}]

let%expect_test "dispatch: amax.default omitted dim reduces over all dims" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.amax.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=1] {5} |}]

let%expect_test "dispatch: amax.default rejects an out-of-range dim" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  List.iter
    (fun d ->
      dispatch_print ~target:"torch.ops.aten.amax.default"
        ~bindings:[ ("self", x) ]
        ~inputs:
          [ in_tensor "self"; in_ints "dim" [ d ]; in_bool "keepdim" false ]
        ~noutputs:1)
    [ 7; -3 ];
  [%expect
    {|
    error: amax.default: invalid dimension 7 for rank 2
    error: amax.default: invalid dimension -3 for rank 2 |}]

let%expect_test "dispatch: pow.Tensor_Scalar exponent=0.5 computes sqrt" =
  let x = float_tensor [ 4 ] [ 0.; 1.; 4.; 9. ] in
  dispatch_print ~target:"torch.ops.aten.pow.Tensor_Scalar"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_float "exponent" 0.5 ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=4] {0, 1, 2, 3} |}]

(* [x] excludes 0: -1/-2/-0.5 are singular there, and every exponent here is
   exercised on the same three values so the table is easy to eyeball against
   [Reduce]'s hand computation. *)
let%expect_test
    "dispatch: pow.Tensor_Scalar over ATen's special-cased exponents" =
  let x = float_tensor [ 3 ] [ 1.; 4.; 9. ] in
  List.iter
    (fun exponent ->
      dispatch_print ~target:"torch.ops.aten.pow.Tensor_Scalar"
        ~bindings:[ ("self", x) ]
        ~inputs:[ in_tensor "self"; in_int "exponent" exponent ]
        ~noutputs:1)
    [ 2; 3; -1; -2 ];
  List.iter
    (fun exponent ->
      dispatch_print ~target:"torch.ops.aten.pow.Tensor_Scalar"
        ~bindings:[ ("self", x) ]
        ~inputs:[ in_tensor "self"; in_float "exponent" exponent ]
        ~noutputs:1)
    [ 0.5; -0.5 ];
  [%expect
    {|
    tensor f32 [C=3] {1, 16, 81}
    tensor f32 [C=3] {1, 64, 729}
    tensor f32 [C=3] {1, 0.25, 0.111111}
    tensor f32 [C=3] {1, 0.0625, 0.0123457}
    tensor f32 [C=3] {1, 2, 3}
    tensor f32 [C=3] {1, 0.5, 0.333333} |}]

(* A generic exponent falls back to [exp(exponent * log x)]: 1.5 on perfect
   squares is exact in real arithmetic (x^1.5 = x * sqrt(x)), so a mismatch
   here would be visible rather than lost in the fallback's own imprecision. *)
let%expect_test
    "dispatch: pow.Tensor_Scalar generic exponent falls back to exp/log" =
  let x = float_tensor [ 3 ] [ 1.; 4.; 9. ] in
  dispatch_print ~target:"torch.ops.aten.pow.Tensor_Scalar"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_float "exponent" 1.5 ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=3] {1, 8, 27} |}]

let%expect_test "dispatch: linalg_vector_norm.default dim=[1] keepdim=true" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.linalg_vector_norm.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ]; in_bool "keepdim" true ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=1] {2.23607, 7.07107} |}]

let%expect_test "dispatch: linalg_vector_norm.default dim=[1] keepdim=false" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.linalg_vector_norm.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "dim" [ 1 ]; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=2] {2.23607, 7.07107} |}]

let%expect_test
    "dispatch: linalg_vector_norm.default omitted dim/ord reduces over all dims"
    =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.linalg_vector_norm.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_bool "keepdim" false ]
    ~noutputs:1;
  [%expect {| tensor f32 [C=1] {7.4162} |}]

let%expect_test "dispatch: linalg_vector_norm.default rejects a non-2 ord" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.linalg_vector_norm.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [
        in_tensor "self";
        in_float "ord" 1.0;
        in_ints "dim" [ 1 ];
        in_bool "keepdim" false;
      ]
    ~noutputs:1;
  [%expect
    {| error: linalg_vector_norm.default: only ord=2 is supported, got 1 |}]

let%expect_test "dispatch: linalg_vector_norm.default rejects a supplied dtype"
    =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.linalg_vector_norm.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [
        in_tensor "self";
        in_ints "dim" [ 1 ];
        in_bool "keepdim" false;
        PT.NamedArgument.make "dtype"
          (PT.Argument.Scalar_type PT.ScalarType.DOUBLE) None;
      ]
    ~noutputs:1;
  [%expect {| error: unsupported scalar_type argument "dtype" |}]
