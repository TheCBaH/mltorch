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

let%expect_test "dispatch: matmul.default 2D @ 2D (batch-less)" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = float_tensor [ 3; 2 ] [ 1.; 0.; 0.; 1.; 1.; 1. ] in
  dispatch_print ~target:"torch.ops.aten.matmul.default"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=2] {4, 5, 10, 11} |}]

(* Confirms the design claim, not just the value: [Tensor_bridge.of_aten]'s
   own right-alignment already lands a rank-2 operand's data exactly where
   [Bmm] reads it, so the arm needs no relayout permute around the op --
   unlike [addmm.default]/[linear.default] above, whose graphs each need one.
   Same values as the previous test. *)
let%expect_test
    "dispatch: matmul.default binds to the existing Bmm node, no relayout" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = float_tensor [ 3; 2 ] [ 1.; 0.; 0.; 1.; 1.; 1. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.matmul.default"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0], t1 f32 [W=3 C=2] ->[n0]]
    nodes:
      n0: [t2 f32 [W=2 C=2]] = bmm input=t0 mat2=t1
    outputs: [t2 f32 [W=2 C=2] <-n0]
    tensor f32 [W=2 C=2] {4, 5, 10, 11} |}]

(* Rank>=3 with every leading axis at extent 1 is the SAME shape family, not
   a separate case: the extra unit axes land on [N]/[D] (never read by
   [Bmm]), so this must produce identical values to the rank-2 test above. *)
let%expect_test
    "dispatch: matmul.default accepts rank>=3 with unit leading axes" =
  let a = float_tensor [ 1; 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = float_tensor [ 1; 3; 2 ] [ 1.; 0.; 0.; 1.; 1.; 1. ] in
  dispatch_print ~target:"torch.ops.aten.matmul.default"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=2] {4, 5, 10, 11} |}]

(* The batched/multi-head shape family (`D`/`H` > 1) is explicitly out of
   scope (`.ai/matmul_softmax_design.md` §5) -- a typed rejection naming both
   actual shapes, not a wrong answer or a bare "unsupported". *)
let%expect_test "dispatch: matmul.default rejects a batched (>1) shape" =
  let a = float_tensor [ 2; 2; 3 ] (List.init 12 float_of_int) in
  let b = float_tensor [ 2; 3; 2 ] (List.init 12 float_of_int) in
  dispatch_print ~target:"torch.ops.aten.matmul.default"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ]
    ~noutputs:1;
  [%expect
    {| error: matmul.default: only rank-2-by-rank-2, or rank>=3 with every axis but the last two at extent 1, is supported, got self=[2, 2, 3] other=[2, 3, 2] |}]

let%expect_test "dispatch: sqrt.default elementwise" =
  let a = float_tensor [ 2; 2 ] [ 0.; 1.; 4.; 2.25 ] in
  dispatch_print ~target:"torch.ops.aten.sqrt.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=2] {0, 1, 2, 1.5} |}]

(* rsqrt(x) = x ** -0.5 legalizes onto the existing [Pow] node -- see the
   arm's comment in op_bridge_pointwise.ml. Printing the graph confirms no
   new node is built. *)
let%expect_test "dispatch: rsqrt.default legalizes to Pow exponent=-0.5" =
  let a = float_tensor [ 2; 2 ] [ 1.; 4.; 0.25; 16. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.rsqrt.default"
    ~bindings:[ ("self", a) ]
    ~inputs:[ in_tensor "self" ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=2] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=2]] = pow x=t0 scalar=-0.5
    outputs: [t1 f32 [W=2 C=2] <-n0]
    tensor f32 [W=2 C=2] {1, 0.5, 2, 0.25} |}]

let%expect_test "dispatch: sub.Tensor elementwise" =
  let a = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let b = float_tensor [ 2; 3 ] [ 0.; 10.; 100.; 1.; 2.; 3. ] in
  dispatch_print ~target:"torch.ops.aten.sub.Tensor"
    ~bindings:[ ("self", a); ("other", b) ]
    ~inputs:[ in_tensor "self"; in_tensor "other" ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=3] {1, -8, -97, 3, 3, 3} |}]

(* Default value=1: decomposes to [Mul]+[Add] only -- no [Mul_scalar] node,
   the corpus's only observed [value]. *)
let%expect_test
    "dispatch: addcmul.default value=1 (default) decomposes to Mul+Add" =
  let self = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let t1 = float_tensor [ 2; 3 ] [ 1.; 1.; 1.; 1.; 1.; 1. ] in
  let t2 = float_tensor [ 2; 3 ] [ 2.; 3.; 4.; 5.; 6.; 7. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.addcmul.default"
    ~bindings:[ ("self", self); ("tensor1", t1); ("tensor2", t2) ]
    ~inputs:[ in_tensor "self"; in_tensor "tensor1"; in_tensor "tensor2" ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs:
      [t0 f32 [W=2 C=3] ->[n1], t1 f32 [W=2 C=3] ->[n0], t2 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t3 f32 [W=2 C=3] ->[n1]] = mul a=t1 b=t2
      n1: [t4 f32 [W=2 C=3]] = add a=t0 b=t3 <-n0
    outputs: [t4 f32 [W=2 C=3] <-n1]
    tensor f32 [W=2 C=3] {3, 5, 7, 9, 11, 13} |}]

(* Non-unit value: an extra [Mul_scalar] node scales the product. *)
let%expect_test "dispatch: addcmul.default value=2 scales via Mul_scalar" =
  let self = float_tensor [ 2; 3 ] [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let t1 = float_tensor [ 2; 3 ] [ 1.; 1.; 1.; 1.; 1.; 1. ] in
  let t2 = float_tensor [ 2; 3 ] [ 2.; 3.; 4.; 5.; 6.; 7. ] in
  dispatch_print ~target:"torch.ops.aten.addcmul.default"
    ~bindings:[ ("self", self); ("tensor1", t1); ("tensor2", t2) ]
    ~inputs:
      [
        in_tensor "self";
        in_tensor "tensor1";
        in_tensor "tensor2";
        in_float "value" 2.0;
      ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=2 C=3] {5, 8, 11, 14, 17, 20} |}]

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

let%expect_test "dispatch: softmax.int dim=1" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.softmax.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim" 1 ]
    ~noutputs:1;
  [%expect
    {| tensor f32 [W=2 C=3] {0.0900306, 0.244728, 0.665241, 0.0900306, 0.244728, 0.665241} |}]

(* Negative [dim] normalizes the same as every other single-[dim] arm
   ([slice.Tensor], [unbind.int]): -1 is the last axis, here the same one
   [dim=1] names above -- same values, proving the normalization rather than
   just that some diagnostic fires. *)
let%expect_test "dispatch: softmax.int dim=-1 normalizes to the last axis" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.softmax.int"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_int "dim" (-1) ]
    ~noutputs:1;
  [%expect
    {| tensor f32 [W=2 C=3] {0.0900306, 0.244728, 0.665241, 0.0900306, 0.244728, 0.665241} |}]

let%expect_test "dispatch: softmax.int rejects an out-of-range dim" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  List.iter
    (fun d ->
      dispatch_print ~target:"torch.ops.aten.softmax.int"
        ~bindings:[ ("self", x) ]
        ~inputs:[ in_tensor "self"; in_int "dim" d ]
        ~noutputs:1)
    [ 7; -3 ];
  [%expect
    {|
    error: softmax.int: invalid dimension 7 for rank 2
    error: softmax.int: invalid dimension -3 for rank 2 |}]

let%expect_test "dispatch: softmax.int rejects a supplied dtype" =
  let x = float_tensor [ 2; 3 ] [ 0.; 1.; 2.; 3.; 4.; 5. ] in
  dispatch_print ~target:"torch.ops.aten.softmax.int"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [
        in_tensor "self";
        in_int "dim" 1;
        PT.NamedArgument.make "dtype"
          (PT.Argument.Scalar_type PT.ScalarType.DOUBLE) None;
      ]
    ~noutputs:1;
  [%expect {| error: unsupported scalar_type argument "dtype" |}]
