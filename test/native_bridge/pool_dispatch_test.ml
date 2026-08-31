(* Op_bridge.dispatch for the pooling/resize family (max_pool2d, avg_pool2d,
   adaptive_avg_pool2d, upsample_bilinear2d, max_pool2d_with_indices) --
   split out of dispatch_test.ml to stay under the tracked file-size ceiling.
   Promote with [dune promote test/native_bridge/pool_dispatch_test.ml]. *)

open Helpers

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
    "dispatch: avg_pool2d.default relayouts, count_include_pad=true divides by \
     the full kernel area" =
  let x =
    float_tensor [ 1; 1; 3; 3 ] (List.init 9 (fun i -> float_of_int (i + 1)))
  in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.avg_pool2d.default"
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
        avg_pool2d
          x=t1 <-n0
          params={kernel={h=2; w=2};
                 stride={h=1; w=1};
                 pad={h=1; w=1};
                 ceil_mode=false;
                 count_include_pad=true}
      n2: [t3 f32 [W=4 C=4]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [W=4 C=4] <-n2]
    tensor f32 [W=4 C=4] {0.25, 0.75, 1.25, 0.75, 1.25, 3, 4, 2.25, ...} |}]

(* Same window as above but [count_include_pad=false]: every corner/edge
   divisor shrinks to the real (non-padding) tap count, so the corners equal
   their single real input value instead of a quarter of it. *)
let%expect_test
    "dispatch: avg_pool2d.default count_include_pad=false divides by the real \
     window area" =
  let x =
    float_tensor [ 1; 1; 3; 3 ] (List.init 9 (fun i -> float_of_int (i + 1)))
  in
  dispatch_print ~target:"torch.ops.aten.avg_pool2d.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [
        in_tensor "self";
        in_ints "kernel_size" [ 2; 2 ];
        in_ints "stride" [ 1; 1 ];
        in_ints "padding" [ 1; 1 ];
        in_bool "count_include_pad" false;
      ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=4 C=4] {1, 1.5, 2.5, 3, 2.5, 3, 4, 4.5, ...} |}]

let%expect_test
    "dispatch: avg_pool2d.default divisor_override is refused, not dropped" =
  let x = float_tensor [ 1; 1; 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  dispatch_print ~target:"torch.ops.aten.avg_pool2d.default"
    ~bindings:[ ("self", x) ]
    ~inputs:
      [
        in_tensor "self";
        in_ints "kernel_size" [ 2; 2 ];
        in_int "divisor_override" 5;
      ]
    ~noutputs:1;
  [%expect
    {| error: torch.ops.aten.avg_pool2d.default: divisor_override=5 is not supported (only none) |}]

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

(* No [align_corners] argument at all, unlike [upsample_bilinear2d.vec]
   above -- see [Resize.Nearest_axis]'s module doc. Otherwise the identical
   output_size/scale_factors contract, sharing
   [Op_bridge_decode.resolve_upsample_size] with the bilinear arm. *)
let%expect_test "dispatch: upsample_nearest2d.vec explicit output_size" =
  let x = float_tensor [ 1; 1; 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.upsample_nearest2d.vec"
    ~bindings:[ ("input", x) ]
    ~inputs:
      [
        in_tensor "input";
        in_ints "output_size" [ 4; 4 ];
        in_none "scale_factors";
      ]
    ~noutputs:1;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=2 C=2] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=2 C=1] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t2 f32 [H=4 W=4 C=1] ->[n2]] =
        upsample_nearest2d x=t1 <-n0 params={output_size={h=4; w=4}}
      n2: [t3 f32 [W=4 C=4]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t3 f32 [W=4 C=4] <-n2]
    tensor f32 [W=4 C=4] {1, 1, 2, 2, 1, 1, 2, 2, ...} |}]

(* Same [2x2 -> 3x3] geometry as [upsample_bilinear2d.vec]'s own
   scale_factors fixture above, but every output is an exact copy of its
   nearest source pixel rather than a blend -- no value here is a fraction
   the source values cannot produce. *)
let%expect_test "dispatch: upsample_nearest2d.vec via scale_factors" =
  let x = float_tensor [ 1; 1; 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  dispatch_print ~target:"torch.ops.aten.upsample_nearest2d.vec"
    ~bindings:[ ("input", x) ]
    ~inputs:
      [
        in_tensor "input";
        in_none "output_size";
        in_floats "scale_factors" [ 1.5; 1.5 ];
      ]
    ~noutputs:1;
  [%expect {| tensor f32 [W=3 C=3] {1, 1, 2, 1, 1, 2, 3, 3, ...} |}]

let%expect_test "dispatch: upsample_nearest2d.vec rejects neither size arg" =
  let x = float_tensor [ 1; 1; 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  dispatch_print ~target:"torch.ops.aten.upsample_nearest2d.vec"
    ~bindings:[ ("input", x) ]
    ~inputs:
      [ in_tensor "input"; in_none "output_size"; in_none "scale_factors" ]
    ~noutputs:1;
  [%expect
    {|
    error: upsample_nearest2d.vec: exactly one of output_size or scale_factors must be given |}]

let%expect_test "dispatch: upsample_nearest2d.vec rejects both size args" =
  let x = float_tensor [ 1; 1; 2; 2 ] [ 1.; 2.; 3.; 4. ] in
  dispatch_print ~target:"torch.ops.aten.upsample_nearest2d.vec"
    ~bindings:[ ("input", x) ]
    ~inputs:
      [
        in_tensor "input";
        in_ints "output_size" [ 3; 3 ];
        in_floats "scale_factors" [ 1.5; 1.5 ];
      ]
    ~noutputs:1;
  [%expect
    {|
    error: upsample_nearest2d.vec: output_size and scale_factors are mutually exclusive |}]

let%expect_test
    "dispatch: adaptive_max_pool2d.default relayouts and discards indices" =
  (* Same 5x5 -> 3x3 non-divisible bins as the adaptive_avg_pool2d fixture
     above, values increasing row-major so each bin's max is its
     bottom-right corner (see pool_test.ml's own hand-computed version of
     this same 5x5/3x3 case for the bin ranges). *)
  let x =
    float_tensor [ 1; 1; 5; 5 ] (List.init 25 (fun i -> float_of_int (i + 1)))
  in
  dispatch_print_with_graph ~print_graph:true
    ~target:"torch.ops.aten.adaptive_max_pool2d.default"
    ~bindings:[ ("self", x) ]
    ~inputs:[ in_tensor "self"; in_ints "output_size" [ 3; 3 ] ]
    ~noutputs:2;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=5 C=5] ->[n0]]
    nodes:
      n0: [t1 f32 [H=5 W=5 C=1] ->[n1]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
      n1: [t2 f32 [H=3 W=3 C=1] ->[n3], t3 f32 [H=3 W=3 C=1] ->[n2]] =
        adaptive_max_pool2d_with_indices
          x=t1 <-n0
          params={output_size={h=3; w=3}}
      n2: [] = discard x=t3 <-n1
      n3: [t4 f32 [W=3 C=3]] = permute x=t2 <-n1 perm=[H<-C, W<-H, C<-W]
    outputs: [t4 f32 [W=3 C=3] <-n3]
    tensor f32 [W=3 C=3] {7, 9, 10, 17, 19, 20, 22, 24, ...} |}]

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
