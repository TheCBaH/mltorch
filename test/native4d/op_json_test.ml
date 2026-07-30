(* One value per Native4D constructor, printed and round-tripped through JSON.

   Two properties, and the second is the one worth having. The round trip checks
   each codec. The COUNT checks that this file exercises every constructor: the
   op registry has one entry per constructor by construction, so comparing the
   sample count against the registry length catches an op added to the variant
   and forgotten here — which no round-trip test can catch on its own, because
   the case it would fail on is the case nobody wrote. *)

open Native4d

let t_ n = Tensor_id.of_int n
let x = t_ 0
let y = t_ 1
let w = t_ 2

let axis_window ~kernel : Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent kernel;
    stride = Op_config.Pos.of_int 1;
    pad_before = Op_config.Nonneg.of_int 0;
    pad_after = Op_config.Nonneg.of_int 1;
    dilation = Op_config.Pos.of_int 1;
  }

let conv_params : Ops4.Conv_params.t =
  {
    h = axis_window ~kernel:3;
    w = axis_window ~kernel:3;
    in_channels = Dim.extent 4;
  }

let hw n : 'a Op_config.Hw.t = { h = n; w = n }

(* Structurally identical records in two modules, so each needs its own
   annotation. *)
let max_params : Pool.MaxPool2d.params =
  {
    kernel = hw (Dim.extent 2);
    stride = hw (Op_config.Pos.of_int 2);
    pad = hw (Op_config.Nonneg.of_int 0);
  }

let avg_params : Pool.AvgPool2d.params =
  {
    kernel = hw (Dim.extent 2);
    stride = hw (Op_config.Pos.of_int 2);
    pad = hw (Op_config.Nonneg.of_int 0);
  }

(* Deliberately in the variant's own alphabetical order, so a reader can check
   the list against [Op.op] by eye. *)
let samples : Op.t list =
  [
    Add { Pointwise.Bin.a = x; b = y };
    Add_scalar { Pointwise.Scalar_bin.x; scalar = 0.1 };
    Avg_pool2d { Pool.AvgPool2d.params = avg_params; x };
    Clamp { Pointwise.Clamp.params = { min = Some 0.; max = Some 6. }; x };
    Conv2d
      { Ops4.Conv_payload.params = conv_params; x; weight = w; bias = None };
    Depthwise_conv2d
      {
        Ops4.Conv_payload.params = conv_params;
        x;
        weight = w;
        bias = Some (t_ 3);
      };
    Div { Pointwise.Bin.a = x; b = y };
    Div_scalar { Pointwise.Scalar_bin.x; scalar = 2. };
    Hardtanh { Pointwise.Hardtanh.params = { min_val = 0.; max_val = 6. }; x };
    Max_pool2d { Pool.MaxPool2d.params = max_params; x };
    Mean_keepdims { Ops4.Mean_keepdims.params = { dims = [ H; W ] }; x };
    Mul { Pointwise.Bin.a = x; b = y };
    Permute4
      {
        Ops4.Permute4.perm =
          Ops4.Permute4.of_fn (function H -> W | W -> H | a -> a);
        x;
      };
    Relu { Pointwise.Relu.x };
    Reshape4
      {
        Ops4.Reshape4.params = { shape = Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:12 };
        x;
      };
    Rms_norm
      {
        Ops4.Rms_norm.params = { dims = [ C ]; eps = 1e-5 };
        x;
        weight = Some w;
      };
    Sqrt { Pointwise.Sqrt.x };
    Sub { Pointwise.Bin.a = x; b = y };
    Transposed_conv2d
      {
        Ops4.Transposed_conv2d.params =
          {
            stride = hw (Op_config.Pos.of_int 2);
            padding = hw (Op_config.Nonneg.of_int 1);
            dilation = hw (Op_config.Pos.of_int 1);
            output_padding = hw (Op_config.Nonneg.of_int 1);
          };
        x;
        weight = w;
        bias = None;
      };
  ]

(* THE COVERAGE CHECK. Every constructor has exactly one registry entry, so if
   these disagree an op was added to [Op.op] and not sampled here. *)
let%expect_test "op4: every constructor is sampled" =
  Format.printf "samples: %d, registry: %d@." (List.length samples)
    (List.length Op.op_registry);
  [%expect {| samples: 19, registry: 19 |}]

let%expect_test "op4: printed" =
  List.iter (fun op -> Format.printf "%a@." Op.pp op) samples;
  [%expect
    {|
    add a=t0 b=t1
    add_scalar x=t0 scalar=0.1
    avg_pool2d x=t0 params={kernel={h=2; w=2}; stride={h=2; w=2}; pad={h=0; w=0}}
    clamp x=t0 params={min=0; max=6}
    conv2d
      x=t0
      weight=t2
      params={h={kernel=3; stride=1; pad_before=0; pad_after=1; dilation=1};
             w={kernel=3; stride=1; pad_before=0; pad_after=1; dilation=1};
             in_channels=4}
    depthwise_conv2d
      x=t0
      weight=t2
      bias=t3
      params={h={kernel=3; stride=1; pad_before=0; pad_after=1; dilation=1};
             w={kernel=3; stride=1; pad_before=0; pad_after=1; dilation=1};
             in_channels=4}
    div a=t0 b=t1
    div_scalar x=t0 scalar=2
    hardtanh x=t0 params={min_val=0; max_val=6}
    max_pool2d x=t0 params={kernel={h=2; w=2}; stride={h=2; w=2}; pad={h=0; w=0}}
    mean_keepdims x=t0 params={dims=[H, W]}
    mul a=t0 b=t1
    permute4 x=t0 perm=[H<-W, W<-H]
    relu x=t0
    reshape4 x=t0 params={shape=[N=1 H=1 W=1 C=12]}
    rms_norm x=t0 weight=t2 params={dims=[C]; eps=1e-05}
    sqrt x=t0
    sub a=t0 b=t1
    transposed_conv2d
      x=t0
      weight=t2
      params={stride={h=2; w=2};
             padding={h=1; w=1};
             dilation={h=1; w=1};
             output_padding={h=1; w=1}} |}]

let%expect_test "op4: round-trips through JSON" =
  List.iter
    (fun op ->
      let back = Json_util.dec Op.jsont (Json_util.enc Op.jsont op) in
      let same =
        Format.asprintf "%a" Op.pp op = Format.asprintf "%a" Op.pp back
      in
      if not same then Format.printf "MISMATCH@ %a@ -> %a@." Op.pp op Op.pp back)
    samples;
  Format.printf "round-tripped %d ops@." (List.length samples);
  [%expect {| round-tripped 19 ops |}]
