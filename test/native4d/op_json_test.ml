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

(* A fourth distinct id, so a codec that confused [Layer_norm]'s two affine
   operands prints a different reference rather than the same one twice. *)
let b = t_ 3

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
    Hardsigmoid { Pointwise.Hardsigmoid.x };
    Hardswish { Pointwise.Hardswish.x };
    Hardtanh { Pointwise.Hardtanh.params = { min_val = 0.; max_val = 6. }; x };
    Layer_norm
      {
        Ops4.Layer_norm.params = { dims = [ C ]; eps = 1e-5 };
        x;
        weight = Some w;
        bias = Some b;
      };
    Max_pool2d { Pool.MaxPool2d.params = max_params; x };
    Mean_keepdims { Ops4.Mean_keepdims.params = { dims = [ H; W ] }; x };
    Mul { Pointwise.Bin.a = x; b = y };
    (* Two axes, an asymmetric pad and a mixed pad/crop, so the codec is proved
       on a SIGNED amount rather than only on the padding half of the range. The
       fill is not f32-exact, so a narrowing round trip would be visible. *)
    Pad4
      {
        Ops4.Pad4.params =
          {
            pads =
              [
                (H, { Pad.Pad.before = 1; after = 2 });
                (W, { before = -1; after = 3 });
              ];
            mode = Pad.Pad.Constant 0.1;
          };
        x;
      };
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
    Silu { Pointwise.Silu.x };
    (* Three distinct bounds and a step that is not 1, so an encoder that
       permuted the fields still decodes and prints differently. *)
    Slice4
      {
        Ops4.Slice4.params =
          { axis = W; start = 1; stop = 8; step = Op_config.Pos.of_int 3 };
        x;
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
    Unbind { Ops4.Unbind.params = { axis = C }; x };
  ]

(* THE COVERAGE CHECK. Every constructor has exactly one registry entry, so if
   these disagree an op was added to [Op.op] and not sampled here. *)
let%expect_test "op4: every constructor is sampled" =
  Format.printf "samples: %d, registry: %d@." (List.length samples)
    (List.length Op.op_registry);
  [%expect {| samples: 26, registry: 26 |}]

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
    hardsigmoid x=t0
    hardswish x=t0
    hardtanh x=t0 params={min_val=0; max_val=6}
    layer_norm x=t0 weight=t2 bias=t3 params={dims=[C]; eps=1e-05}
    max_pool2d x=t0 params={kernel={h=2; w=2}; stride={h=2; w=2}; pad={h=0; w=0}}
    mean_keepdims x=t0 params={dims=[H, W]}
    mul a=t0 b=t1
    pad4 x=t0 params={pads=[H:1,2, W:-1,3] mode=constant(0.1)}
    permute4 x=t0 perm=[H<-W, W<-H]
    relu x=t0
    reshape4 x=t0 params={shape=[N=1 H=1 W=1 C=12]}
    rms_norm x=t0 weight=t2 params={dims=[C]; eps=1e-05}
    silu x=t0
    slice4 x=t0 params={axis=W start=1 stop=8 step=3}
    sqrt x=t0
    sub a=t0 b=t1
    transposed_conv2d
      x=t0
      weight=t2
      params={stride={h=2; w=2};
             padding={h=1; w=1};
             dilation={h=1; w=1};
             output_padding={h=1; w=1}}
    unbind x=t0 params={axis=C} |}]

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
  [%expect {| round-tripped 26 ops |}]

(* ---- Group-2 payloads the constructor sweep above does not reach --------- *)

(* [samples] is a coverage check over CONSTRUCTORS: exactly one entry per op, so
   it cannot also carry a second configuration of one. These are the Group-2
   shapes that differ from the sampled ones in a way a codec can get wrong --
   a multi-axis [dims] and an ABSENT weight, the case the ATen bridge used to
   make unreachable by materializing a ones tensor. *)
let group2_samples =
  let open Op in
  [
    Rms_norm
      {
        Ops4.Rms_norm.params = { dims = [ H; W; C ]; eps = 1e-5 };
        x;
        weight = Some w;
      };
    Rms_norm
      { Ops4.Rms_norm.params = { dims = [ W; C ]; eps = 0. }; x; weight = None };
    (* The four affine states are four distinct encodings, and the two
       one-sided ones are what a codec pairing the operands gets wrong. [eps] is
       0.1 rather than a power of ten: it is not exact in f32, so a scalar
       narrowed on the way through is visible in the printed value. *)
    Layer_norm
      {
        Ops4.Layer_norm.params = { dims = [ H; W; C ]; eps = 0.1 };
        x;
        weight = Some w;
        bias = None;
      };
    Layer_norm
      {
        Ops4.Layer_norm.params = { dims = [ W; C ]; eps = 0. };
        x;
        weight = None;
        bias = Some b;
      };
    Layer_norm
      {
        Ops4.Layer_norm.params = { dims = [ C ]; eps = 1e-6 };
        x;
        weight = None;
        bias = None;
      };
    Depthwise_conv2d
      {
        Ops4.Conv_payload.params =
          {
            h =
              {
                kernel = Dim.extent 3;
                stride = Op_config.Pos.of_int 2;
                pad_before = Op_config.Nonneg.of_int 1;
                pad_after = Op_config.Nonneg.of_int 1;
                dilation = Op_config.Pos.of_int 1;
              };
            w =
              {
                kernel = Dim.extent 2;
                stride = Op_config.Pos.of_int 1;
                pad_before = Op_config.Nonneg.of_int 0;
                pad_after = Op_config.Nonneg.of_int 0;
                dilation = Op_config.Pos.of_int 2;
              };
            in_channels = Dim.extent 4;
          };
        x;
        weight = w;
        bias = None;
      };
    Max_pool2d
      {
        Pool.MaxPool2d.params =
          {
            kernel = Op_config.Hw.{ h = Dim.extent 3; w = Dim.extent 2 };
            stride =
              Op_config.Hw.
                { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 1 };
            pad =
              Op_config.Hw.
                { h = Op_config.Nonneg.of_int 1; w = Op_config.Nonneg.of_int 0 };
          };
        x;
      };
  ]

let%expect_test "op4: Group-2 non-default payloads round-trip" =
  List.iter (fun op -> Format.printf "%a@." Op.pp op) group2_samples;
  List.iter
    (fun op ->
      let back = Json_util.dec Op.jsont (Json_util.enc Op.jsont op) in
      if Format.asprintf "%a" Op.pp op <> Format.asprintf "%a" Op.pp back then
        Format.printf "MISMATCH@ %a@ -> %a@." Op.pp op Op.pp back)
    group2_samples;
  Format.printf "round-tripped %d ops@." (List.length group2_samples);
  [%expect
    {|
    rms_norm x=t0 weight=t2 params={dims=[H, W, C]; eps=1e-05}
    rms_norm x=t0 params={dims=[W, C]; eps=0}
    layer_norm x=t0 weight=t2 params={dims=[H, W, C]; eps=0.1}
    layer_norm x=t0 bias=t3 params={dims=[W, C]; eps=0}
    layer_norm x=t0 params={dims=[C]; eps=1e-06}
    depthwise_conv2d
      x=t0
      weight=t2
      params={h={kernel=3; stride=2; pad_before=1; pad_after=1; dilation=1};
             w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=2};
             in_channels=4}
    max_pool2d x=t0 params={kernel={h=3; w=2}; stride={h=2; w=1}; pad={h=1; w=0}}
    round-tripped 7 ops |}]
