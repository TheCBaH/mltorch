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

(* [in_channels=8], [groups=2]: neither 1 nor depthwise (groups=in_channels),
   so distinct from both [conv_params] uses above. *)
let grouped_conv_params : Ops4.Grouped_conv_params.t =
  {
    h = axis_window ~kernel:3;
    w = axis_window ~kernel:3;
    in_channels = Dim.extent 8;
    groups = Op_config.Pos.of_int 2;
  }

let hw n : 'a Op_config.Hw.t = { h = n; w = n }

(* Structurally identical records in two modules, so each needs its own
   annotation. *)
let max_params : Pool.MaxPool2d.params =
  {
    ceil_mode = false;
    kernel = hw (Dim.extent 2);
    stride = hw (Op_config.Pos.of_int 2);
    pad = hw (Op_config.Nonneg.of_int 0);
  }

let avg_params : Pool.AvgPool2d.params =
  {
    ceil_mode = false;
    count_include_pad = true;
    kernel = hw (Dim.extent 2);
    stride = hw (Op_config.Pos.of_int 2);
    pad = hw (Op_config.Nonneg.of_int 0);
  }

let adaptive_params : Pool.AdaptiveAvgPool2d.params =
  { output_size = hw (Op_config.Pos.of_int 3) }

let adaptive_max_params : Pool.AdaptiveMaxPool2d.params =
  { output_size = hw (Op_config.Pos.of_int 4) }

let upsample_params : Resize.Bilinear2d.params =
  { output_size = hw (Op_config.Pos.of_int 5); align_corners = false }

let nearest_params : Resize.Nearest2d.params =
  { output_size = hw (Op_config.Pos.of_int 5) }

(* Deliberately in the variant's own alphabetical order, so a reader can check
   the list against [Op.op] by eye. *)
let samples : Op.t list =
  [
    Add { Pointwise.Bin.a = x; b = y };
    Add_scalar { Pointwise.Scalar_bin.x; scalar = 0.1 };
    Adaptive_avg_pool2d { Pool.AdaptiveAvgPool2d.params = adaptive_params; x };
    Adaptive_max_pool2d
      { Pool.AdaptiveMaxPool2d.params = adaptive_max_params; x };
    Avg_pool2d { Pool.AvgPool2d.params = avg_params; x };
    Batch_norm_no_stats
      {
        Ops4.Batch_norm_no_stats.params = { channel = C; eps = 1e-5 };
        x;
        weight = Some w;
        bias = Some b;
      };
    Batched_matmul { Matmul.Batched_matmul.input = x; mat2 = y };
    Clamp { Pointwise.Clamp.params = { min = Some 0.; max = Some 6. }; x };
    (* Three operands, so a codec that dropped or reordered one is visible. *)
    Concat4 { Ops4.Concat4.params = { axis = W }; xs = [ x; y; w ] };
    Conv2d
      { Ops4.Conv_payload.params = conv_params; x; weight = w; bias = None };
    Cumsum4 { Ops4_cumsum.Cumsum4.params = { axis = C }; x };
    Depthwise_conv2d
      {
        Ops4.Conv_payload.params = conv_params;
        x;
        weight = w;
        bias = Some (t_ 3);
      };
    Div { Pointwise.Bin.a = x; b = y };
    Div_scalar { Pointwise.Scalar_bin.x; scalar = 2. };
    Expand4
      { Ops4.Expand4.params = { size = Shape4.of_ints ~n:1 ~h:3 ~w:4 ~c:5 }; x };
    Gelu { Pointwise.Gelu.x; approximate = Exact };
    Group_norm4
      {
        Ops4.Group_norm4.params =
          { channel = C; groups = Op_config.Pos.of_int 2; eps = 1e-5 };
        x;
        weight = Some w;
        bias = Some b;
      };
    Grouped_conv2d
      {
        Ops4.Grouped_conv_payload.params = grouped_conv_params;
        x;
        weight = w;
        bias = None;
      };
    Hardsigmoid { Pointwise.Hardsigmoid.x };
    Hardswish { Pointwise.Hardswish.x };
    Hardtanh { Pointwise.Hardtanh.params = { min_val = 0.; max_val = 6. }; x };
    (* Axis distinct from [Select4]'s (H) and [Select_scatter4]'s (W), and
       two distinct operands, so an encoder that dropped the axis or
       confused [self]/[index] still prints differently. *)
    IndexTensor4
      { Ops4.IndexTensor4.params = { axis = N }; self = x; index = y };
    Layer_norm
      {
        Ops4.Layer_norm.params = { dims = [ C ]; eps = 1e-5 };
        x;
        weight = Some w;
        bias = Some b;
      };
    Leaky_relu { Pointwise.Leaky_relu.params = { negative_slope = 0.2 }; x };
    (* One layer, one direction, WITH bias: distinct ids for every one of the
       seven operand roles (input, weight_ih, weight_hh, bias_ih, bias_hh, h0,
       c0), so a codec confusing any two still prints and round-trips
       differently. [reverse = None] is exercised here; [Sdpa]'s own sample
       just above already covers this file's "optional operand present"
       case, so this one leaves it absent instead of restating that point. *)
    Lstm
      {
        Lstm.Lstm.params =
          { hidden_size = 2; input_size = 3; batch_first = false };
        layers =
          [
            {
              Lstm.Lstm.Layer.forward =
                {
                  Lstm.Lstm.Direction.weight_ih = x;
                  weight_hh = y;
                  bias = Some (w, b);
                };
              reverse = None;
            };
          ];
        input = t_ 4;
        h0 = t_ 5;
        c0 = t_ 6;
      };
    Max_keepdims { Ops4.Max_keepdims.params = { dims = [ H; W ] }; x };
    Max_pool2d { Pool.MaxPool2d.params = max_params; x };
    Mean_keepdims { Ops4.Mean_keepdims.params = { dims = [ H; W ] }; x };
    Mul { Pointwise.Bin.a = x; b = y };
    Mul_scalar { Pointwise.Scalar_bin.x; scalar = 2. };
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
    Pow { Pointwise.Scalar_bin.x; scalar = 2. };
    Relu { Pointwise.Relu.x };
    (* Two nontrivial multipliers (H, C) and two unit ones (N, W), so a codec
       that dropped or reordered an axis is visible. *)
    Repeat4
      {
        Ops4.Repeat4.params = { repeats = Shape4.of_ints ~n:1 ~h:2 ~w:1 ~c:3 };
        x;
      };
    (* Axis distinct from [Repeat4]'s own sample and a multiplier that is
       neither 1 nor either of [Repeat4]'s, so an encoder that confused the
       two still prints differently. *)
    RepeatInterleave4
      {
        Ops4.RepeatInterleave4.params =
          { axis = W; repeats = Op_config.Pos.of_int 4 };
        x;
      };
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
    (* Two distinct scalars, neither the schema default (alpha=1), so an
       encoder that dropped or swapped either field would still print
       differently. *)
    Rsub_scalar { Pointwise.Rsub_scalar.params = { other = 1.; alpha = 2. }; x };
    (* Mask PRESENT, so a codec that dropped the optional fourth operand still
       prints and round-trips differently. *)
    Sdpa
      {
        Attention.Sdpa.params = { scale = Attention.Sdpa.Scale.Default };
        query = x;
        key = y;
        value = w;
        mask = Some b;
      };
    (* Axis distinct from every other sample's and an index that is neither
       0 nor the axis's last valid one, so an encoder that dropped or
       defaulted either field would still print differently. *)
    Select4 { Ops4.Select4.params = { axis = H; index = 2 }; x };
    (* Axis and index distinct from [Select4]'s own sample, and two distinct
       operands, so an encoder that dropped or swapped [self]/[src] or
       confused the two ops still prints differently. *)
    Select_scatter4
      {
        Ops4.Select_scatter4.params = { axis = W; index = 1 };
        self = x;
        src = y;
      };
    Sigmoid { Pointwise.Sigmoid.x };
    Silu { Pointwise.Silu.x };
    (* Three distinct bounds and a step that is not 1, so an encoder that
       permuted the fields still decodes and prints differently. *)
    Slice4
      {
        Ops4.Slice4.params =
          { axis = W; start = 1; stop = 8; step = Op_config.Pos.of_int 3 };
        x;
      };
    (* Axis distinct from [Slice4]'s own sample, so an encoder that confused
       the two still prints differently. *)
    Softmax4 { Ops4.Softmax4.params = { axis = C }; x };
    (* Three unequal sizes, so an encoder that dropped or reordered one
       differs in printed output as well as in count. *)
    Split_with_sizes4
      { Ops4.Split_with_sizes4.params = { axis = W; sizes = [ 2; 3; 1 ] }; x };
    Sqrt { Pointwise.Sqrt.x };
    (* Two operands and an axis distinct from [Concat4]'s own sample above, so
       an encoder that confused the two variadic-operand ops still prints
       differently. *)
    Stack4 { Ops4.Stack4.params = { axis = N }; xs = [ y; b ] };
    Sub { Pointwise.Bin.a = x; b = y };
    Sum_keepdims { Ops4.Sum_keepdims.params = { dims = [ H; W ] }; x };
    To_copy { Pointwise.To_copy.target = Long; x };
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
    Upsample_bilinear2d { Resize.Bilinear2d.params = upsample_params; x };
    Upsample_nearest2d { Resize.Nearest2d.params = nearest_params; x };
    Vector_norm_keepdims
      { Ops4.Vector_norm_keepdims.params = { dims = [ H; W ] }; x };
    Arange4
      {
        Ops4.Arange4.params =
          { start = 0.5; stop = 4.; step = 1.; fmt = Payload.Fmt Payload.F32 };
      };
    Zeros4
      {
        Ops4.Zeros4.params =
          {
            shape = Shape4.of_ints ~n:1 ~h:2 ~w:3 ~c:4;
            fmt = Payload.Fmt Payload.F64;
          };
      };
    Eye4
      {
        Ops4.Eye4.params =
          {
            shape = Shape4.of_ints ~n:1 ~h:1 ~w:2 ~c:3;
            fmt = Payload.Fmt Payload.F32;
          };
      };
  ]

(* THE COVERAGE CHECK. Every constructor has exactly one registry entry, so if
   these disagree an op was added to [Op.op] and not sampled here. *)
let%expect_test "op4: every constructor is sampled" =
  Format.printf "samples: %d, registry: %d@." (List.length samples)
    (List.length Op.op_registry);
  [%expect {| samples: 60, registry: 60 |}]

let%expect_test "op4: printed" =
  List.iter (fun op -> Format.printf "%a@." Op.pp op) samples;
  [%expect
    {|
    add a=t0 b=t1
    add_scalar x=t0 scalar=0.1
    adaptive_avg_pool2d x=t0 params={output_size={h=3; w=3}}
    adaptive_max_pool2d x=t0 params={output_size={h=4; w=4}}
    avg_pool2d
      x=t0
      params={kernel={h=2; w=2};
             stride={h=2; w=2};
             pad={h=0; w=0};
             ceil_mode=false;
             count_include_pad=true}
    batch_norm_no_stats x=t0 weight=t2 bias=t3 params={channel=C; eps=1e-05}
    batched_matmul input=t0 mat2=t1
    clamp x=t0 params={min=0; max=6}
    concat4 xs=[t0, t1, t2] params={axis=W}
    conv2d
      x=t0
      weight=t2
      params={h={kernel=3; stride=1; pad_before=0; pad_after=1; dilation=1};
             w={kernel=3; stride=1; pad_before=0; pad_after=1; dilation=1};
             in_channels=4}
    cumsum4 x=t0 params={axis=C}
    depthwise_conv2d
      x=t0
      weight=t2
      bias=t3
      params={h={kernel=3; stride=1; pad_before=0; pad_after=1; dilation=1};
             w={kernel=3; stride=1; pad_before=0; pad_after=1; dilation=1};
             in_channels=4}
    div a=t0 b=t1
    div_scalar x=t0 scalar=2
    expand4 x=t0 params={size=[N=1 H=3 W=4 C=5]}
    gelu x=t0 approximate=none
    group_norm4 x=t0 weight=t2 bias=t3 params={channel=C; groups=2; eps=1e-05}
    grouped_conv2d
      x=t0
      weight=t2
      params={h={kernel=3; stride=1; pad_before=0; pad_after=1; dilation=1};
             w={kernel=3; stride=1; pad_before=0; pad_after=1; dilation=1};
             in_channels=8;
             groups=2}
    hardsigmoid x=t0
    hardswish x=t0
    hardtanh x=t0 params={min_val=0; max_val=6}
    index_tensor4 self=t0 index=t1 params={axis=N}
    layer_norm x=t0 weight=t2 bias=t3 params={dims=[C]; eps=1e-05}
    leaky_relu x=t0 params={negative_slope=0.2}
    lstm
      input=t4
      layers=[{forward={weight_ih=t0; weight_hh=t1; bias=(t2,t3)}; reverse=none}]
      h0=t5
      c0=t6
      params={hidden_size=2; input_size=3; batch_first=false}
    max_keepdims x=t0 params={dims=[H, W]}
    max_pool2d
      x=t0
      params={kernel={h=2; w=2};
             stride={h=2; w=2};
             pad={h=0; w=0};
             ceil_mode=false}
    mean_keepdims x=t0 params={dims=[H, W]}
    mul a=t0 b=t1
    mul_scalar x=t0 scalar=2
    pad4 x=t0 params={pads=[H:1,2, W:-1,3] mode=constant(0.1)}
    permute4 x=t0 perm=[H<-W, W<-H]
    pow x=t0 scalar=2
    relu x=t0
    repeat4 x=t0 params={repeats=[N=1 H=2 W=1 C=3]}
    repeat_interleave4 x=t0 params={axis=W repeats=4}
    reshape4 x=t0 params={shape=[N=1 H=1 W=1 C=12]}
    rms_norm x=t0 weight=t2 params={dims=[C]; eps=1e-05}
    rsub_scalar x=t0 params={other=1; alpha=2}
    sdpa query=t0 key=t1 value=t2 mask=t3 params={scale=default}
    select4 x=t0 params={axis=H index=2}
    select_scatter4 self=t0 src=t1 params={axis=W index=1}
    sigmoid x=t0
    silu x=t0
    slice4 x=t0 params={axis=W start=1 stop=8 step=3}
    softmax4 x=t0 params={axis=C}
    split_with_sizes4 x=t0 params={axis=W sizes=[2, 3, 1]}
    sqrt x=t0
    stack4 xs=[t1, t3] params={axis=N}
    sub a=t0 b=t1
    sum_keepdims x=t0 params={dims=[H, W]}
    to_copy x=t0 target=long
    transposed_conv2d
      x=t0
      weight=t2
      params={stride={h=2; w=2};
             padding={h=1; w=1};
             dilation={h=1; w=1};
             output_padding={h=1; w=1}}
    unbind x=t0 params={axis=C}
    upsample_bilinear2d x=t0 params={output_size={h=5; w=5}; align_corners=false}
    upsample_nearest2d x=t0 params={output_size={h=5; w=5}}
    vector_norm_keepdims x=t0 params={dims=[H, W]}
    arange4 start=0.5 stop=4 step=1 fmt=f32
    zeros4 shape=[N=1 H=2 W=3 C=4] fmt=f64
    eye4 shape=[N=1 H=1 W=2 C=3] fmt=f32 |}]

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
  [%expect {| round-tripped 60 ops |}]

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
            ceil_mode = true;
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
    max_pool2d
      x=t0
      params={kernel={h=3; w=2};
             stride={h=2; w=1};
             pad={h=1; w=0};
             ceil_mode=true}
    round-tripped 7 ops |}]
