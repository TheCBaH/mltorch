(* In-process tests for the spec runner: decode a spec, then verify it (ATen vs
   native) or evaluate and print both outputs. These exercise the same path as
   bin/aten_spec_verify but without spawning the CLI per case, so the libtorch
   build is loaded once and the add-generator coverage runs fast. The cram test
   keeps a couple of end-to-end cases through the actual binary + stdin. *)

let decode s =
  match Jsont_bytesrw.decode_string Aten_op_spec.jsont s with
  | Ok spec -> spec
  | Error e -> failwith ("decode: " ^ e)

(* Run an action that prints through a Format formatter, capturing to stdout so
   ppx_expect sees it. *)
let capture f = print_string (Core.Pretty.capture_to_string f)
let eval s = capture (fun ppf -> Aten_spec_run.eval_print ~ppf (decode s))

(* --- add.Tensor, one generator per input (same generator on both) --- *)

let%expect_test "add: uniform + uniform" =
  eval
    {|{ "target": "torch.ops.aten.add.Tensor",
        "args": {
          "self":  { "dtype": "f32", "shape": [2, 2], "random": { "uniform": { "low": -5.0, "high": 5.0 } } },
          "other": { "dtype": "f32", "shape": [2, 2], "random": { "uniform": { "low": 0.0, "high": 1.0 } } } } }|};
  [%expect
    {|
    [eval] torch.ops.aten.add.Tensor
      aten   out0 = [-3.22136; 4.8131; -0.765634; 4.3739]
      native out0 = [-3.22136; 4.8131; -0.765634; 4.3739] |}]

let%expect_test "add: normal + normal" =
  eval
    {|{ "target": "torch.ops.aten.add.Tensor",
        "args": {
          "self":  { "dtype": "f32", "shape": [3], "random": { "normal": { "mean": 2.0,  "variance": 0.5 } } },
          "other": { "dtype": "f32", "shape": [3], "random": { "normal": { "mean": -1.0, "variance": 4.0 } } } } }|};
  [%expect
    {|
    [eval] torch.ops.aten.add.Tensor
      aten   out0 = [-0.488476; 4.42813; -1.17741]
      native out0 = [-0.488476; 4.42813; -1.17741] |}]

(* --- add.Tensor, different generator for each of the two inputs --- *)

let%expect_test "add: uniform self + normal other" =
  eval
    {|{ "target": "torch.ops.aten.add.Tensor",
        "args": {
          "self":  { "dtype": "f32", "shape": [2, 3], "random": { "uniform": { "low": -1.0, "high": 1.0 } } },
          "other": { "dtype": "f32", "shape": [2, 3], "random": { "normal":  { "mean": 0.0, "variance": 1.0 } } } } }|};
  [%expect
    {|
    [eval] torch.ops.aten.add.Tensor
      aten   out0 = [-2.14905; 2.19213; -1.263; -0.075148; 0.903916; 1.46828]
      native out0 = [-2.14905; 2.19213; -1.263; -0.075148; 0.903916; 1.46828] |}]

let%expect_test "add: sequence self + values other (exact)" =
  eval
    {|{ "target": "torch.ops.aten.add.Tensor",
        "args": {
          "self":  { "dtype": "f32", "shape": [4], "sequence": { "start": 0.0, "step": 1.0 } },
          "other": { "dtype": "f32", "shape": [4], "values": [{"float":0.25}, {"float":0.25}, {"float":0.25}, {"float":0.25}] } } }|};
  [%expect
    {|
    [eval] torch.ops.aten.add.Tensor
      aten   out0 = [0.25; 1.25; 2.25; 3.25]
      native out0 = [0.25; 1.25; 2.25; 3.25] |}]

(* --- add.Tensor, the remaining single generators, with exact printed output --- *)

let%expect_test "add: values + values" =
  eval
    {|{ "target": "torch.ops.aten.add.Tensor",
        "args": {
          "self":  { "dtype": "f32", "shape": [2], "values": [{"float":1}, {"float":2}] },
          "other": { "dtype": "f32", "shape": [2], "values": [{"float":0.5}, {"float":0.5}] } } }|};
  [%expect
    {|
    [eval] torch.ops.aten.add.Tensor
      aten   out0 = [1.5; 2.5]
      native out0 = [1.5; 2.5] |}]

let%expect_test "add: sequence + sequence" =
  eval
    {|{ "target": "torch.ops.aten.add.Tensor",
        "args": {
          "self":  { "dtype": "f32", "shape": [3], "sequence": { "start": 0.0,  "step": 1.0 } },
          "other": { "dtype": "f32", "shape": [3], "sequence": { "start": 10.0, "step": 10.0 } } } }|};
  [%expect
    {|
    [eval] torch.ops.aten.add.Tensor
      aten   out0 = [10; 21; 32]
      native out0 = [10; 21; 32] |}]

(* --- relu, sequence crossing zero --- *)

let%expect_test "relu: sequence crossing zero" =
  eval
    {|{ "target": "torch.ops.aten.relu.default",
        "args": { "self": { "dtype": "f32", "shape": [5], "sequence": { "start": -2.0, "step": 1.0 } } } }|};
  [%expect
    {|
    [eval] torch.ops.aten.relu.default
      aten   out0 = [0; 0; 0; 1; 2]
      native out0 = [0; 0; 0; 1; 2] |}]

(* --- bmm: rank-3 batched matmul, no relayout (B/rows/cols land on H/W/C) --- *)

let%expect_test "bmm: 1x2x2 @ 1x2x2" =
  eval
    {|{ "target": "torch.ops.aten.bmm.default",
        "args": {
          "self": { "dtype": "f32", "shape": [1, 2, 2], "sequence": { "start": 1.0, "step": 1.0 } },
          "mat2": { "dtype": "f32", "shape": [1, 2, 2], "sequence": { "start": 1.0, "step": 1.0 } } } }|};
  [%expect
    {|
    [eval] torch.ops.aten.bmm.default
      aten   out0 = [7; 10; 15; 22]
      native out0 = [7; 10; 15; 22] |}]

(* --- conv2d.padding: string padding overload, distinct native op --- *)

let%expect_test "conv2d.default: dilated 2d matches native" =
  eval
    {|{ "target": "torch.ops.aten.conv2d.default",
        "args": {
          "input": { "dtype": "f32", "shape": [1, 1, 1, 5], "values": [{"float":0},{"float":1},{"float":2},{"float":3},{"float":4}] },
          "weight": { "dtype": "f32", "shape": [1, 1, 1, 3], "values": [{"float":1},{"float":1},{"float":1}] },
          "bias": { "dtype": "f32", "shape": [1], "values": [{"float":0}] },
          "stride": [1, 1],
          "padding": [0, 1],
          "dilation": [1, 2],
          "groups": 1 } }|};
  [%expect
    {|
    [eval] torch.ops.aten.conv2d.default
      aten   out0 = [4; 6; 4]
      native out0 = [4; 6; 4] |}]

let%expect_test "conv2d.padding: same padding matches native" =
  eval
    {|{ "target": "torch.ops.aten.conv2d.padding",
        "args": {
          "input": { "dtype": "f32", "shape": [1, 1, 3, 3], "sequence": { "start": 0.0, "step": 1.0 } },
          "weight": { "dtype": "f32", "shape": [1, 1, 3, 3], "values": [{"float":1},{"float":1},{"float":1},{"float":1},{"float":1},{"float":1},{"float":1},{"float":1},{"float":1}] },
          "bias": { "dtype": "f32", "shape": [1], "values": [{"float":0}] },
          "stride": [1, 1],
          "padding": "same",
          "dilation": [1, 1],
          "groups": 1 } }|};
  [%expect
    {|
    [eval] torch.ops.aten.conv2d.padding
      aten   out0 = [8; 15; 12; 21; 36; 27; 20; 33; 24]
      native out0 = [8; 15; 12; 21; 36; 27; 20; 33; 24] |}]

let%expect_test "convolution.default: non-transposed 2d matches native" =
  eval
    {|{ "target": "torch.ops.aten.convolution.default",
        "args": {
          "input": { "dtype": "f32", "shape": [1, 1, 3, 3], "sequence": { "start": 0.0, "step": 1.0 } },
          "weight": { "dtype": "f32", "shape": [1, 1, 2, 2], "values": [{"float":1},{"float":1},{"float":1},{"float":1}] },
          "bias": { "dtype": "f32", "shape": [1], "values": [{"float":0}] },
          "stride": [1, 1],
          "padding": [0, 0],
          "dilation": [1, 1],
          "transposed": false,
          "output_padding": [0, 0],
          "groups": 1 } }|};
  [%expect
    {|
    [eval] torch.ops.aten.convolution.default
      aten   out0 = [8; 12; 20; 24]
      native out0 = [8; 12; 20; 24] |}]

let%expect_test "convolution.default: dilated 2d matches native" =
  eval
    {|{ "target": "torch.ops.aten.convolution.default",
        "args": {
          "input": { "dtype": "f32", "shape": [1, 1, 1, 5], "values": [{"float":0},{"float":1},{"float":2},{"float":3},{"float":4}] },
          "weight": { "dtype": "f32", "shape": [1, 1, 1, 3], "values": [{"float":1},{"float":1},{"float":1}] },
          "bias": { "dtype": "f32", "shape": [1], "values": [{"float":0}] },
          "stride": [1, 1],
          "padding": [0, 1],
          "dilation": [1, 2],
          "transposed": false,
          "output_padding": [0, 0],
          "groups": 1 } }|};
  [%expect
    {|
    [eval] torch.ops.aten.convolution.default
      aten   out0 = [4; 6; 4]
      native out0 = [4; 6; 4] |}]

let%expect_test "convolution.default: transposed 2d matches native" =
  eval
    {|{ "target": "torch.ops.aten.convolution.default",
        "args": {
          "input": { "dtype": "f32", "shape": [1, 1, 2, 2], "values": [{"float":1},{"float":2},{"float":3},{"float":4}] },
          "weight": { "dtype": "f32", "shape": [1, 1, 2, 2], "values": [{"float":1},{"float":1},{"float":1},{"float":1}] },
          "bias": { "dtype": "f32", "shape": [1], "values": [{"float":0}] },
          "stride": [1, 1],
          "padding": [0, 0],
          "dilation": [1, 1],
          "transposed": true,
          "output_padding": [0, 0],
          "groups": 1 } }|};
  [%expect
    {|
    [eval] torch.ops.aten.convolution.default
      aten   out0 = [1; 3; 2; 4; 10; 6; 3; 7; 4]
      native out0 = [1; 3; 2; 4; 10; 6; 3; 7; 4] |}]

(* --- mean.dim: reduced axes via axis_of_dim, both keepdim flags --- *)

let%expect_test "mean.dim: dim=[1] keepdim=true" =
  eval
    {|{ "target": "torch.ops.aten.mean.dim",
        "args": {
          "self": { "dtype": "f32", "shape": [2, 3], "sequence": { "start": 0.0, "step": 1.0 } },
          "dim": [1],
          "keepdim": true } }|};
  [%expect
    {|
    [eval] torch.ops.aten.mean.dim
      aten   out0 = [1; 4]
      native out0 = [1; 4] |}]

let%expect_test "mean.dim: dim=[1] keepdim=false" =
  eval
    {|{ "target": "torch.ops.aten.mean.dim",
        "args": {
          "self": { "dtype": "f32", "shape": [2, 3], "sequence": { "start": 0.0, "step": 1.0 } },
          "dim": [1],
          "keepdim": false } }|};
  [%expect
    {|
    [eval] torch.ops.aten.mean.dim
      aten   out0 = [1; 4]
      native out0 = [1; 4] |}]

let%expect_test "mean.dim: dim=[] reduces over all dims" =
  eval
    {|{ "target": "torch.ops.aten.mean.dim",
        "args": {
          "self": { "dtype": "f32", "shape": [2, 3], "sequence": { "start": 0.0, "step": 1.0 } },
          "dim": [],
          "keepdim": false } }|};
  [%expect
    {|
    [eval] torch.ops.aten.mean.dim
      aten   out0 = [2.5]
      native out0 = [2.5] |}]

let%expect_test "mean.dim: omitted dim reduces over all dims" =
  eval
    {|{ "target": "torch.ops.aten.mean.dim",
        "args": {
          "self": { "dtype": "f32", "shape": [2, 3], "sequence": { "start": 0.0, "step": 1.0 } },
          "keepdim": false } }|};
  [%expect
    {|
    [eval] torch.ops.aten.mean.dim
      aten   out0 = [2.5]
      native out0 = [2.5] |}]

(* --- rms_norm: normalise over the trailing (innermost) axis, with weight --- *)

let%expect_test "rms_norm: normalized_shape=[3] with weight" =
  eval
    {|{ "target": "torch.ops.aten.rms_norm.default",
        "args": {
          "input": { "dtype": "f32", "shape": [2, 3], "sequence": { "start": 1.0, "step": 1.0 } },
          "normalized_shape": [3],
          "weight": { "dtype": "f32", "shape": [3], "values": [{"float":1}, {"float":1}, {"float":1}] },
          "eps": 9.9999997473787516e-06 } }|};
  [%expect
    {|
    [eval] torch.ops.aten.rms_norm.default
      aten   out0 = [0.46291; 0.925819; 1.38873; 0.789542; 0.986927; 1.18431]
      native out0 = [0.46291; 0.925819; 1.38873; 0.789542; 0.986927; 1.18431] |}]

let%expect_test "rms_norm: normalized_shape=[3] no weight (identity scale)" =
  eval
    {|{ "target": "torch.ops.aten.rms_norm.default",
        "args": {
          "input": { "dtype": "f32", "shape": [2, 3], "sequence": { "start": 1.0, "step": 1.0 } },
          "normalized_shape": [3],
          "eps": 9.9999997473787516e-06 } }|};
  [%expect
    {|
    [eval] torch.ops.aten.rms_norm.default
      aten   out0 = [0.46291; 0.925819; 1.38873; 0.789542; 0.986927; 1.18431]
      native out0 = [0.46291; 0.925819; 1.38873; 0.789542; 0.986927; 1.18431] |}]

(* --- unbind.int: the Tensor[] return through every spec-runner path -------

   Three separately-migrated readers meet here. [eval_print] prints VALUES via
   [pp_aten], so it is the one that would show a view read off the wrong part of
   its storage; [eval_report] and [walk_eval] read output NAMES via
   [Interp_decode.output_names], which has to flatten the single as_tensors
   output — an Argument.Tensor-only filter prints no output lines at all, which
   reads as a pass. *)

let unbind_spec =
  {|{ "target": "torch.ops.aten.unbind.int",
      "args": {
        "self": { "dtype": "f32", "shape": [3, 2], "sequence": { "start": 0.0, "step": 1.0 } } } }|}

(* [dim] is absent from the spec, as it is in a real exported node: the
   generated codec's ~dec_absent:(Int 0) supplies the schema default, which is
   why the report below echoes it back as dim=0. Each printed row is one result,
   and each is an OFFSET VIEW onto self's storage — every row showing [0; 1]
   would be the unmaterialized read. There is no native unbind. *)
let%expect_test "unbind: eval_print shows each result's own values" =
  eval unbind_spec;
  [%expect
    {|
    [eval] torch.ops.aten.unbind.int
      aten   out0 = [0; 1]
      native out0 = [0; 1]
      aten   out1 = [2; 3]
      native out1 = [2; 3]
      aten   out2 = [4; 5]
      native out2 = [4; 5] |}]

let%expect_test "unbind: eval_report lists every synthesized output name" =
  capture (fun ppf -> Aten_spec_run.eval_report ~ppf (decode unbind_spec));
  [%expect
    {|
    [node] torch.ops.aten.unbind.int(self=f32[3,2]~sequence(start=0,step=1), dim=0)
      -> out0: [2]
      -> out1: [2]
      -> out2: [2]
      status: ok |}]

(* The walk re-synthesizes and re-runs per step, so outputs_for derives the
   count afresh each time rather than reading it off the config. The per-result
   summaries are the other half: out1 reports min=2 max=3, not out0's 0/1, so
   [pp_tensor_summary] is summarizing each view's own elements and not its
   storage base. *)
let%expect_test "unbind: walk_eval re-derives the count each step" =
  capture (fun ppf ->
      ignore (Aten_spec_run.walk_eval ~ppf ~steps:3 (decode unbind_spec)));
  [%expect
    {|
    [step 1/3] modified axis: self
      torch.ops.aten.unbind.int(self=f32[3,2]~sequence(start=0,step=1), dim=0)
      -> out0: [2] min=0 max=1 mean=0.5
      -> out1: [2] min=2 max=3 mean=2.5
      -> out2: [2] min=4 max=5 mean=4.5
      status: ok
    [step 2/3] modified axis: self
      torch.ops.aten.unbind.int(self=f32[3,2]~sequence(start=0,step=1), dim=0)
      -> out0: [2] min=0 max=1 mean=0.5
      -> out1: [2] min=2 max=3 mean=2.5
      -> out2: [2] min=4 max=5 mean=4.5
      status: ok
    [step 3/3] modified axis: self
      torch.ops.aten.unbind.int(self=f32[3,2]~sequence(start=0,step=1), dim=0)
      -> out0: [2] min=0 max=1 mean=0.5
      -> out1: [2] min=2 max=3 mean=2.5
      -> out2: [2] min=4 max=5 mean=4.5
      status: ok |}]
