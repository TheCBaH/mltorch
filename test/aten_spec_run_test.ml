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
let capture f =
  let buf = Buffer.create 256 in
  let ppf = Format.formatter_of_buffer buf in
  f ppf;
  Format.pp_print_flush ppf ();
  print_string (Buffer.contents buf)

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
          "other": { "dtype": "f32", "shape": [4], "values": [{"float":"0x3e800000"}, {"float":"0x3e800000"}, {"float":"0x3e800000"}, {"float":"0x3e800000"}] } } }|};
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
          "self":  { "dtype": "f32", "shape": [2], "values": [{"float":"0x3f800000"}, {"float":"0x40000000"}] },
          "other": { "dtype": "f32", "shape": [2], "values": [{"float":"0x3f000000"}, {"float":"0x3f000000"}] } } }|};
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

(* --- rms_norm: normalise over the trailing (innermost) axis, with weight --- *)

let%expect_test "rms_norm: normalized_shape=[3] with weight" =
  eval
    {|{ "target": "torch.ops.aten.rms_norm.default",
        "args": {
          "input": { "dtype": "f32", "shape": [2, 3], "sequence": { "start": 1.0, "step": 1.0 } },
          "normalized_shape": [3],
          "weight": { "dtype": "f32", "shape": [3], "values": [{"float":"0x3f800000"}, {"float":"0x3f800000"}, {"float":"0x3f800000"}] },
          "eps": "0x3727c5ac" } }|};
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
          "eps": "0x3727c5ac" } }|};
  [%expect
    {|
    [eval] torch.ops.aten.rms_norm.default
      aten   out0 = [0.46291; 0.925819; 1.38873; 0.789542; 0.986927; 1.18431]
      native out0 = [0.46291; 0.925819; 1.38873; 0.789542; 0.986927; 1.18431] |}]
