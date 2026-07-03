(* Hermetic tests for the JSON verification format: no C++/ATen build needed.
   Covers PCG reproducibility + f32-canonical/finite draws, Float32 hex
   round-trip, and decode/encode round-trip of an op spec through the generated
   codec. *)

let pf = Printf.printf

(* --- Float32 bit-exact hex --- *)

let%expect_test "float32 hex round-trip" =
  List.iter
    (fun f ->
      let hex = Aten_spec.Float32.to_hex f in
      let back = Aten_spec.Float32.of_hex hex in
      (* round-trip recovers the f32-canonical value of f *)
      pf "%-12g %s eq=%b\n" f hex (back = Aten_spec.Float32.to_f32 f))
    [ 1.0; -0.5; 3.14159265; 0.0; 1e-20 ];
  (* to_f32 is idempotent on an already-f32 value *)
  let x = Aten_spec.Float32.to_f32 0.1 in
  pf "idempotent=%b\n" (x = Aten_spec.Float32.to_f32 x);
  [%expect
    {|
    1            0x3f800000 eq=true
    -0.5         0xbf000000 eq=true
    3.14159      0x40490fdb eq=true
    0            0x00000000 eq=true
    1e-20        0x1e3ce508 eq=true
    idempotent=true |}]

let encode codec v =
  match Jsont_bytesrw.encode_string codec v with
  | Ok s -> s
  | Error e -> failwith e

let decode codec s =
  match Jsont_bytesrw.decode_string codec s with
  | Ok v -> v
  | Error e -> failwith e

let%expect_test "float32 json encoding uses numbers except exceptional bits" =
  List.iter
    (fun (label, f) ->
      pf "%-10s %s -> %s\n" label
        (Aten_spec.Float32.to_hex f)
        (encode Aten_spec.Float32.jsont f))
    [
      ("one", 1.0);
      ("point-one", Aten_spec.Float32.to_f32 0.1);
      ("pi", Aten_spec.Float32.to_f32 Float.pi);
      ("tiny", Aten_spec.Float32.of_hex "0x00000001");
      ("neg-zero", Aten_spec.Float32.of_hex "0x80000000");
      ("inf", Aten_spec.Float32.of_hex "0x7f800000");
      ("nan", Aten_spec.Float32.of_hex "0x7fc00001");
    ];
  [%expect
    {|
    one        0x3f800000 -> 1
    point-one  0x3dcccccd -> 0.10000000149011612
    pi         0x40490fdb -> 3.1415927410125732
    tiny       0x00000001 -> 1.4012984643248171e-45
    neg-zero   0x80000000 -> "0x80000000"
    inf        0x7f800000 -> "0x7f800000"
    nan        0x7fc00001 -> "0x7fc00001" |}]

let%expect_test "float32 json decoding accepts numbers and raw hex" =
  List.iter
    (fun json ->
      let f = decode Aten_spec.Float32.jsont json in
      pf "%s -> %s\n" json (Aten_spec.Float32.to_hex f))
    [ "1.0"; "\"0x3f800000\""; "0.1"; "\"0x3dcccccd\"" ];
  [%expect
    {|
    1.0 -> 0x3f800000
    "0x3f800000" -> 0x3f800000
    0.1 -> 0x3dcccccd
    "0x3dcccccd" -> 0x3dcccccd |}]

let%expect_test "tensor values decode both float forms and encode compactly" =
  let t =
    decode Aten_spec.Tensor_spec.jsont
      {|{"dtype":"f32","shape":[5],"values":[1.0,{"float":0.5},{"float":"0x80000000"},{"float":"0x7f800000"},{"float":"0x7fc00001"}]}|}
  in
  pf "%s\n" (encode Aten_spec.Tensor_spec.jsont t);
  [%expect
    {| {"dtype":"f32","shape":[5],"values":[{"float":1},{"float":0.5},{"float":"0x80000000"},{"float":"0x7f800000"},{"float":"0x7fc00001"}]} |}]

(* --- PCG: reproducible, f32-canonical, finite --- *)

let draws_uniform seed n =
  let rec go i pcg acc =
    if i = 0 then List.rev acc
    else
      let v, pcg = Aten_spec.Pcg.uniform ~low:(-1.0) ~high:1.0 pcg in
      go (i - 1) pcg (v :: acc)
  in
  go n (Aten_spec.Pcg.seed ~seed ~seq:1L) []

let%expect_test "pcg uniform reproducible and bounded" =
  let a = draws_uniform 42L 6 in
  let b = draws_uniform 42L 6 in
  pf "same seed equal=%b\n" (a = b);
  pf "different seed differs=%b\n" (draws_uniform 7L 6 <> a);
  let all_f32 = List.for_all (fun v -> v = Aten_spec.Float32.to_f32 v) a in
  let in_range = List.for_all (fun v -> v >= -1.0 && v < 1.0) a in
  pf "f32-canonical=%b in[-1,1)=%b\n" all_f32 in_range;
  List.iter (fun v -> pf "%s\n" (Aten_spec.Float32.to_hex v)) a;
  [%expect
    {|
    same seed equal=true
    different seed differs=true
    f32-canonical=true in[-1,1)=true
    0xbec838d0
    0x3f4b070e
    0xbe9c4988
    0x3f67c6f6
    0x3f4ecc86
    0xbe2810e0 |}]

let%expect_test "pcg normal is finite and f32-canonical" =
  let rec go i pcg ok =
    if i = 0 then ok
    else
      let v, pcg = Aten_spec.Pcg.normal ~mean:0.0 ~std:1.0 pcg in
      go (i - 1) pcg (ok && Float.is_finite v && v = Aten_spec.Float32.to_f32 v)
  in
  pf "finite+f32 over 1000 draws=%b\n"
    (go 1000 (Aten_spec.Pcg.seed ~seed:1L ~seq:1L) true);
  [%expect {| finite+f32 over 1000 draws=true |}]

(* --- decode / encode round-trip through the generated codec --- *)

let spec_json =
  {|{ "target": "torch.ops.aten.add.Tensor",
      "args": {
        "self":  { "dtype": "f32", "shape": [2, 3], "random": { "uniform": { "low": -1.0, "high": 1.0 } } },
        "other": { "dtype": "f32", "shape": [2, 3], "random": { "normal":  { "mean": 0.0, "variance": 1.0 } } }
      } }|}

let%expect_test "op spec decode/encode round-trip" =
  match Jsont_bytesrw.decode_string Aten_op_spec.jsont spec_json with
  | Error e -> pf "decode error: %s\n" e
  | Ok spec ->
      pf "target=%s nargs=%d\n" spec.target (List.length spec.args);
      pf "alpha-defaulted=%b\n" (List.mem_assoc "alpha" spec.args);
      (* encode, then decode again, and confirm stability *)
      (match Jsont_bytesrw.encode_string Aten_op_spec.jsont spec with
      | Error e -> pf "encode error: %s\n" e
      | Ok s2 -> (
          match Jsont_bytesrw.decode_string Aten_op_spec.jsont s2 with
          | Error e -> pf "re-decode error: %s\n" e
          | Ok spec2 ->
              pf "stable target=%b nargs=%b\n"
                (spec.target = spec2.target)
                (List.length spec.args = List.length spec2.args)));
      [%expect
        {|
        target=torch.ops.aten.add.Tensor nargs=3
        alpha-defaulted=true
        stable target=true nargs=true |}]

(* Decode -> encode -> decode and report target + arg count stability, so the
   non-tensor arg shapes (int[] dims, bool keepdim, float eps, optional
   weight) of the newly-bridged ops are exercised end to end. *)
let roundtrip json =
  match Jsont_bytesrw.decode_string Aten_op_spec.jsont json with
  | Error e -> pf "decode error: %s\n" e
  | Ok spec -> (
      match Jsont_bytesrw.encode_string Aten_op_spec.jsont spec with
      | Error e -> pf "encode error: %s\n" e
      | Ok s2 -> (
          match Jsont_bytesrw.decode_string Aten_op_spec.jsont s2 with
          | Error e -> pf "re-decode error: %s\n" e
          | Ok spec2 ->
              pf "target=%s nargs=%d stable=%b\n" spec.target
                (List.length spec.args)
                (spec.target = spec2.target
                && List.length spec.args = List.length spec2.args)))

let%expect_test "bmm round-trip" =
  roundtrip
    {|{ "target": "torch.ops.aten.bmm.default",
        "args": {
          "self": { "dtype": "f32", "shape": [1, 2, 2], "sequence": { "start": 1.0, "step": 1.0 } },
          "mat2": { "dtype": "f32", "shape": [1, 2, 2], "sequence": { "start": 1.0, "step": 1.0 } } } }|};
  [%expect {| target=torch.ops.aten.bmm.default nargs=2 stable=true |}]

let%expect_test "conv2d.padding round-trip (string padding)" =
  roundtrip
    {|{ "target": "torch.ops.aten.conv2d.padding",
        "args": {
          "input": { "dtype": "f32", "shape": [1, 1, 3, 3], "sequence": { "start": 0.0, "step": 1.0 } },
          "weight": { "dtype": "f32", "shape": [1, 1, 3, 3], "values": [{"float":1},{"float":1},{"float":1},{"float":1},{"float":1},{"float":1},{"float":1},{"float":1},{"float":1}] },
          "bias": { "dtype": "f32", "shape": [1], "values": [{"float":0}] },
          "stride": [1, 1],
          "padding": "same",
          "dilation": [1, 1],
          "groups": 1 } }|};
  [%expect {| target=torch.ops.aten.conv2d.padding nargs=7 stable=true |}]

let%expect_test "convolution.default round-trip" =
  roundtrip
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
  [%expect {| target=torch.ops.aten.convolution.default nargs=9 stable=true |}]

let%expect_test "mean.dim round-trip (dim + keepdim)" =
  roundtrip
    {|{ "target": "torch.ops.aten.mean.dim",
        "args": {
          "self": { "dtype": "f32", "shape": [2, 3], "sequence": { "start": 0.0, "step": 1.0 } },
          "dim": [1],
          "keepdim": true } }|};
  [%expect {| target=torch.ops.aten.mean.dim nargs=3 stable=true |}]

let%expect_test "rms_norm round-trip (normalized_shape + weight + eps)" =
  roundtrip
    {|{ "target": "torch.ops.aten.rms_norm.default",
        "args": {
          "input": { "dtype": "f32", "shape": [2, 3], "sequence": { "start": 1.0, "step": 1.0 } },
          "normalized_shape": [3],
          "weight": { "dtype": "f32", "shape": [3], "values": [{"float":1}, {"float":1}, {"float":1}] },
          "eps": 9.9999999392252903e-09 } }|};
  [%expect {| target=torch.ops.aten.rms_norm.default nargs=4 stable=true |}]
