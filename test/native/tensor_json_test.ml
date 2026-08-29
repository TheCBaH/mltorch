(* Tensor payload JSON codec: Array encoding, encode -> decode -> verify,
   int64 bit-exactness, elision past max_elts, decoding a literal, and f32
   special values. Split from graph_json_test.ml. *)

open Graph_json_fixtures

(* ---- tensor payload ------------------------------------------------------- *)

let%expect_test "tensor: encode with Array payload" =
  let t = Tensor.materialize (s1c 4) (fun c -> float_of_int (chan c)) in
  Format.printf "%a@." (pp_result Format.pp_print_string) (encode_tensor t);
  [%expect
    {|
    {
      "data": {
        "Array": [
          0,
          1,
          2,
          3
        ]
      },
      "fmt": "f32",
      "quant": null,
      "shape": [
        1,
        1,
        1,
        1,
        1,
        4
      ]
    } |}]

let%expect_test "tensor: encode → decode → verify values" =
  let original =
    Tensor.materialize (s1c 4) (fun c -> float_of_int (chan c) *. 0.5)
  in
  let result =
    let open Err.Syntax in
    let* json = encode_tensor original in
    let* decoded = decode_tensor json in
    Err.return (original, decoded)
  in
  Format.printf "%a@." (pp_result pp_original_and_tensor) result;
  [%expect
    {|
    original: tensor f32 [C=4] {0, 0.5, 1, 1.5}
    decoded:  tensor f32 [C=4] {0, 0.5, 1, 1.5} |}]

let%expect_test "tensor: int64 JSON round-trip preserves all bits" =
  let data =
    Bigarray.(
      Array1.of_array int64 c_layout
        [| Int64.min_int; 9_007_199_254_740_993L; Int64.max_int |])
  in
  let original =
    Tensor.Tensor
      {
        Tensor.shape = s1c 3;
        payload = { Payload.fmt = Payload.I64; quant = Payload.No_quant; data };
      }
  in
  let result =
    let open Err.Syntax in
    let* json = encode_tensor original in
    let* decoded = decode_tensor json in
    Err.return (json, decoded)
  in
  Format.printf "%a@."
    (pp_result (fun ppf (json, decoded) ->
         Format.fprintf ppf "%s@.%a" json Tensor.pp decoded))
    result;
  [%expect
    {|
    {
      "data": {
        "Array": [
          "-9223372036854775808",
          "9007199254740993",
          "9223372036854775807"
        ]
      },
      "fmt": "i64",
      "quant": null,
      "shape": [
        1,
        1,
        1,
        1,
        1,
        3
      ]
    }
    tensor i64 [C=3] {-9223372036854775808, 9007199254740993, 9223372036854775807} |}]

let%expect_test "tensor: payload elided when numel exceeds max_elts" =
  let t = Tensor.materialize (s1c 8) (fun c -> float_of_int (chan c)) in
  let result =
    let open Err.Syntax in
    let* full_json = encode_tensor t in
    let* elided_json = encode_tensor ~max_elts:4 t in
    Err.return (full_json, elided_json)
  in
  Format.printf "%a@." (pp_result pp_full_and_elided) result;
  [%expect
    {|
    full (8 elts):
    {
      "data": {
        "Array": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7
        ]
      },
      "fmt": "f32",
      "quant": null,
      "shape": [
        1,
        1,
        1,
        1,
        1,
        8
      ]
    }
    elided (max_elts=4):
    {
      "data": {
        "None": null
      },
      "fmt": "f32",
      "quant": null,
      "shape": [
        1,
        1,
        1,
        1,
        1,
        8
      ]
    } |}]

let%expect_test "tensor: decode JSON literal" =
  let json =
    {|{"data":{"Array":[1,2,3]},"fmt":"f32","quant":null,"shape":[1,1,1,1,1,3]}|}
  in
  Format.printf "%a@." (pp_result Tensor.pp) (decode_tensor json);
  [%expect {| tensor f32 [C=3] {1, 2, 3} |}]

let%expect_test "tensor: f32 special values roundtrip" =
  let t =
    Tensor.materialize (s1c 3) (fun c ->
        match chan c with 0 -> Float.nan | 1 -> 1.0 /. 0.0 | _ -> -0.0)
  in
  Format.printf "%a@." (pp_result Format.pp_print_string) (encode_tensor t);
  [%expect
    {|
    {
      "data": {
        "Array": [
          "0x7fc00000",
          "0x7f800000",
          "0x80000000"
        ]
      },
      "fmt": "f32",
      "quant": null,
      "shape": [
        1,
        1,
        1,
        1,
        1,
        3
      ]
    } |}]

(* A multi-output node has to survive the round trip with EVERY output edge, not
   just the producer of the graph output: the codec stores [Node.outputs] as a
   list, and a node that decoded back to one edge would still print and evaluate
   until something read the missing slices. Two of the three are dead here, so
   only the arity check in [Graph_view] and this golden would notice. *)
