(* Strict schema-v1 wire behavior: canonical roundtrips and malformed input. *)

open Inference_compare_schema
module F = Schema_fixtures

let pp_wire_error = Wire.pp_error

(* Print, parse, decode, print again: the text must be a fixed point. *)
let roundtrip name decode encode v =
  let text = Wire_text.to_string (encode v) in
  match Wire_text.decode_string decode text with
  | Error e ->
      Format.printf "%s: DECODE ERROR %a@." name
        (Err.Error.pp_kind pp_wire_error)
        e
  | Ok v' ->
      let text' = Wire_text.to_string (encode v') in
      Format.printf "%s: %s@." name
        (if String.equal text text' then "roundtrip" else "DIFFERS")

let reject name decode text =
  match Wire_text.decode_string decode text with
  | Ok _ -> Format.printf "%s: ACCEPTED@." name
  | Error e -> Format.printf "%s: %a@." name (Err.Error.pp_kind pp_wire_error) e

let%expect_test "graph and mapping roundtrip" =
  roundtrip "graph_key" Graph_key.decode Graph_key.encode F.left_graph;
  roundtrip "graph_desc" Graph_desc.decode Graph_desc.encode F.graph_desc;
  roundtrip "mapping" Mapping_artifact.decode Mapping_artifact.encode F.mapping;
  [%expect
    {|
    graph_key: roundtrip
    graph_desc: roundtrip
    mapping: roundtrip |}]

let%expect_test "output occurrences keep their repeats" =
  let flat = Graph_desc.Output_tree.flatten F.graph_desc.output_tree in
  Format.printf "%a@."
    Fmt.(
      list ~sep:sp
        (using
           (fun (r : Tensor_ref.t) -> Id.Tensor_local.to_string r.local_id)
           string))
    flat;
  [%expect {|
    y y
    h |}]

let%expect_test "strict objects" =
  let d = Graph_key.decode in
  reject "duplicate key" d
    {|{"dialect":"a","dialect":"b","version":"v","graph_digest":"sha256:00"}|};
  reject "unknown member" d
    {|{"dialect":"a","version":"v","graph_digest":"sha256:00","extra":1}|};
  reject "missing member" d {|{"dialect":"a","version":"v"}|};
  reject "wrong sort" d {|[]|};
  reject "bad digest hex" d
    {|{"dialect":"a","version":"v","graph_digest":"sha256:0G"}|};
  reject "odd digest" d
    {|{"dialect":"a","version":"v","graph_digest":"sha256:abc"}|};
  reject "upper-case digest" d
    {|{"dialect":"a","version":"v","graph_digest":"sha256:AB"}|};
  reject "no algorithm" d
    {|{"dialect":"a","version":"v","graph_digest":"abcd"}|};
  reject "empty identifier" d
    {|{"dialect":"","version":"v","graph_digest":"sha256:ab"}|};
  [%expect
    {|
    duplicate key: duplicate key "dialect" at /
    unknown member: unknown member "extra" at /
    missing member: missing member "graph_digest" at /
    wrong sort: expected object at /
    bad digest hex: invalid digest (invalid encoding) at /graph_digest
    odd digest: invalid digest (invalid encoding) at /graph_digest
    upper-case digest: invalid digest (invalid encoding) at /graph_digest
    no algorithm: invalid digest (missing ':' separator) at /graph_digest
    empty identifier: dialect: empty at /dialect |}]

let%expect_test "unknown tags and versions" =
  reject "claim tag" Claim.decode {|{"tag":"nearly"}|};
  reject "claim without tag" Claim.decode {|{}|};
  reject "schema version" Schema_version.decode {|"2"|};
  [%expect
    {|
    claim tag: unknown claim tag "nearly" at /
    claim without tag: missing member "tag" at /
    schema version: unsupported schema version "2" at / |}]

let%expect_test "decimal integers" =
  let d = Wire.int64 in
  List.iter
    (fun s -> reject s d s)
    [
      {|"0"|};
      {|"-5"|};
      {|"9223372036854775807"|};
      {|"9223372036854775808"|};
      {|"0x10"|};
      {|"+5"|};
      {|"007"|};
      {|"-0"|};
      {|"1_0"|};
      {|""|};
      {|5|};
    ];
  [%expect
    {|
    "0": ACCEPTED
    "-5": ACCEPTED
    "9223372036854775807": ACCEPTED
    "9223372036854775808": invalid decimal "9223372036854775808" at /
    "0x10": invalid decimal "0x10" at /
    "+5": invalid decimal "+5" at /
    "007": invalid decimal "007" at /
    "-0": invalid decimal "-0" at /
    "1_0": invalid decimal "1_0" at /
    "": invalid decimal "" at /
    5: expected string at / |}]

let%expect_test "native int range" =
  let d = Wire.int in
  List.iter
    (fun s -> reject s d s)
    [ {|"1073741823"|}; {|"1073741824"|}; {|"-1073741823"|} ];
  [%expect
    {|
    "1073741823": ACCEPTED
    "1073741824": out of range at /
    "-1073741823": ACCEPTED |}]

let%expect_test "structural limits" =
  reject "negative extent" Tensor_spec.decode
    {|{"dtype":{"codec":"f32","parameters_digest":null},"logical_shape":["-1"],"layout_id":"r","value_semantics":true}|};
  reject "port out of range" Input_port.decode {|"65536"|};
  reject "port at bound" Input_port.decode {|"65535"|};
  reject "tensor list holding a tuple" Graph_desc.Output_tree.decode
    {|{"tag":"tensor_list","items":[{"tag":"tuple","items":[]}]}|};
  reject "empty all_of" Tensor_selector.decode {|{"tag":"all_of","items":[]}|};
  reject "empty any_of" Metric_selector.decode {|{"tag":"any_of","items":[]}|};
  [%expect
    {|
    negative extent: negative value at /logical_shape
    port out of range: out of range at /
    port at bound: ACCEPTED
    tensor list holding a tuple: out of range at /
    empty all_of: empty list at /
    empty any_of: empty list at / |}]

let%expect_test "nonfinite numbers never encode" =
  (try ignore (Wire.enc_float Float.nan)
   with Invalid_argument m -> Format.printf "nan: %s@." m);
  (try ignore (Wire.enc_float Float.infinity)
   with Invalid_argument m -> Format.printf "inf: %s@." m);
  [%expect
    {|
    nan: Wire.enc_float: nonfinite value must be tagged
    inf: Wire.enc_float: nonfinite value must be tagged |}]

let%expect_test "duplicate identifiers inside a list" =
  let m = Mapping_artifact.encode F.mapping in
  let text_of j = Wire_text.to_string j in
  (* Re-decode the fixture with its first unit repeated. *)
  let doubled =
    match m with
    | Jsont.Object (mems, meta) ->
        Jsont.Object
          ( List.map
              (fun ((n, nm), v) ->
                match (n, v) with
                | "units", Jsont.Array (u :: rest, am) ->
                    ((n, nm), Jsont.Array (u :: u :: rest, am))
                | _ -> ((n, nm), v))
              mems,
            meta )
    | j -> j
  in
  reject "repeated unit id" Mapping_artifact.decode (text_of doubled);
  reject "repeated recipe node" Recipe.decode
    {|{"id":"r","nodes":[
        {"id":"n","operation":{"tag":"concat","axis":"0"},"arguments":[]},
        {"id":"n","operation":{"tag":"concat","axis":"0"},"arguments":[]}],
       "result":{"tag":"conversion_result","node":"n"},
       "output_spec":{"dtype":{"codec":"f32","parameters_digest":null},
                      "logical_shape":[],"layout_id":"l","value_semantics":true},
       "lossless":true}|};
  [%expect
    {|
    repeated unit id: duplicate element at /units/1
    repeated recipe node: duplicate element at /nodes/1 |}]
