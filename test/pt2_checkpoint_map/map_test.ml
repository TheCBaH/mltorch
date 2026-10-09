module Map = Pt2_checkpoint_map
open Fixtures

(* A third-party (Jsont) message carries terminal styling and source positions
   on further lines; the first line, unstyled, is the part under test. *)
let unstyle text =
  let b = Buffer.create (String.length text) in
  let n = String.length text in
  let i = ref 0 in
  while !i < n do
    if text.[!i] = '\027' then begin
      while !i < n && text.[!i] <> 'm' do
        incr i
      done;
      incr i
    end
    else begin
      Buffer.add_char b text.[!i];
      incr i
    end
  done;
  Buffer.contents b

let show_fault f =
  let text = unstyle (Fmt.str "%a" Map.Fault.pp_error f) in
  print_endline
    (match String.index_opt text '\n' with
    | Some i -> String.sub text 0 i
    | None -> text)

(* Print the fault of a failed result, or [ok]. *)
let report = function Ok _ -> print_endline "ok" | Error e -> show_fault e
let decode ?limits s = Err.payload (Map.Document.of_string ?limits s)
let decode_map ?limits s = report (decode ?limits s)
let check_ok doc g = report (Err.payload (Map.Validate.check doc g))

let document () =
  match decode map_json with
  | Ok d -> d
  | Error e ->
      show_fault e;
      failwith "the baseline map must decode"

let%expect_test "the baseline map decodes and validates" =
  let doc = document () in
  Fmt.pr "%d captures, %d checkpoint file, graph-owned %b@."
    (Schema_runtime.String_map.cardinal doc.tensors)
    (List.length doc.checkpoint_files)
    (Option.is_some doc.graph_owned);
  Schema_runtime.String_map.iter
    (fun target (e : Map.Document.Entry.t) ->
      Fmt.pr "%s %a elements=%Ld bytes=%Ld %s@." target Map.Dtype.pp e.dtype
        (Map.Document.Entry.element_count e)
        (Map.Document.Entry.byte_count e)
        (match e.origin with
        | Checkpoint { convert = Identity; _ } -> "checkpoint"
        | Checkpoint { convert = Cast { from; to_ }; _ } ->
            Fmt.str "checkpoint %a->%a" Map.Dtype.pp from Map.Dtype.pp to_
        | Empty -> "empty"
        | Fill raw -> "fill " ^ string_of_int (String.length raw)
        | Inline raw -> "inline " ^ string_of_int (String.length raw)
        | Pack key -> "pack " ^ key))
    doc.tensors;
  check_ok doc (graph ());
  [%expect
    {|
    6 captures, 1 checkpoint file, graph-owned true
    b I64 elements=3 bytes=24 fill 8
    c F32 elements=1 bytes=4 inline 4
    e F32 elements=0 bytes=0 empty
    h F32 elements=2 bytes=8 checkpoint BF16->F32
    k F32 elements=2 bytes=8 pack k
    w F32 elements=6 bytes=24 checkpoint
    ok
    |}]

(* Each corruption must change the bytes -- [replace] raises when its target is
   absent -- and must be refused at the intended check. *)
let%expect_test "document-level corruption" =
  let t sub by = decode_map (replace ~sub ~by map_json) in
  t {|"schema_version":2|} {|"schema_version":1|};
  t {|"unmapped":[]|} {|"unmapped":["x"]|};
  t {|"model_id":"toy"|} {|"model_id":"toy","extra":1|};
  t {|"tied_aliases":[]|} {|"tied_aliases":[],"extra":1|};
  t {|"kind":"pack"|} {|"kind":"mystery"|};
  t {|"dtype":"F32","shape":[2,3]|} {|"dtype":"F31","shape":[2,3]|};
  t {|"shape":[2,3]|} {|"shape":[2,-3]|};
  t {|"shape":[2,3]|} {|"shape":[2,3,1,1,1,1,1,1,1]|};
  t {|"shape":[2,3]|}
    {|"shape":[9007199254740991,9007199254740991,9007199254740991]|};
  t {|"shape":[2,3]|} {|"shape":[1099511627777,1]|};
  t sha_w (String.uppercase_ascii sha_w);
  t sha_w "abc";
  [%expect
    {|
    unsupported checkpoint map schema_version 1
    the map lists 1 unmapped capture(s): x
    failed to decode the checkpoint map: Unexpected member extra for checkpoint map object
    failed to decode the checkpoint map: Unexpected member extra for checkpoint origin object
    failed to decode the checkpoint map: Unexpected member kind value in origin object: <tag>.
    unknown dtype code "F31"
    capture "w": a shape extent is negative
    tensor rank is 9, over the limit 8
    tensor byte size is 9223372036854775807, over the limit 1099511627776
    tensor byte size is 4398046511108, over the limit 1099511627776
    "E2C0A71510B5394DF7773B63FB5F54372B84C3564E67811BDE7D665BE227976D" is not a 64-digit lower-case sha256
    "abc" is not a 64-digit lower-case sha256 |}]

let%expect_test "origins are consistent with their entries" =
  let t sub by = decode_map (replace ~sub ~by map_json) in
  (* fill: element width, hex spelling *)
  t {|"element_hex":"0100000000000000"|} {|"element_hex":"010000000000000000"|};
  print_endline "-- control: the unchanged map";
  t {|"element_hex":"0100000000000000"|} {|"element_hex":"0100000000000000"|};
  t {|"element_hex":"0100000000000000"|} {|"element_hex":"01000000000000G0"|};
  t {|"element_hex":"0100000000000000"|} {|"element_hex":"0100000000000000A0"|};
  t {|"element_hex":"0100000000000000"|} {|"element_hex":""|};
  (* empty must have no elements *)
  t {|"shape":[0],"sha256":|} {|"shape":[1],"sha256":|};
  (* inline: size, canonical base64 *)
  t {|"data_base64":"AAAgQA=="|} {|"data_base64":"AAAgQAAA"|};
  t {|"data_base64":"AAAgQA=="|} {|"data_base64":"AAAgQB=="|};
  t {|"data_base64":"AAAgQA=="|} {|"data_base64":"AAAgQA"|};
  t {|"data_base64":"AAAgQA=="|} {|"data_base64":"AAAg QA=="|};
  t {|"data_base64":"AAAgQA=="|} {|"data_base64":"AA=gQA=="|};
  t {|"data_base64":"AAAgQA=="|} {|"data_base64":"AAAgQA=A"|};
  (* casts *)
  t {|"from":"BF16","to":"F32"|} {|"from":"BF16","to":"F16"|};
  t {|"from":"BF16","to":"F32"|} {|"from":"F32","to":"F32"|};
  t {|"from":"BF16","to":"F32"|} {|"from":"BF17","to":"F32"|};
  t {|"convert":{"op":"none"}|} {|"convert":{"op":"none","from":"F32"}|};
  t {|"convert":{"op":"none"}|} {|"convert":{"op":"squash"}|};
  (* sources *)
  t {|"file":"toy.safetensors","key":"model.w"|}
    {|"file":"nope","key":"model.w"|};
  t {|"graph_owned":{"name":"pack.safetensors"|}
    {|"graph_owned":{"name":"toy.safetensors"|};
  [%expect
    {|
    capture "b": fill element is 9 bytes, dtype needs 8
    -- control: the unchanged map
    ok
    capture "b": fill element is not lower-case hex
    capture "b": fill element is not lower-case hex
    capture "b": fill element is not lower-case hex
    capture "e": generated empty but the shape has elements
    capture "c": inline data is 6 bytes, shape needs 4
    capture "c": inline data is not canonical base64
    capture "c": inline data is not canonical base64
    capture "c": inline data is not canonical base64
    capture "c": inline data is not canonical base64
    capture "c": inline data is not canonical base64
    capture "h": cast BF16 to F16 does not produce the declared dtype
    capture "h": cast F32 to F32 does not produce the declared dtype
    unknown dtype code "BF17"
    capture "w": malformed conversion
    capture "w": malformed conversion
    capture "w" reads "nope", which no source declares
    duplicate checkpoint file "toy.safetensors" |}]

let%expect_test "pack needs the graph-owned source and pins are well formed" =
  decode_map
    (replace ~sub:(jstr {|,"graph_owned":%s|} (pack_pin ())) ~by:"" map_json);
  let t sub by = decode_map (replace ~sub ~by map_json) in
  t {|"name":"toy.safetensors"|} {|"name":"a/b"|};
  t {|"name":"toy.safetensors"|} {|"name":".."|};
  t {|"name":"toy.safetensors"|} {|"name":""|};
  t
    (jstr {|"size":%d,"url":"https://example.org/toy.safetensors"|}
       (String.length toy_bytes))
    {|"size":0,"url":"https://example.org/toy.safetensors"|};
  t {|"url":"https://example.org/toy.safetensors"|}
    {|"url":"http://example.org/toy.safetensors"|};
  t rev "abc";
  t rev (String.uppercase_ascii rev);
  t {|,"repo_id":"o/toy","revision":"|} {|,"revision":"|};
  t {|"name":"pack.safetensors"|} {|"name":"toy.safetensors"|};
  t {|"graph_sha256":"|} {|"graph_sha256":"0|};
  [%expect
    {|
    capture "k" reads the graph-owned file, which the map omits
    invalid pinned-file name "a/b"
    invalid pinned-file name ".."
    invalid pinned-file name ""
    invalid pinned-file size "0"
    invalid pinned-file url "http://example.org/toy.safetensors"
    invalid pinned-file revision "abc"
    invalid pinned-file revision "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    invalid pinned-file provenance (repo_id and revision, or derived_from) "toy.safetensors"
    duplicate checkpoint file "toy.safetensors"
    "02366c272d3d3a041a6504a897864deba52d40ac66fcd8842ca1b51b1facacea8" is not a 64-digit lower-case sha256 |}]

let%expect_test "duplicates and limits" =
  (* The same member twice is refused, not collapsed into the later one. *)
  let doc =
    replace ~sub:{|"tensors":{|}
      ~by:
        (jstr
           {|"tensors":{"w":{"dtype":"F32","shape":[2,3],"sha256":%S,"origin":{"kind":"generated","op":"empty"}},|}
           sha_w)
      map_json
  in
  decode_map doc;
  let limits = { Map.Limits.default with max_captures = 5 } in
  decode_map ~limits map_json;
  decode_map
    ~limits:{ Map.Limits.default with max_checkpoint_files = 0 }
    map_json;
  decode_map
    ~limits:{ Map.Limits.default with max_document_bytes = 100 }
    map_json;
  decode_map ~limits:{ Map.Limits.default with max_inline_bytes = 3 } map_json;
  decode_map ~limits:{ Map.Limits.default with max_rank = 1 } map_json;
  decode_map ~limits:{ Map.Limits.default with max_tensor_bytes = 20L } map_json;
  [%expect
    {|
    duplicate capture "w"
    capture count is 6, over the limit 5
    checkpoint file count is 1, over the limit 0
    document size is 1782, over the limit 100
    inline value size is 4, over the limit 3
    tensor rank is 2, over the limit 1
    tensor byte size is 24, over the limit 20 |}]

(* --- against the graph --- *)

let validate ?(map = map_json) g =
  match decode map with Error e -> show_fault e | Ok doc -> check_ok doc g

let cap_h =
  capture ~target:"h" ~kind:"PARAMETER" ~dtype:"float32" ~shape:"2" ~sha:sha_h

let%expect_test "map against the graph" =
  let captures = captures_json program_json in
  let with_captures f = graph ~captures:(f captures) () in
  let program_digest =
    Pt2_sha256.Digest.to_hex (Pt2_sha256.string program_json)
  in
  print_endline "-- identity";
  validate (graph ~artifact_id:"other/artifact" ());
  validate
    (graph
       ~captures:(captures_json ~artifact_id:"other/artifact" program_json)
       ());
  print_endline "-- graph bytes";
  validate
    (graph
       ~program:(replace ~sub:{|"minor":5|} ~by:{|"minor":6|} program_json)
       ~captures ());
  validate
    (with_captures (replace ~sub:program_digest ~by:(String.make 64 '1')));
  print_endline "-- coverage";
  validate (with_captures (replace ~sub:{|"target":"w"|} ~by:{|"target":"w2"|}));
  validate
    (with_captures
       (replace ~sub:{|],"counts"|}
          ~by:
            (","
            ^ capture ~target:"z" ~kind:"PARAMETER" ~dtype:"float32" ~shape:"2"
                ~sha:sha_k
            ^ {|],"counts"|})));
  validate
    ~map:(replace ~sub:{|"w":{"dtype"|} ~by:{|"w2":{"dtype"|} map_json)
    (graph ());
  validate
    ~map:
      (replace ~sub:{|"tensors":{|}
         ~by:
           (jstr
              {|"tensors":{"z":{"dtype":"F32","shape":[2],"sha256":%S,"origin":{"kind":"pack","key":"z"}},|}
              sha_k)
         map_json)
    (graph ());
  print_endline "-- configs";
  validate
    (graph
       ~weights:(replace ~sub:{|"w":{|} ~by:{|"w9":{|} (weights_json ()))
       ());
  validate
    (graph ~constants:(replace ~sub:{|"b":{|} ~by:{|"w":{|} constants_json) ());
  validate
    (graph
       ~weights:
         (replace ~sub:{|"h":{|}
            ~by:
              {|"zz":{"path_name":"x","is_param":true,"use_pickle":false,"tensor_meta":{"dtype":7,"sizes":[{"as_int":2}],"requires_grad":false,"device":{"type":"cpu","index":null},"strides":[{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}},"h":{|}
            (weights_json ()))
       ());
  [%expect
    {|
    -- identity
    the map is for artifact "toy/task/reference/forward/fp32/dynamo/static/ckpt-aaaaaaaaaaaa", not "other/artifact"
    captures.json is for artifact "other/artifact", not "toy/task/reference/forward/fp32/dynamo/static/ckpt-aaaaaaaaaaaa"
    -- graph bytes
    graph bytes hash to 6fd9205743a4c1820b1d3d4c9495f6df005b0672523734ce47b9091d485098ca, the map pins 2366c272d3d3a041a6504a897864deba52d40ac66fcd8842ca1b51b1facacea8
    graph bytes hash to 2366c272d3d3a041a6504a897864deba52d40ac66fcd8842ca1b51b1facacea8, captures.json pins 1111111111111111111111111111111111111111111111111111111111111111
    -- coverage
    graph capture "w" is not in captures.json
    captures.json lists "z", which the graph does not capture
    capture "w" has no entry in the map
    map entry "z" is not a capture of the graph
    -- configs
    a payload config lists "w9", which the graph does not capture
    capture "b" is in neither weights nor constants config
    a payload config lists "zz", which the graph does not capture |}]

let%expect_test "each capture against the graph, the inventory and its digest" =
  print_endline "-- kind";
  validate
    (graph
       ~captures:
         (replace ~sub:cap_h
            ~by:
              (capture ~target:"h" ~kind:"BUFFER" ~dtype:"float32" ~shape:"2"
                 ~sha:sha_h)
            (captures_json program_json))
       ());
  print_endline "-- dtype: config, then inventory";
  validate
    (graph
       ~weights:
         (replace
            ~sub:
              {|"h":{"path_name":"p_h","is_param":true,"use_pickle":false,"tensor_meta":{"dtype":7|}
            ~by:
              {|"h":{"path_name":"p_h","is_param":true,"use_pickle":false,"tensor_meta":{"dtype":5|}
            (weights_json ()))
       ());
  validate
    (graph
       ~captures:
         (replace ~sub:cap_h
            ~by:
              (capture ~target:"h" ~kind:"PARAMETER" ~dtype:"int64" ~shape:"2"
                 ~sha:sha_h)
            (captures_json program_json))
       ());
  print_endline "-- shape: config, then inventory";
  validate (graph ~weights:(weights_json ~h_sizes:[ 3 ] ()) ());
  validate
    (graph
       ~captures:
         (replace ~sub:cap_h
            ~by:
              (capture ~target:"h" ~kind:"PARAMETER" ~dtype:"float32" ~shape:"3"
                 ~sha:sha_h)
            (captures_json program_json))
       ());
  print_endline "-- layout";
  validate (graph ~weights:(weights_json ~w_strides:[ 1; 2 ] ()) ());
  print_endline "-- value digest";
  validate
    (graph
       ~captures:
         (replace ~sub:cap_h
            ~by:
              (capture ~target:"h" ~kind:"PARAMETER" ~dtype:"float32" ~shape:"2"
                 ~sha:sha_k)
            (captures_json program_json))
       ());
  print_endline "-- conversions";
  validate
    ~map:(replace ~sub:{|"from":"BF16"|} ~by:{|"from":"F16"|} map_json)
    (graph ());
  validate
    ~map:(replace ~sub:{|"from":"BF16"|} ~by:{|"from":"F64"|} map_json)
    (graph ());
  validate
    ~map:(replace ~sub:{|"from":"BF16"|} ~by:{|"from":"I8"|} map_json)
    (graph ());
  [%expect
    {|
    -- kind
    capture "h": per the graph signature kind is PARAMETER, captures.json says BUFFER
    -- dtype: config, then inventory
    capture "h": per the graph config dtype is I64, map says F32
    capture "h": per captures.json dtype is I64, map says F32
    -- shape: config, then inventory
    capture "h": per the graph config shape is [3], map says [2]
    capture "h": per captures.json shape is [3], map says [2]
    -- layout
    capture "w": graph tensor is not a dense row-major buffer
    -- value digest
    capture "h": per captures.json digest is 621e6b9d912e2d0c9b2fc35bfd56ec75345026a6e61d0821763646eabacf47aa, map says 252b3318179cc24998f3670913d52d39085cf65b0dfa98fa523ffeab4b6683fe
    -- conversions
    ok
    capture "h": conversion F64 to F32 is not implemented
    capture "h": conversion I8 to F32 is not implemented |}]

let%expect_test "a repeated signature capture is refused" =
  let dup =
    replace ~sub:{|{"parameter":{"arg":{"name":"h"},"parameter_name":"h"}}|}
      ~by:{|{"parameter":{"arg":{"name":"h"},"parameter_name":"w"}}|}
      program_json
  in
  let digest = Pt2_sha256.Digest.to_hex (Pt2_sha256.string dup) in
  let map =
    replace
      ~sub:(Pt2_sha256.Digest.to_hex (Pt2_sha256.string program_json))
      ~by:digest map_json
  in
  validate ~map (graph ~program:dup ~captures:(captures_json dup) ());
  [%expect {| duplicate capture "w" |}]

let%expect_test "strict codecs" =
  let show f x =
    match f x with
    | None -> print_endline "none"
    | Some s -> Printf.printf "%S\n" s
  in
  print_endline "-- hex";
  List.iter (show Map.Codec.hex_decode) [ "00ff"; "0A"; "0"; ""; "zz"; "0g" ];
  print_endline "-- base64";
  List.iter
    (show Map.Codec.base64_decode)
    [
      "";
      "TWFu";
      "TWE=";
      "TQ==";
      "TQ=";
      "TR==";
      "TWF=";
      "T===";
      "TWFu\n";
      "TW-u";
    ];
  [%expect
    {|
    -- hex
    "\000\255"
    none
    none
    none
    none
    none
    -- base64
    ""
    "Man"
    "Ma"
    "M"
    none
    none
    none
    none
    none
    none |}]
