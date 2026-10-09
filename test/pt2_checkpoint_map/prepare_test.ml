module Map = Pt2_checkpoint_map
open Fixtures

let unstyle = Map_test.unstyle

let report = function
  | Ok _ -> print_endline "ok"
  | Error e -> Map_test.show_fault e

let bigstring_of s : Safetensors.Bigstring.t =
  let b =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout (String.length s)
  in
  String.iteri (Bigarray.Array1.set b) s;
  b

let to_string (b : Pt2_storage.t) =
  String.init (Bigarray.Array1.dim b) (Bigarray.Array1.get b)

let document ?(map = map_json) () =
  match Err.payload (Map.Document.of_string map) with
  | Ok d -> d
  | Error e ->
      Map_test.show_fault e;
      failwith "map must decode"

let sources ?(toy = toy_bytes) ?(pack = pack_bytes) () =
  [
    { Map.Prepare.name = "toy.safetensors"; bytes = bigstring_of toy };
    { Map.Prepare.name = "pack.safetensors"; bytes = bigstring_of pack };
  ]

let prepare ?limits ?map ?(srcs = sources ()) () =
  let doc = document ?map () in
  Err.payload
    (let open Err.Syntax in
     let* v = Map.Prepare.verify_sources ?limits doc srcs in
     Map.Prepare.capture_set ?limits doc v)

let hex_bytes s =
  String.concat " "
    (List.init (String.length s) (fun i ->
         Printf.sprintf "%02x" (Char.code s.[i])))

let%expect_test "every origin prepares and matches its independent digest" =
  (match prepare () with
  | Error e -> Map_test.show_fault e
  | Ok set ->
      Fmt.pr "targets: %s@." (String.concat " " (Map.Prepare.targets set));
      Fmt.pr "owned %Ld, sources %Ld (file sizes %d + %d)@."
        (Map.Prepare.owned_bytes set)
        (Map.Prepare.source_bytes set)
        (String.length toy_bytes) (String.length pack_bytes);
      List.iter
        (fun t ->
          let s = Option.get (Map.Prepare.find set t) in
          Fmt.pr "%s: %s@." t (hex_bytes (to_string s)))
        [ "b"; "c"; "e"; "h" ];
      (* No allocation, I/O or recomputation on a repeated lookup. *)
      let again = Option.get (Map.Prepare.find set "h") in
      Fmt.pr "same storage on repeat: %b@."
        (again == Option.get (Map.Prepare.find set "h"));
      Fmt.pr "load rejects unknown: %s@."
        (match Map.Prepare.load set "zz" with Ok _ -> "?" | Error m -> m));
  [%expect
    {|
    targets: b c e h k w
    owned 36, sources 320 (file sizes 250 + 70)
    b: 01 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00 01 00 00 00 00 00 00 00
    c: 00 00 20 40
    e:
    h: 00 00 c0 3f 00 00 00 c0
    same storage on repeat: true
    load rejects unknown: capture "zz" is not in the prepared set |}]

let%expect_test
    "an uncast checkpoint tensor is a view of the source, a widened one a copy"
    =
  let srcs = sources () in
  match prepare ~srcs () with
  | Error e -> Map_test.show_fault e
  | Ok set ->
      let w = Option.get (Map.Prepare.find set "w") in
      let h = Option.get (Map.Prepare.find set "h") in
      let before_w = to_string w and before_h = to_string h in
      (* Overwrite the whole toy source: a view must follow it, a copy not. *)
      let toy = (List.hd srcs).bytes in
      Bigarray.Array1.fill toy '\xaa';
      Fmt.pr "view follows the source: %b@." (to_string w <> before_w);
      Fmt.pr "converted value is unchanged: %b@." (to_string h = before_h);
      [%expect
        {|
        view follows the source: true
        converted value is unchanged: true |}]

let%expect_test "source corruption" =
  let flip s i =
    String.mapi
      (fun j c -> if i = j then Char.chr (Char.code c lxor 1) else c)
      s
  in
  print_endline "-- a flipped payload byte";
  report
    (prepare
       ~srcs:(sources ~toy:(flip toy_bytes (String.length toy_bytes - 1)) ())
       ());
  print_endline "-- truncated";
  report
    (prepare
       ~srcs:
         (sources
            ~toy:(String.sub toy_bytes 0 (String.length toy_bytes - 1))
            ())
       ());
  print_endline "-- appended";
  report (prepare ~srcs:(sources ~toy:(toy_bytes ^ "x") ()) ());
  print_endline "-- missing / surplus / repeated";
  report (prepare ~srcs:[ List.hd (sources ()) ] ());
  report
    (prepare
       ~srcs:
         ({ Map.Prepare.name = "extra.safetensors"; bytes = bigstring_of "x" }
         :: sources ())
       ());
  report (prepare ~srcs:(sources () @ [ List.hd (sources ()) ]) ());
  print_endline "-- pinned bytes that are not a safetensors file";
  let junk = "not a safetensors file at all" in
  let map =
    replace ~sub:(toy_pin ())
      ~by:
        (file_pin ~name:"toy.safetensors" ~data:junk
           ~extra:(jstr {|,"repo_id":"o/toy","revision":"%s"|} rev)
           ())
      map_json
  in
  report (prepare ~map ~srcs:(sources ~toy:junk ()) ());
  [%expect
    {|
    -- a flipped payload byte
    source "toy.safetensors" hashes to a5d0fa457391693498e8cfa366f038973dad65f5d155aff29693cbe43e710926, the map pins ef5f96f5c32e0bbad0811851b5ccab377754223601ceb3c5213f38376cb69d16
    -- truncated
    source "toy.safetensors" is 249 bytes, the map pins 250
    -- appended
    source "toy.safetensors" is 251 bytes, the map pins 250
    -- missing / surplus / repeated
    source "pack.safetensors" was not supplied
    source "extra.safetensors" was supplied but the map declares no such file
    duplicate checkpoint file "toy.safetensors"
    -- pinned bytes that are not a safetensors file
    source "toy.safetensors" is not a valid safetensors file: resource_limit: header allocation limit exceeded |}]

let%expect_test "stored tensors must be the ones the map describes" =
  let map_with sub by = replace ~sub ~by map_json in
  print_endline "-- key";
  report (prepare ~map:(map_with {|"key":"model.w"|} {|"key":"model.nope"|}) ());
  print_endline
    "-- dtype: uncast key holding another dtype, cast whose source differs";
  report
    (prepare ~map:(map_with {|"key":"model.w"|} {|"key":"model.other"|}) ());
  report (prepare ~map:(map_with {|"from":"BF16"|} {|"from":"F16"|}) ());
  print_endline "-- shape";
  report
    (prepare
       ~map:
         (map_with {|"dtype":"F32","shape":[2,3]|}
            {|"dtype":"F32","shape":[3,2]|})
       ());
  print_endline "-- value digest";
  report (prepare ~map:(map_with sha_w (String.make 64 '0')) ());
  report (prepare ~map:(map_with sha_c (String.make 64 '0')) ());
  report (prepare ~map:(map_with sha_b (String.make 64 '0')) ());
  report (prepare ~map:(map_with sha_h (String.make 64 '0')) ());
  [%expect
    {|
    -- key
    capture "w": "toy.safetensors" has no tensor "model.nope"
    -- dtype: uncast key holding another dtype, cast whose source differs
    capture "w": the checkpoint stores I64, the map needs F32
    capture "h": the checkpoint stores BF16, the map needs F16
    -- shape
    capture "w": per the source header shape is [2; 3], map says [3; 2]
    -- value digest
    capture "w": per the prepared bytes digest is e2c0a71510b5394df7773b63fb5f54372b84c3564e67811bde7d665be227976d, map says 0000000000000000000000000000000000000000000000000000000000000000
    capture "c": per the prepared bytes digest is 072e3304b03423a4767d28c5fed09f81d5190ff60a3d078c6c1350eeb8bee28b, map says 0000000000000000000000000000000000000000000000000000000000000000
    capture "b": per the prepared bytes digest is 605390e5a369ee568b19ead1733af824c7c1d286d7d24b86283238fc44a99334, map says 0000000000000000000000000000000000000000000000000000000000000000
    capture "h": per the prepared bytes digest is 252b3318179cc24998f3670913d52d39085cf65b0dfa98fa523ffeab4b6683fe, map says 0000000000000000000000000000000000000000000000000000000000000000 |}]

let%expect_test "budgets are checked before the allocation they guard" =
  let l = Map.Limits.default in
  report (prepare ~limits:{ l with max_allocation_bytes = 8L } ());
  report (prepare ~limits:{ l with max_prepared_bytes = 100L } ());
  report
    (prepare
       ~limits:
         {
           l with
           max_prepared_bytes =
             Int64.of_int
               (String.length toy_bytes + String.length pack_bytes + 20);
         }
       ());
  report (prepare ~limits:{ l with max_source_bytes = 10L } ());
  [%expect
    {|
    buffer size is 24, over the limit 8
    prepared bytes is 320, over the limit 100
    prepared bytes is 344, over the limit 340
    source file size is 250, over the limit 10 |}]

(* --- widening, over every 16-bit pattern --- *)

let all_patterns =
  let b = Bytes.create (2 * 65536) in
  for u = 0 to 65535 do
    Bytes.set_uint16_le b (2 * u) u
  done;
  Bytes.to_string b

let widened from =
  let src = bigstring_of all_patterns in
  let dst =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout (4 * 65536)
  in
  Map.Widen.widen from ~src ~dst;
  dst

let%expect_test "widening matches Python's tables for all 65536 patterns" =
  (* Source table, then the binary32 tables hashlib computed from struct's
     'e' decoding (NaN by the documented quieting rule). *)
  print_endline (hex_of all_patterns);
  print_endline
    (Pt2_sha256.Digest.to_hex (Pt2_sha256.bigstring (widened Map.Dtype.F16)));
  print_endline
    (Pt2_sha256.Digest.to_hex (Pt2_sha256.bigstring (widened Map.Dtype.BF16)));
  [%expect
    {|
    68e419472d25e0b85e9917ccf692fd58245c5e95e9a46f07d1df81d2e9da246b
    b636c5716ff84d972782faf02d0194cb8951526bea4cc487082feb47b1860ddf
    9207d7eb28680a098c73dbe536d1ff7b94311dc417b9a385e0af6660683e93ca |}]

let%expect_test "widening edge values by name" =
  let one dtype u =
    let src =
      bigstring_of
        (let b = Bytes.create 2 in
         Bytes.set_uint16_le b 0 u;
         Bytes.to_string b)
    in
    let dst = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 4 in
    Map.Widen.widen dtype ~src ~dst;
    Bytes.get_int32_le (Bytes.of_string (to_string dst)) 0
  in
  let show name dtype u = Fmt.pr "%-18s %04x -> %08lx@." name u (one dtype u) in
  show "f16 +0" Map.Dtype.F16 0x0000;
  show "f16 -0" Map.Dtype.F16 0x8000;
  show "f16 min subnormal" Map.Dtype.F16 0x0001;
  show "f16 max subnormal" Map.Dtype.F16 0x03ff;
  show "f16 min normal" Map.Dtype.F16 0x0400;
  show "f16 1.0" Map.Dtype.F16 0x3c00;
  show "f16 max" Map.Dtype.F16 0x7bff;
  show "f16 +inf" Map.Dtype.F16 0x7c00;
  show "f16 -inf" Map.Dtype.F16 0xfc00;
  show "f16 signaling NaN" Map.Dtype.F16 0x7c01;
  show "f16 -quiet NaN" Map.Dtype.F16 0xfe00;
  show "bf16 -0" Map.Dtype.BF16 0x8000;
  show "bf16 signaling NaN" Map.Dtype.BF16 0x7f81;
  show "bf16 -inf" Map.Dtype.BF16 0xff80;
  show "bf16 subnormal" Map.Dtype.BF16 0x0001;
  (try
     Map.Widen.widen Map.Dtype.F32 ~src:Pt2_storage.empty ~dst:Pt2_storage.empty
   with Invalid_argument m -> print_endline m);
  [%expect
    {|
    f16 +0             0000 -> 00000000
    f16 -0             8000 -> 80000000
    f16 min subnormal  0001 -> 33800000
    f16 max subnormal  03ff -> 387fc000
    f16 min normal     0400 -> 38800000
    f16 1.0            3c00 -> 3f800000
    f16 max            7bff -> 477fe000
    f16 +inf           7c00 -> 7f800000
    f16 -inf           fc00 -> ff800000
    f16 signaling NaN  7c01 -> 7fc02000
    f16 -quiet NaN     fe00 -> ffc00000
    bf16 -0            8000 -> 80000000
    bf16 signaling NaN 7f81 -> 7f810000
    bf16 -inf          ff80 -> ff800000
    bf16 subnormal     0001 -> 00010000
    Widen.widen: only BF16 and F16 widen to F32 |}]

(* Two checkpoint files, and two captures that name the same stored tensor
   (tied weights). *)
let%expect_test "several sources and tied aliases" =
  let shard = safetensors [ ("s", "F32", [ 2 ], k_bytes) ] in
  let entry target sha file key =
    jstr
      {|%S:{"dtype":"F32","shape":[%s],"sha256":%S,"origin":{"kind":"checkpoint","file":%S,"key":%S,"convert":{"op":"none"},"tied_aliases":[]}}|}
      target
      (if file = "shard.safetensors" then "2" else "2,3")
      sha file key
  in
  let map =
    jstr
      {|{"schema_version":2,"artifact_id":%S,"graph_sha256":%S,"model_id":"toy","sources":{"checkpoint":{"files":[%s,%s]}},"tensors":{%s},"unmapped":[]}|}
      artifact_id (hex_of program_json) (toy_pin ())
      (file_pin ~name:"shard.safetensors" ~data:shard
         ~extra:(jstr {|,"repo_id":"o/toy","revision":"%s"|} rev)
         ())
      (String.concat ","
         [
           entry "w" sha_w "toy.safetensors" "model.w";
           entry "w_tied" sha_w "toy.safetensors" "model.w";
           entry "s" sha_k "shard.safetensors" "s";
         ])
  in
  let srcs =
    [
      { Map.Prepare.name = "toy.safetensors"; bytes = bigstring_of toy_bytes };
      { Map.Prepare.name = "shard.safetensors"; bytes = bigstring_of shard };
    ]
  in
  (match prepare ~map ~srcs () with
  | Error e -> Map_test.show_fault e
  | Ok set ->
      Fmt.pr "owned %Ld, sources %Ld@."
        (Map.Prepare.owned_bytes set)
        (Map.Prepare.source_bytes set);
      Fmt.pr "aliases agree: %b@."
        (to_string (Option.get (Map.Prepare.find set "w"))
        = to_string (Option.get (Map.Prepare.find set "w_tied"))));
  [%expect {|
    owned 0, sources 320
    aliases agree: true |}]
