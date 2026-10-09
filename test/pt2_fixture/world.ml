(* The whole release, as bytes. Every pin is computed here from the bytes it
   pins, so a test corrupts exactly one thing and the chain must notice. *)

module Fx = Pt2_checkpoint_map_test.Fixtures

let jstr = Printf.sprintf
let hex = Fx.hex_of
let artifact_id = Fx.artifact_id
let url name = "https://example.org/" ^ name

(* --- tar and gzip, written independently of the reader under test --- *)

let octal width n =
  let s = Printf.sprintf "%0*o" (width - 1) n in
  s ^ "\000"

let tar_header ?(typeflag = '0') ?(linkname = "") ?(prefix = "") ~name ~size ()
    =
  let b = Bytes.make 512 '\000' in
  let put off s = Bytes.blit_string s 0 b off (String.length s) in
  put 0 name;
  put 100 (octal 8 0o644);
  put 108 (octal 8 0);
  put 116 (octal 8 0);
  put 124 (octal 12 size);
  put 136 (octal 12 0);
  put 148 "        ";
  Bytes.set b 156 typeflag;
  put 157 linkname;
  put 257 "ustar\00000";
  put 345 prefix;
  let sum = ref 0 in
  Bytes.iter (fun c -> sum := !sum + Char.code c) b;
  put 148 (Printf.sprintf "%06o\000 " !sum);
  Bytes.to_string b

let pad data =
  let n = String.length data in
  let r = n mod 512 in
  if r = 0 then data else data ^ String.make (512 - r) '\000'

let tar_member ?typeflag ?linkname ?prefix name data =
  tar_header ?typeflag ?linkname ?prefix ~name ~size:(String.length data) ()
  ^ pad data

let tar members =
  String.concat "" (List.map (fun (n, d) -> tar_member n d) members)
  ^ String.make 1024 '\000'

let gzip data =
  let body =
    match Zipc_deflate.deflate ~level:`Default data with
    | Ok s -> s
    | Error m -> failwith m
  in
  let le32 v =
    let b = Bytes.create 4 in
    Bytes.set_int32_le b 0 v;
    Bytes.to_string b
  in
  "\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff" ^ body
  ^ le32 (Zipc_deflate.Crc_32.string data)
  ^ le32 (Int32.of_int (String.length data))

(* --- the release --- *)

let contract_json = jstr {|{"artifact_id":%S,"schema_version":1}|} artifact_id

let members =
  [
    ("captures.json", Fx.captures_json Fx.program_json);
    ("cases.json", {|{"cases":[]}|});
    ("cases/case-00/inputs.pt", "inputs");
    ("cases/case-00/outputs.pt", "outputs");
    ("contract.json", contract_json);
    ("data/constants/model_constants_config.json", Fx.constants_json);
    ("data/weights/model_weights_config.json", Fx.weights_json ());
    ("models/model.json", Fx.program_json);
    ("models/safetensors.v2.json", Fx.map_json);
  ]

let archive_bytes = gzip (tar members)

let pin_json ~name ~data =
  jstr {|{"name":%S,"sha256":%S,"size":%d,"url":%S}|} name (hex data)
    (String.length data) (url name)

let member_json (name, data) =
  jstr {|%S:{"sha256":%S,"size":%d}|} name (hex data) (String.length data)

let manifest_json ?(payload = "null") ?(members = members)
    ?(cases = {|["case-00"]|}) ?(graph = hex Fx.program_json) () =
  jstr
    {|{"archive":{"name":"archive.tar.gz","sha256":%S,"size":%d},"artifact_id":%S,"cases":%s,"contract_sha256":%S,"graph_sha256":%S,"map_v2":{"assets":[],"member":"models/safetensors.v2.json"},"members":{%s},"payload":%s,"producer_commit":"p","schema_version":1}|}
    (hex archive_bytes)
    (String.length archive_bytes)
    artifact_id cases (hex contract_json) graph
    (String.concat "," (List.map member_json members))
    payload

let archive_pin = pin_json ~name:"archive.tar.gz" ~data:archive_bytes
let manifest_bytes = manifest_json ()
let manifest_pin = pin_json ~name:"manifest.json" ~data:manifest_bytes

let publication_json ?(archive = archive_pin) ?(manifest = manifest_pin)
    ?(release_tag = "tag-1") ?(graph = hex Fx.program_json) () =
  jstr
    {|{"artifacts":[{"artifact_id":%S,"assets":{"archive":%s,"manifest":%s},"graph_sha256":%S}],"release_tag":%S,"repository":"o/r","schema_version":1}|}
    artifact_id archive manifest graph release_tag

let publication_bytes = publication_json ()

let cohort_json ?(publication = publication_bytes) ?(archive = archive_pin)
    ?(manifest = manifest_pin) ?(graph = hex Fx.program_json)
    ?(contract = hex contract_json) ?(map = hex Fx.map_json)
    ?(sources = [ Fx.toy_bytes; Fx.pack_bytes ]) ?(cases = {|["case-00"]|}) () =
  let source name data = pin_json ~name ~data in
  ignore sources;
  jstr
    {|{"artifacts":[{"archive":%s,"artifact_id":%S,"cases":%s,"contract_sha256":%S,"graph_sha256":%S,"manifest":%s,"map_member":"models/safetensors.v2.json","map_sha256":%S,"map_sources":{"checkpoint":{"files":[%s]}},"role":"test"}],"publication":{"sha256":%S,"size":%d,"url":%S},"release_producer_commit":"p","release_tag":"tag-1","repository":"o/r","schema_version":1}|}
    archive artifact_id cases contract graph manifest map
    (source "toy.safetensors" Fx.toy_bytes)
    (hex publication)
    (String.length publication)
    (url "publication.json")

let cohort_bytes = cohort_json ()

(* Every file the release serves, by URL. *)
let served =
  [
    (url "publication.json", publication_bytes);
    (url "manifest.json", manifest_bytes);
    (url "archive.tar.gz", archive_bytes);
    (url "toy.safetensors", Fx.toy_bytes);
    (url "pack.safetensors", Fx.pack_bytes);
  ]

let decode_cohort text =
  match Err.payload (Pt2_fixture.Cohort.of_string text) with
  | Ok c -> c
  | Error e ->
      Fmt.epr "%a@." Pt2_fixture.Fault.pp_error e;
      failwith "cohort fixture"

let cohort = decode_cohort cohort_bytes
let entry = List.hd cohort.entries
