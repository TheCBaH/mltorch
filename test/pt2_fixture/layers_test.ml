module F = Pt2_fixture
open World

let unstyle = Pt2_checkpoint_map_test.Map_test.unstyle

let show = function
  | Ok _ -> print_endline "ok"
  | Error e ->
      let text = unstyle (Fmt.str "%a" F.Fault.pp_error e) in
      print_endline
        (match String.index_opt text '\n' with
        | Some i -> String.sub text 0 i
        | None -> text)

let replace = Pt2_checkpoint_map_test.Fixtures.replace

(* A different, still well-formed digest. *)
let bump d = (if d.[0] = '0' then "1" else "0") ^ String.sub d 1 63

let check_publication ?(cohort = cohort) text =
  Err.payload
    (let open Err.Syntax in
     let* p = F.Publication.of_string text in
     F.Publication.check cohort p entry)

let check_manifest ?(entry = entry) text =
  Err.payload
    (let open Err.Syntax in
     let* m = F.Manifest.of_string text in
     F.Manifest.check entry m)

let%expect_test "the untouched chain is accepted" =
  show (check_publication publication_bytes);
  show (check_manifest manifest_bytes);
  [%expect {|
    ok
    ok |}]

let%expect_test "cohort pins are enforced on the publication index" =
  let t sub by =
    show (check_publication (replace ~sub ~by publication_bytes))
  in
  t {|"release_tag":"tag-1"|} {|"release_tag":"tag-2"|};
  t {|"repository":"o/r"|} {|"repository":"o/other"|};
  t artifact_id "other/artifact";
  (* the archive pin: name, digest, size, url *)
  t {|"name":"archive.tar.gz"|} {|"name":"archive2.tar.gz"|};
  t (hex archive_bytes) (String.make 64 '0');
  t
    (Printf.sprintf {|"size":%d,"url":"%s"|}
       (String.length archive_bytes)
       (url "archive.tar.gz"))
    (Printf.sprintf {|"size":%d,"url":"%s"|}
       (String.length archive_bytes + 1)
       (url "archive.tar.gz"));
  t (url "archive.tar.gz") "https://elsewhere.example/archive.tar.gz";
  (* the manifest pin and the graph digest *)
  t (hex manifest_bytes) (String.make 64 '1');
  t
    (hex Pt2_checkpoint_map_test.Fixtures.program_json)
    (bump (hex Pt2_checkpoint_map_test.Fixtures.program_json));
  [%expect
    {|
    the publication index: release tag is "tag-2", expected "tag-1"
    the publication index: repository is "o/other", expected "o/r"
    the publication index has no entry for artifact "toy/task/reference/forward/fp32/dynamo/static/ckpt-aaaaaaaaaaaa"
    the archive: file name is "archive2.tar.gz", expected "archive.tar.gz"
    the archive: digest is 0000000000000000000000000000000000000000000000000000000000000000, expected 1b2884d6a47093d45e0bc504d913eb01687afa516f7efb89e1a5df391a55f26f
    the archive: size is 1941 bytes, expected 1940
    the archive: URL is "https://elsewhere.example/archive.tar.gz", expected "https://example.org/archive.tar.gz"
    the manifest: digest is 1111111111111111111111111111111111111111111111111111111111111111, expected 854e82e137d0516da5ef0ed10ed96c386197da85417aa8e304018f15cd2ab181
    the graph: digest is 0366c272d3d3a041a6504a897864deba52d40ac66fcd8842ca1b51b1facacea8, expected 2366c272d3d3a041a6504a897864deba52d40ac66fcd8842ca1b51b1facacea8 |}]

let%expect_test "the manifest is the cohort's artifact" =
  let t ?(entry = entry) sub by =
    show (check_manifest ~entry (replace ~sub ~by manifest_bytes))
  in
  t {|"payload":null|} {|"payload":{"x":1}|};
  t artifact_id "other/artifact";
  t
    (hex Pt2_checkpoint_map_test.Fixtures.program_json)
    (bump (hex Pt2_checkpoint_map_test.Fixtures.program_json));
  t (hex contract_json) (bump (hex contract_json));
  t {|"cases":["case-00"]|} {|"cases":["case-00","case-01"]|};
  t {|"name":"archive.tar.gz"|} {|"name":"x.tar.gz"|};
  t (hex archive_bytes) (bump (hex archive_bytes));
  t {|"member":"models/safetensors.v2.json"|} {|"member":"models/other.json"|};
  (* a listed member that the replay needs is dropped *)
  show
    (check_manifest
       (manifest_json ~members:(List.remove_assoc "contract.json" members) ()));
  show
    (check_manifest
       (manifest_json
          ~members:(List.remove_assoc "cases/case-00/outputs.pt" members)
          ()));
  (* a member whose bytes are not the ones the cohort pins *)
  show
    (check_manifest
       (manifest_json
          ~members:
            (("models/model.json", "{}")
            :: List.remove_assoc "models/model.json" members)
          ()));
  [%expect
    {|
    the manifest: payload is "present", expected "null (a slim bundle)"
    the manifest: artifact id is "other/artifact", expected "toy/task/reference/forward/fp32/dynamo/static/ckpt-aaaaaaaaaaaa"
    the graph: digest is 0366c272d3d3a041a6504a897864deba52d40ac66fcd8842ca1b51b1facacea8, expected 2366c272d3d3a041a6504a897864deba52d40ac66fcd8842ca1b51b1facacea8
    contract.json: digest is 0340bfc94de5e845755366acd4a0ed397fe123709f14edc036498c3d40237449, expected 5340bfc94de5e845755366acd4a0ed397fe123709f14edc036498c3d40237449
    the manifest: case list is "case-00,case-01", expected "case-00"
    the archive: file name is "x.tar.gz", expected "archive.tar.gz"
    the archive: digest is 0b2884d6a47093d45e0bc504d913eb01687afa516f7efb89e1a5df391a55f26f, expected 1b2884d6a47093d45e0bc504d913eb01687afa516f7efb89e1a5df391a55f26f
    the manifest: map member is "models/other.json", expected "models/safetensors.v2.json"
    archive lacks member "contract.json"
    archive lacks member "cases/case-00/outputs.pt"
    archive member "models/model.json": digest is 44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a, expected 2366c272d3d3a041a6504a897864deba52d40ac66fcd8842ca1b51b1facacea8 |}]

let%expect_test "cohort decoding" =
  let t sub by =
    match Err.payload (F.Cohort.of_string (replace ~sub ~by cohort_bytes)) with
    | Ok _ -> print_endline "ok"
    | Error e -> show (Error e)
  in
  t {|"release_tag":"tag-1"|} {|"release_tag":1|};
  t (hex Pt2_checkpoint_map_test.Fixtures.map_json) "abc";
  [%expect
    {|
    failed to decode the cohort manifest: Expected string but found number
    "abc" is not a 64-digit lower-case sha256 |}]

(* The map member's digest is pinned by the cohort separately from the
   manifest's own listing of it. *)
let%expect_test "the cohort pins the map's digest" =
  let bumped =
    decode_cohort
      (cohort_json
         ~map:(bump (hex Pt2_checkpoint_map_test.Fixtures.map_json))
         ())
  in
  show (check_manifest ~entry:(List.hd bumped.entries) manifest_bytes);
  [%expect
    {| the checkpoint map: digest is 7c2654dc80e72258a54516b984c066bda4bc1a4fc7769c9424a3fbbf9f249f85, expected 0c2654dc80e72258a54516b984c066bda4bc1a4fc7769c9424a3fbbf9f249f85 |}]

(* --- contract.json and cases.json --- *)

module Fx = Pt2_checkpoint_map_test.Fixtures

let contract_json
    ?(call = {|{"args":[],"kwargs":["x"],"outputs":"tensor_tuple"}|})
    ?(mutations = "[]") ?(dynamic = "{}")
    ?(inputs = {|[{"dtype":"float32","name":"x","shape":[2]}]|}) () =
  Printf.sprintf
    {|{"artifact_id":%S,"call":%s,"dynamic_constraints":%s,"graph_sha256":%S,"inputs":%s,"mutations":%s,"outputs":[{"dtype":"float32","name":"y","shape":[2]}],"tolerances":{"atol":1e-05,"rtol":0.0001},"verified_cases":2,"schema_version":1}|}
    artifact_id call dynamic (hex Fx.program_json) inputs mutations

let case id inputs outputs =
  Printf.sprintf
    {|{"id":%S,"inputs":%s,"inputs_sha256":%S,"outputs":%s,"outputs_sha256":%S}|}
    id inputs (String.make 64 'a') outputs (String.make 64 'b')

let cases_json ?(artifact = artifact_id) ?(atol = "1e-05")
    ?(cases =
      [ case "case-00" {|["x"]|} {|["y"]|}; case "case-01" {|["x"]|} {|["y"]|} ])
    () =
  Printf.sprintf
    {|{"artifact_id":%S,"cases":[%s],"schema_version":1,"tolerances":{"atol":%s,"rtol":0.0001}}|}
    artifact (String.concat "," cases) atol

let check_contract text =
  match Err.payload (F.Contract.of_string text) with
  | Ok c ->
      Printf.printf "ok (%d inputs, %d outputs, dynamic %b)\n"
        (List.length c.inputs) (List.length c.outputs) c.dynamic
  | Error e -> show (Error e)

let%expect_test "only a plain functional tensor call is a contract" =
  check_contract (contract_json ());
  check_contract
    (contract_json ~call:{|{"args":["x"],"kwargs":[],"outputs":"tensor_tuple"}|}
       ());
  check_contract
    (contract_json ~call:{|{"args":[],"kwargs":["x"],"outputs":"tensor"}|} ());
  check_contract
    (contract_json ~call:{|{"args":[],"kwargs":["z"],"outputs":"tensor_tuple"}|}
       ());
  check_contract (contract_json ~mutations:{|[{"buffer":"b"}]|} ());
  check_contract (contract_json ~dynamic:{|{"batch":[1,8]}|} ());
  check_contract
    (contract_json
       ~call:{|{"args":[],"kwargs":["x","x"],"outputs":"tensor_tuple"}|}
       ~inputs:
         {|[{"dtype":"float32","name":"x","shape":[2]},{"dtype":"float32","name":"x","shape":[2]}]|}
       ());
  check_contract
    (contract_json ~inputs:{|[{"dtype":"float33","name":"x","shape":[2]}]|} ());
  [%expect
    {|
    ok (1 inputs, 1 outputs, dynamic false)
    the contract is not a plain tensor call: positional arguments
    the contract is not a plain tensor call: result is tensor
    the contract is not a plain tensor call: keyword order differs from the input list
    the contract declares mutations: mutation
    ok (1 inputs, 1 outputs, dynamic true)
    the contract is not a plain tensor call: duplicate input name "x"
    unknown dtype code "float33" |}]

let check_cases ?(contract = contract_json ()) text =
  let contract =
    match Err.payload (F.Contract.of_string contract) with
    | Ok c -> c
    | Error _ -> failwith "contract"
  in
  show
    (Err.payload
       (let open Err.Syntax in
        let* cases = F.Cases.of_string text in
        F.Cases.check contract cases))

let%expect_test "cases.json is held to the contract" =
  check_cases (cases_json ());
  check_cases (cases_json ~artifact:"other" ());
  check_cases (cases_json ~atol:"1e-06" ());
  check_cases (cases_json ~cases:[ case "case-00" {|["x"]|} {|["y"]|} ] ());
  check_cases (cases_json ~cases:[] ());
  check_cases
    (cases_json
       ~cases:
         [
           case "case-00" {|["x"]|} {|["y"]|};
           case "case-02" {|["x"]|} {|["y"]|};
         ]
       ());
  check_cases
    (cases_json
       ~cases:
         [
           case "case-00" {|["w"]|} {|["y"]|};
           case "case-01" {|["x"]|} {|["y"]|};
         ]
       ());
  check_cases
    (cases_json
       ~cases:
         [
           case "case-00" {|["x"]|} {|["y"]|};
           case "case-01" {|["x"]|} {|["z"]|};
         ]
       ());
  [%expect
    {|
    ok
    cases.json: artifact id is "other", expected "toy/task/reference/forward/fp32/dynamo/static/ckpt-aaaaaaaaaaaa"
    cases.json: tolerances is "atol=1e-06 rtol=0.0001", expected "atol=1e-05 rtol=0.0001"
    cases.json: case list is "1", expected "2"
    cases.json: case list is "0", expected "2"
    cases.json: case ids is "case-02", expected "case-01"
    case case-00 inputs: tensors are [w], expected [x]
    case case-01 outputs: tensors are [z], expected [y] |}]
