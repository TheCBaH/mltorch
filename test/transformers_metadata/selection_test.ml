open Transformers_metadata.Json_util
open World
module J = Jsont.Json

let%expect_test
    "regeneration uses metadata only, is deterministic and reopens offline" =
  with_dir (fun dir ->
      let calls = ref [] in
      let online = config dir calls in
      let first, _ = unwrap (T.Selection.generate online selection producer) in
      Fmt.pr "metadata downloads: %d@." (List.length !calls);
      let offline = U.Bundle.config online.cache in
      let second, _ =
        unwrap (T.Selection.generate offline selection producer)
      in
      Fmt.pr "same output: %b@." (bytes first = bytes second);
      let decoded = unwrap (Pt2_fixture.Cohort.of_string (bytes first)) in
      Fmt.pr "entries %d, sources %d@."
        (List.length decoded.entries)
        (List.length (List.hd decoded.entries).sources);
      let manifest = (List.hd decoded.entries).archive in
      let graph_file =
        Filename.concat
          (U.Cache.bundle_path online.cache manifest.sha256)
          "models/model.json"
      in
      write graph_file "corrupt";
      show (T.Selection.generate offline selection producer));
  [%expect
    {|
metadata downloads: 3
same output: true
entries 1, sources 1
archive member "models/model.json": size is 7 bytes, expected 1004|}]

let%expect_test "wrong publication bytes and inconsistent producer pins fail" =
  with_dir (fun dir ->
      show
        (T.Selection.generate
           (config
              ~publication:(replace "repository" (J.string "wrong") publication)
              dir (ref []))
           selection producer));
  let mixed =
    replace "artifacts"
      (J.list
         [
           replace "producer_commit"
             (J.string (String.make 40 'd'))
             publication_row;
         ])
      publication
  in
  with_dir (fun dir ->
      let selection = unwrap (T.Selection.of_json (selection_for mixed)) in
      show
        (T.Selection.generate
           (config ~publication:mixed dir (ref []))
           selection producer));
  [%expect
    {|
    the publication index: size is 1117 bytes, expected 1115
    toy/task/reference/forward/fp32/dynamo/static/ckpt-aaaaaaaaaaaa: producer_commit is dddddddddddddddddddddddddddddddddddddddd, expected bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb |}]

let%expect_test
    "schema, duplicate IDs, unsupported source component and missing map member"
    =
  show
    (T.Selection.of_json
       (replace "schema_version" (J.int 2) (selection_for publication)));
  let rows = unwrap (member "artifacts" (selection_for publication)) in
  let row = List.hd (unwrap (array rows)) in
  show
    (T.Selection.of_json
       (replace "artifacts" (J.list [ row; row ]) (selection_for publication)));
  show
    (T.Producer.source_artifact producer
       "toy/task/reference/decode/fp32/dynamo/static-h4/ckpt-aaaaaaaaaaaa");
  let pub = unwrap (Pt2_fixture.Publication.of_string (bytes publication)) in
  let manifest = unwrap (Pt2_fixture.Manifest.of_string (bytes manifest)) in
  show
    (T.Selection.bootstrap selection "test" (List.hd pub.entries)
       { manifest with map_member = "models/missing.json" });
  [%expect
    {|
    invalid metadata: unsupported schema_version
    duplicate metadata identity: toy/task/reference/forward/fp32/dynamo/static/ckpt-aaaaaaaaaaaa
    missing metadata: source component recipe toy/task/tiny/decode/fp32/dynamo/static-h4
    archive lacks member "models/missing.json" |}]

let%expect_test
    "source checkpoint and contract pin mismatches never generate a cohort" =
  let source =
    replace "reference"
      (replace "revision" (J.string (String.make 40 'e')) reference)
      candidate
  in
  with_dir (fun dir ->
      show
        (T.Selection.generate
           (config dir (ref []))
           selection
           { producer with candidates = [ source ] }));
  let changed_manifest =
    replace "producer_commit" (J.string (String.make 40 'e')) manifest
  in
  let pin =
    json (W.pin_json ~name:"manifest.json" ~data:(bytes changed_manifest))
  in
  let changed_pub =
    replace "artifacts"
      (J.list
         [
           replace "assets"
             (obj [ ("archive", archive_pin); ("manifest", pin) ])
             publication_row;
         ])
      publication
  in
  let changed_selection =
    unwrap (T.Selection.of_json (selection_for changed_pub))
  in
  with_dir (fun dir ->
      show
        (T.Selection.generate
           (config ~publication:changed_pub ~manifest:changed_manifest dir
              (ref []))
           changed_selection producer));
  [%expect
    {|
    toy/task/reference/forward/fp32/dynamo/static/ckpt-aaaaaaaaaaaa: source reference revision is aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, expected eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
    toy/task/reference/forward/fp32/dynamo/static/ckpt-aaaaaaaaaaaa: manifest producer_commit is eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee, expected bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb |}]
