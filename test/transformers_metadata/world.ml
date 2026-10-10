open Transformers_metadata.Json_util
module T = Transformers_metadata
module W = Pt2_fixture_test.World
module Fx = Pt2_checkpoint_map_test.Fixtures
module U = Pt2_fixture_unix
module J = Jsont.Json

let unwrap result = Err.or_raise ~pp_error:T.Fault.pp_error result
let json bytes = unwrap (parse bytes)
let bytes j = unwrap (text j)
let producer_commit = String.make 40 'b'
let revision = String.make 40 'a'

let reference =
  obj
    [
      ("repo", J.string "o/r");
      ("revision", J.string revision);
      ("config_sha256", J.string (W.hex "{}"));
    ]

let weights = obj (("kind", J.string "checkpoint") :: unwrap (members reference))

let candidate =
  obj
    [
      ("id", J.string "toy");
      ("category", J.string "task");
      ("reference", reference);
    ]

let producer : T.Producer.t =
  {
    candidates = [ candidate ];
    catalogue = [];
    commit = String.make 40 'c';
    metadata = obj [];
    root = "/synthetic";
    task_models = [];
  }

let contract =
  obj
    [
      ("artifact_id", J.string W.artifact_id);
      ("model_id", J.string "toy");
      ("schema_version", J.int 1);
      ("weights", weights);
    ]

let member_bytes =
  List.map
    (fun (name, data) ->
      if name = "contract.json" then (name, bytes contract) else (name, data))
    W.members

let archive = W.gzip (W.tar member_bytes)
let archive_pin = json (W.pin_json ~name:"archive.tar.gz" ~data:archive)

let replace key value j =
  obj ((key, value) :: List.remove_assoc key (unwrap (members j)))

let manifest =
  json (W.manifest_json ~members:member_bytes ())
  |> replace "contract_sha256" (J.string (W.hex (bytes contract)))
  |> replace "producer_commit" (J.string producer_commit)
  |> replace "archive"
       (obj
          [
            ("name", J.string "archive.tar.gz");
            ("sha256", J.string (W.hex archive));
            ("size", J.int (String.length archive));
          ])

let manifest_pin =
  json (W.pin_json ~name:"manifest.json" ~data:(bytes manifest))

let publication_row =
  obj
    [
      ("artifact_id", J.string W.artifact_id);
      ("assets", obj [ ("archive", archive_pin); ("manifest", manifest_pin) ]);
      ("graph_sha256", J.string (W.hex Fx.program_json));
      ("model_id", J.string "toy");
      ("producer_commit", J.string producer_commit);
      ("weight_source", weights);
    ]

let publication =
  obj
    [
      ("artifacts", J.list [ publication_row ]);
      ("release_tag", J.string "tag-1");
      ("repository", J.string "o/r");
      ("schema_version", J.int 1);
    ]

let selection_for publication =
  obj
    [
      ("schema_version", J.int 1);
      ( "publication",
        json (W.pin_json ~name:"publication.json" ~data:(bytes publication)) );
      ("repository", J.string "o/r");
      ("release_tag", J.string "tag-1");
      ("release_producer_commit", J.string producer_commit);
      ( "artifacts",
        J.list
          [
            obj
              [
                ("artifact_id", J.string W.artifact_id);
                ("role", J.string "test");
              ];
          ] );
    ]

let selection = unwrap (T.Selection.of_json (selection_for publication))
let with_dir = Pt2_fixture_unix_test.Support.with_dir
let write = Pt2_fixture_unix_test.Support.write

let config ?(publication = publication) ?(manifest = manifest) dir calls =
  let cache = unwrap (U.Cache.create dir) in
  let transport : U.Transport.t =
   fun ~url ~dest ->
    calls := url :: !calls;
    match
      List.assoc_opt url
        [
          (W.url "publication.json", bytes publication);
          (W.url "manifest.json", bytes manifest);
          (W.url "archive.tar.gz", archive);
        ]
    with
    | None -> Error "unserved source: generation must not fetch weights"
    | Some data ->
        write dest data;
        Ok ()
  in
  U.Bundle.config ~transport cache

let show result =
  match Err.payload result with
  | Ok _ -> print_endline "ok"
  | Error e -> Fmt.pr "%a@." T.Fault.pp_error e
