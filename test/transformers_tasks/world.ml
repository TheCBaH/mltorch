open Transformers_metadata.Json_util
module T = Transformers_tasks
module U = Pt2_fixture_unix
module F = Pt2_fixture
module R = Pt2_fixture_replay_test.Release
module W = Pt2_fixture_test.World
module J = Jsont.Json

let get result = Err.or_raise ~pp_error:T.Fault.pp result
let bytes json = get (text json)
let json bytes = get (parse bytes)

let set key value json =
  obj ((key, value) :: List.remove_assoc key (get (members json)))

let member_pin data =
  obj
    [ ("sha256", J.string (W.hex data)); ("size", J.int (String.length data)) ]

let pin_json name data = json (W.pin_json ~name ~data)

let producer =
  obj
    [
      ("architecture", J.string "aarch64");
      ("core_override", J.bool false);
      ("core_revision", J.string (String.make 40 'c'));
      ("core_source_sha256", J.string (String.make 64 'd'));
      ("device", J.string "cpu");
      ("lock_sha256", J.string (String.make 64 'e'));
      ("python", J.string "fixture");
      ("tools_sha256", J.string (String.make 64 'f'));
      ( "versions",
        obj
          [
            ("torch", J.string "fixture"); ("transformers", J.string "fixture");
          ] );
    ]

let generator =
  obj
    (get (members producer)
    @ [
        ("commit", J.string (String.make 40 'a'));
        ("consumer_execution", J.string "not_measured");
        ("routes", J.list [ J.string "eager"; J.string "reexport" ]);
        ( "cpu",
          obj
            ([ ("threads", J.int 1); ("interop_threads", J.int 1) ]
            @ List.map
                (fun key -> (key, J.string "fixture"))
                [
                  "machine";
                  "platform";
                  "cpuinfo";
                  "capability";
                  "parallel";
                  "torch_config";
                ]) );
      ])

let config_bytes = "{}"

let contract =
  json (R.contract_json (R.program ()))
  |> set "producer" producer
  |> set "weights" (obj [ ("config_sha256", J.string (W.hex config_bytes)) ])

let rel = R.build ~contract:(bytes contract) ()
let pub = json (List.assoc (W.url "publication.json") rel.served)
let pub_rows = get (member "artifacts" pub >>= array)

let pub =
  set "artifacts"
    (J.list (List.map (set "producer_commit" (J.string "p")) pub_rows))
    pub

let pub_pin = get (pin_of (pin_json "publication.json" (bytes pub)))
let cohort = { rel.cohort with F.Cohort.publication = pub_pin }
let entry = List.hd cohort.entries
let component_manifest = json (List.assoc (W.url "manifest.json") rel.served)
let component_inventory = get (member "members" component_manifest)

let reference =
  obj
    ([
       ("artifact_id", J.string entry.artifact_id);
       ("producer_commit", J.string "p");
       ( "assets",
         obj
           [ ("archive", pin entry.archive); ("manifest", pin entry.manifest) ]
       );
       ("graph", get (member "models/model.json" component_inventory));
       ("contract", get (member "contract.json" component_inventory));
       ("map", get (member entry.map_member component_inventory));
       ("config", member_pin config_bytes);
     ]
    @ List.map
        (fun (key, original) ->
          let v =
            match Err.payload (member original contract) with
            | Ok v -> v
            | Error _ -> J.null ()
          in
          (key, v))
        [
          ("inputs", "inputs");
          ("outputs", "outputs");
          ("tolerances", "tolerances");
          ("weight_source", "weights");
          ("release_environment", "producer");
          ("exporter", "exporter");
          ("variant", "variant");
          ("dynamic_constraints", "dynamic_constraints");
          ("state", "state");
        ])

let descriptors (tensors : R.tensor list) =
  J.list
    (List.map
       (fun tensor ->
         obj
           [
             ("name", J.string tensor.R.name);
             ( "dtype",
               J.string
                 (match tensor.kind with
                 | R.Float -> "float32"
                 | R.Long -> "int64"
                 | R.Bool -> "bool") );
             ("shape", J.list [ J.int 2; J.int 3 ]);
             ("sha256", J.string (W.hex tensor.raw));
           ])
       tensors)

let comparison =
  obj
    [
      ("status", J.string "pass");
      ( "outputs",
        J.list
          (List.map
             (fun name ->
               obj
                 [
                   ("name", J.string name);
                   ("elements", J.int 6);
                   ("over_tolerance", J.int 0);
                   ("max_absolute_error", J.number 0.);
                   ("bitwise", J.bool true);
                 ])
             [ "y"; "z" ]) );
    ]

let raw i =
  let name = Printf.sprintf "case-%02d" i in
  obj
    [
      ("artifact_id", J.string entry.artifact_id);
      ("case_id", J.string name);
      ( "published_members",
        obj
          (List.map
             (fun role ->
               ( role ^ ".pt",
                 get
                   (member
                      ("cases/" ^ name ^ "/" ^ role ^ ".pt")
                      component_inventory) ))
             [ "inputs"; "outputs" ]) );
    ]

let case_rows =
  List.mapi
    (fun i (case : R.case) ->
      let id = Printf.sprintf "reference-00-case-%02d" i in
      obj
        [
          ("id", J.string id);
          ("artifact_id", J.string entry.artifact_id);
          ( "files",
            obj
              (List.map
                 (fun (role, tensors) ->
                   ( "cases/" ^ id ^ "/" ^ role ^ ".pt",
                     obj [ ("tensors", descriptors tensors) ] ))
                 [
                   ("inputs", case.inputs);
                   ("published", case.outputs);
                   ("eager", case.outputs);
                   ("exported", case.outputs);
                 ]) );
          ( "raw",
            J.list
              [
                set "path"
                  (J.string ("raw/" ^ id ^ ".json"))
                  (member_pin (bytes (raw i)));
              ] );
          ( "comparisons",
            obj
              (List.map
                 (fun key -> (key, comparison))
                 [
                   "eager_vs_published";
                   "reexport_vs_eager";
                   "reexport_vs_published";
                 ]) );
        ])
    R.base_cases

let recipe_sha = String.make 64 '1'
let request_sha = String.make 64 '2'

let fixture_id =
  String.concat "/"
    [
      entry.artifact_id;
      "task";
      "toy-diagnostic";
      recipe_sha;
      "generator";
      String.make 40 'a';
      "request";
      request_sha;
    ]

let base_contract =
  obj
    [
      ("schema_version", J.int 1);
      ("tensor_format", J.string "torch-flat-tensor-map-v1");
      ("kind", J.string "diagnostic");
      ("fixture_id", J.string fixture_id);
      ("artifact_id", J.string entry.artifact_id);
      ("recipe_id", J.string "toy-diagnostic");
      ("recipe_sha256", J.string recipe_sha);
      ("request_sha256", J.string request_sha);
      ("references", J.list [ reference ]);
      ( "reference_publication",
        obj (List.remove_assoc "name" (get (members (pin pub_pin)))) );
      ("generator", generator);
      ( "model_loading",
        obj
          (List.map
             (fun k -> (k, J.list []))
             [
               "missing_keys";
               "unexpected_keys";
               "mismatched_keys";
               "error_msgs";
             ]) );
      ("tolerances", get (member "tolerances" contract));
      ( "expected_cases",
        J.list (List.map (fun row -> get (member "id" row)) case_rows) );
      ( "recipe",
        obj
          [
            ( "assets",
              obj
                [
                  ( "config.json",
                    set "url"
                      (J.string (W.url "config.json"))
                      (member_pin config_bytes) );
                ] );
          ] );
      ("cases", J.list case_rows);
      ("consumer_status", J.string "not_measured");
      ("producer_status", J.string "diagnostic");
    ]

module Fixture = struct
  type t = {
    request : T.Spec.Request.t;
    manifest : Jsont.json;
    served : (string * string) list;
  }
end

let build ?(mutate = Fun.id) () =
  let embedded = mutate base_contract in
  let case_members =
    List.concat
      (List.mapi
         (fun i (case : R.case) ->
           let id = Printf.sprintf "reference-00-case-%02d" i in
           ("raw/" ^ id ^ ".json", bytes (raw i))
           :: List.map
                (fun (role, tensors) ->
                  ("cases/" ^ id ^ "/" ^ role ^ ".pt", R.pt_file tensors))
                [
                  ("inputs", case.inputs);
                  ("published", case.outputs);
                  ("eager", case.outputs);
                  ("exported", case.outputs);
                ])
         R.base_cases)
  in
  let member_bytes =
    [
      ("task-contract.json", bytes embedded);
      ("environment.json", bytes generator);
      ("assets/config.json", config_bytes);
    ]
    @ case_members
  in
  let archive = W.gzip (W.tar member_bytes) in
  let archive_pin = pin_json "task.tar.gz" archive in
  let manifest =
    embedded
    |> set "archive" (obj (List.remove_assoc "url" (get (members archive_pin))))
    |> set "members"
         (obj
            (List.map
               (fun (name, data) -> (name, member_pin data))
               member_bytes))
  in
  let manifest_pin = pin_json "task.manifest.json" (bytes manifest) in
  let row =
    obj
      ([
         ("assets", obj [ ("archive", archive_pin); ("manifest", manifest_pin) ]);
       ]
      @ List.map
          (fun k -> (k, get (member k embedded)))
          [
            "artifact_id";
            "fixture_id";
            "kind";
            "recipe_id";
            "recipe_sha256";
            "expected_cases";
          ])
  in
  let index =
    obj
      [
        ("schema_version", J.int 1);
        ("fixtures", J.list [ row ]);
        ("request_sha256", J.string request_sha);
        ("reference_publication", get (member "reference_publication" embedded));
      ]
  in
  let request =
    T.Spec.Request.
      {
        fixture_id;
        generator = String.make 40 'a';
        index = get (pin_of (pin_json "task-fixtures.json" (bytes index)));
      }
  in
  let served =
    (W.url "publication.json", bytes pub)
    :: List.remove_assoc (W.url "publication.json") rel.served
    @ [
        (W.url "task.manifest.json", bytes manifest);
        (W.url "task.tar.gz", archive);
        (W.url "task-fixtures.json", bytes index);
      ]
  in
  Fixture.{ request; manifest; served }

let online fixture dir calls =
  let cache = get (U.Cache.create dir) in
  let transport : U.Transport.t =
   fun ~url ~dest ->
    calls := url :: !calls;
    match List.assoc_opt url fixture.Fixture.served with
    | None -> Error "404"
    | Some bytes ->
        Out_channel.with_open_bin dest (fun oc -> output_string oc bytes);
        Ok ()
  in
  U.Bundle.config ~transport cache
