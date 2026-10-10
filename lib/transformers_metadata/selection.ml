(* Select only verified metadata. Reference tensor computation and conversion
   stay in the producer; network use exists only when a transport is supplied. *)
open Err.Syntax
open Json_util
module J = Jsont.Json
module U = Pt2_fixture_unix
module C = Pt2_fixture.Cohort
module D = Pt2_checkpoint_map.Document
module P = Pt2_fixture.Publication
module M = Pt2_fixture.Manifest
module Smap = Schema_runtime.String_map

type t = {
  artifacts : (string * string) list;
  producer_commit : string;
  publication : D.Pin.t;
  release_tag : string;
  repository : string;
}

let commit s =
  if
    String.length s = 40
    && String.for_all
         (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
         s
  then Ok ()
  else invalid "expected full producer commit"

let of_json j =
  let* () = schema 1 j in
  let* publication = member "publication" j >>= pin_of in
  let* producer_commit = field "release_producer_commit" j in
  let* () = commit producer_commit in
  let* release_tag = field "release_tag" j in
  let* repository = field "repository" j in
  let* rows = member "artifacts" j >>= array in
  let* () =
    if rows <> [] && List.length rows <= 1000 then Ok ()
    else invalid "selection artifact count"
  in
  let* artifacts =
    Err.List.map
      (fun r ->
        let* id = field "artifact_id" r in
        let+ role = field "role" r in
        (id, role))
      rows
  in
  let* () = unique (List.map fst artifacts) in
  Ok { artifacts; producer_commit; publication; release_tag; repository }

let row_by_id rows id =
  let rec find = function
    | [] -> missing ("publication artifact " ^ id)
    | r :: rest ->
        let* have = field "artifact_id" r in
        if have = id then Ok r else find rest
  in
  find rows

let verified_publication (config : U.Bundle.config) selection =
  let* file =
    U.Fetch.ensure ?transport:config.transport
      ~layer:Pt2_fixture.Fault.Publication config.cache selection.publication
  in
  let* bytes = U.Fetch.read file in
  let* j = parse bytes in
  let* () = schema 1 j in
  let* publication = P.of_string bytes in
  let* () =
    same ~identity:"publication" ~field:"release_tag" publication.release_tag
      selection.release_tag
  in
  let* () =
    same ~identity:"publication" ~field:"repository" publication.repository
      selection.repository
  in
  let* rows = member "artifacts" j >>= array in
  let* ids = Err.List.map (field "artifact_id") rows in
  let* () = unique ids in
  let* () =
    Err.List.iter
      (fun (id, _) ->
        let* r = row_by_id rows id in
        let* producer = field "producer_commit" r in
        same ~identity:id ~field:"producer_commit" producer
          selection.producer_commit)
      selection.artifacts
  in
  Ok (publication, rows)

let bootstrap (selection : t) role (entry : P.entry) (manifest : M.t) =
  let* m =
    Err.of_option (`Member_missing manifest.map_member)
      (Smap.find_opt manifest.map_member manifest.members)
  in
  let want =
    {
      C.artifact_id = entry.artifact_id;
      archive = entry.archive;
      cases = manifest.cases;
      contract_sha256 = manifest.contract_sha256;
      graph_sha256 = entry.graph_sha256;
      manifest = entry.manifest;
      map_member = manifest.map_member;
      map_sha256 = m.sha256;
      role;
      sources = manifest.map_assets;
    }
  in
  let cohort =
    {
      C.entries = [ want ];
      publication = selection.publication;
      release_producer_commit = selection.producer_commit;
      release_tag = selection.release_tag;
      repository = selection.repository;
    }
  in
  let* () = M.check want manifest in
  let* _ =
    P.check cohort
      {
        P.entries = [ entry ];
        release_tag = selection.release_tag;
        repository = selection.repository;
      }
      want
  in
  let* () =
    Err.List.iter
      (fun p ->
        match
          List.find_opt
            (fun (a : D.Pin.t) -> a.name = p.D.Pin.name)
            manifest.map_assets
        with
        | Some a -> Pt2_fixture.Check.pin (Pt2_fixture.Fault.Source p.name) p a
        | None -> missing ("manifest released source " ^ p.name))
      entry.sources
  in
  Ok (cohort, want)

let select_one config selection producer publication rows (id, role) =
  let* source, catalogue_row = Producer.source_artifact producer id in
  let* released = row_by_id rows id in
  let* entry =
    Err.of_option (`Metadata_missing id)
      (List.find_opt
         (fun (e : P.entry) -> e.artifact_id = id)
         publication.P.entries)
  in
  let* manifest_file =
    U.Fetch.ensure ?transport:config.U.Bundle.transport
      ~layer:Pt2_fixture.Fault.Manifest config.cache entry.manifest
  in
  let* manifest_bytes = U.Fetch.read manifest_file in
  let* manifest_json = parse manifest_bytes in
  let* () = schema 1 manifest_json in
  let* producer_commit = field "producer_commit" manifest_json in
  let* () =
    same ~identity:id ~field:"manifest producer_commit" producer_commit
      selection.producer_commit
  in
  let* manifest = M.of_string manifest_bytes in
  let* cohort, want = bootstrap selection role entry manifest in
  let* bundle = U.Bundle.ensure config cohort want in
  let* map_bytes = U.Bundle.read_member bundle want.map_member in
  let* map_json = parse map_bytes in
  let* document = D.of_string map_bytes in
  let* () =
    same ~identity:id ~field:"map artifact_id" document.artifact_id id
  in
  let* () =
    Pt2_fixture.Check.digest Pt2_fixture.Fault.Map document.graph_sha256
      entry.graph_sha256
  in
  let* contract_bytes = U.Bundle.read_member bundle "contract.json" in
  let* contract = parse contract_bytes in
  let* () = schema 1 contract in
  let* parts = Source.safe id in
  let* () =
    match parts with
    | [ _; _; _; _; _; _; shape; _ ]
      when String.starts_with ~prefix:"static-h" shape ->
        let* kind = path [ "variant"; "kind" ] contract >>= string in
        let* () =
          same ~identity:id ~field:"variant kind" kind "static-history"
        in
        let* history = path [ "variant"; "history" ] contract >>= integer in
        same ~identity:id ~field:"history snapshot"
          ("static-h" ^ Int64.to_string history)
          shape
    | _ -> Ok ()
  in
  let* model_bytes = U.Bundle.read_member bundle "models/model.json" in
  let* program = Pt2_archive.program_of_json model_bytes in
  let* weights_bytes =
    U.Bundle.read_member bundle "data/weights/model_weights_config.json"
  in
  let* weights = Pt2_archive.weights_config_of_json weights_bytes in
  let* constants_bytes =
    U.Bundle.read_member bundle "data/constants/model_constants_config.json"
  in
  let* constants = Pt2_archive.constants_config_of_json constants_bytes in
  let* captures_bytes = U.Bundle.read_member bundle "captures.json" in
  let* captures = Pt2_checkpoint_map.Captures.of_string captures_bytes in
  let* () =
    Pt2_checkpoint_map.Validate.check document
      {
        artifact_id = id;
        captures;
        constants;
        graph_digest = Pt2_sha256.string model_bytes;
        program;
        weights;
      }
  in
  let* contract_id = field "artifact_id" contract in
  let* () = same ~identity:id ~field:"contract artifact_id" contract_id id in
  let* model = field "model_id" released in
  let* () = same ~identity:id ~field:"map model_id" document.model_id model in
  let* contract_model = field "model_id" contract in
  let* () = same ~identity:id ~field:"contract model_id" contract_model model in
  let* source_model = field "id" source in
  let* () = same ~identity:id ~field:"source model_id" source_model model in
  let* weight_source = member "weight_source" released in
  let* kind = field "kind" weight_source in
  let* () = same ~identity:id ~field:"weight source" kind "checkpoint" in
  let* contract_weights = member "weights" contract in
  let* () =
    equal ~identity:id ~field:"contract checkpoint identity" contract_weights
      weight_source
  in
  let* () =
    Err.List.iter
      (fun key ->
        let* actual = field key weight_source in
        let* expected = path [ "reference"; key ] source >>= string in
        same ~identity:id ~field:("source reference " ^ key) actual expected)
      [ "repo"; "revision"; "config_sha256" ]
  in
  let* revision = field "revision" weight_source in
  let* () = commit revision in
  let* () =
    if String.ends_with ~suffix:("ckpt-" ^ String.sub revision 0 12) id then
      Ok ()
    else invalid ("checkpoint revision differs from artifact ID: " ^ id)
  in
  let* () =
    Err.List.iter
      (fun asset ->
        match
          List.find_opt
            (fun (s : D.Source.t) -> s.pin.name = asset.D.Pin.name)
            document.checkpoint_files
        with
        | None -> missing ("map released source " ^ asset.name)
        | Some s ->
            Pt2_fixture.Check.pin (Pt2_fixture.Fault.Source asset.name) s.pin
              asset)
      manifest.map_assets
  in
  let* graph = member "graph_sha256" released in
  let* contract_hash = member "contract_sha256" manifest_json in
  let* map_sources = member "sources" map_json in
  let* cases = member "cases" manifest_json in
  let* source_catalogue =
    match catalogue_row with
    | None -> Ok (J.null ())
    | Some row ->
        let* artifact = member "artifact_id" row in
        let+ graph = member "graph_sha256" row in
        obj [ ("artifact_id", artifact); ("graph_sha256", graph) ]
  in
  let source_info =
    obj
      [
        ("release_artifact_id", J.string id);
        ("model_id", J.string source_model);
        ("source_catalogue", source_catalogue);
        ( "recipe_inventory",
          J.string
            (if catalogue_row = None then "candidate forward recipe"
             else "catalogue component recipe") );
      ]
  in
  Ok
    ( obj
        [
          ("artifact_id", J.string id);
          ("role", J.string role);
          ("archive", pin entry.archive);
          ("manifest", pin entry.manifest);
          ("graph_sha256", graph);
          ("contract_sha256", contract_hash);
          ("map_member", J.string want.map_member);
          ("map_sha256", J.string (Pt2_sha256.Digest.to_hex want.map_sha256));
          ("cases", cases);
          ("released_sources", J.list (List.map pin entry.sources));
          ("map_sources", map_sources);
          ("capture_count", J.int (Smap.cardinal document.tensors));
          ("weight_source", weight_source);
        ],
      source_info )

let generate config selection producer =
  let* publication, rows = verified_publication config selection in
  let* selected =
    Err.List.map
      (select_one config selection producer publication rows)
      selection.artifacts
  in
  let root =
    obj
      [
        ("schema_version", J.int 1);
        ("release_tag", J.string selection.release_tag);
        ("repository", J.string selection.repository);
        ("release_producer_commit", J.string selection.producer_commit);
        ( "publication",
          obj
            [
              ("size", J.number (Int64.to_float selection.publication.size));
              ( "sha256",
                J.string (Pt2_sha256.Digest.to_hex selection.publication.sha256)
              );
              ("url", J.string selection.publication.url);
            ] );
        ("artifacts", J.list (List.map fst selected));
      ]
  in
  let* bytes = text root in
  let* cohort = C.of_string bytes in
  let sum sizes =
    Err.List.fold_left
      (fun total n ->
        if n < 0L || total > Int64.sub Int64.max_int n then
          invalid "resource-size overflow"
        else Ok (Int64.add total n))
      0L sizes
  in
  let* archive_bytes =
    sum (List.map (fun (e : C.entry) -> e.archive.size) cohort.entries)
  in
  let all_sources =
    List.concat_map (fun (e : C.entry) -> e.sources) cohort.entries
  in
  let unique_sources =
    List.sort_uniq
      (fun (a : D.Pin.t) (b : D.Pin.t) ->
        String.compare
          (Pt2_sha256.Digest.to_hex a.sha256)
          (Pt2_sha256.Digest.to_hex b.sha256))
      all_sources
  in
  let* checkpoint_bytes =
    sum (List.map (fun (p : D.Pin.t) -> p.size) unique_sources)
  in
  let report =
    obj
      [
        ("source_gitlink", J.string producer.Producer.commit);
        ("source_metadata", producer.metadata);
        ("source_artifacts", J.list (List.map snd selected));
        ("release_graphs_are_independent", J.bool true);
        ("artifacts", J.int (List.length selected));
        ( "resource_estimates",
          obj
            [
              ( "selected_archive_bytes",
                J.string (Int64.to_string archive_bytes) );
              ( "unique_checkpoint_bytes",
                J.string (Int64.to_string checkpoint_bytes) );
              ("unique_checkpoint_blobs", J.int (List.length unique_sources));
            ] );
        ("source_catalogue_schema", J.int 1);
        ("release_manifest_schema", J.int 1);
        ("release_map_schema", J.int 2);
      ]
  in
  Ok (root, report)
