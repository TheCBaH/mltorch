open Err.Syntax
open Transformers_metadata.Json_util
open Spec
module Selection = Transformers_metadata.Selection

module Reference = struct
  type t = {
    bundle : U.Bundle.t;
    contract : Jsont.json;
    descriptor : Jsont.json;
  }
end

let load config (cohort : F.Cohort.t) descriptor =
  let* id = field "artifact_id" descriptor in
  let selection =
    Selection.
      {
        artifacts = [ (id, "task reference") ];
        producer_commit = cohort.release_producer_commit;
        publication = cohort.publication;
        release_tag = cohort.release_tag;
        repository = cohort.repository;
      }
  in
  let* publication, rows = Selection.verified_publication config selection in
  let* row = row_by "artifact_id" id rows in
  let* declared_assets = member "assets" descriptor in
  let* actual_assets = member "assets" row in
  let* () =
    equal ~identity:id ~field:"publication assets" declared_assets actual_assets
  in
  let* entry =
    Err.of_option (`Metadata_missing id)
      (List.find_opt
         (fun e -> e.F.Publication.artifact_id = id)
         publication.entries)
  in
  let* file = fetch config F.Fault.Manifest entry.manifest in
  let* bytes = U.Fetch.read file in
  let* manifest_json = parse bytes in
  let* manifest = F.Manifest.of_string bytes in
  let* producer = field "producer_commit" manifest_json in
  let* () =
    same ~identity:id ~field:"manifest producer" producer
      selection.producer_commit
  in
  let* reference_producer = field "producer_commit" descriptor in
  let* () =
    same ~identity:id ~field:"reference producer" reference_producer
      selection.producer_commit
  in
  let* ref_cohort, want =
    Selection.bootstrap selection "task reference" entry manifest
  in
  let* bundle = U.Bundle.ensure config ref_cohort want in
  let* () =
    Err.List.iter
      (fun (key, member_name) ->
        let* pin = member key descriptor in
        let* expected =
          Err.of_option (`Member_missing member_name)
            (Smap.find_opt member_name bundle.manifest.members)
        in
        let* sha = field "sha256" pin in
        let* size = member "size" pin >>= integer in
        require
          (sha = Pt2_sha256.Digest.to_hex expected.sha256
          && size = expected.size)
          ("reference " ^ key ^ " member pin"))
      [
        ("contract", "contract.json");
        ("graph", "models/model.json");
        ("map", want.map_member);
      ]
  in
  let* contract = U.Bundle.read_member bundle "contract.json" >>= parse in
  let* () =
    Err.List.iter
      (fun (key, original) ->
        let* actual = member key descriptor in
        let* fields = members contract in
        let expected =
          Option.value ~default:(J.null ()) (List.assoc_opt original fields)
        in
        equal ~identity:id ~field:original actual expected)
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
      ]
  in
  let* config_sha = path [ "config"; "sha256" ] descriptor >>= string in
  let* original_sha = path [ "weights"; "config_sha256" ] contract >>= string in
  let* () = same ~identity:id ~field:"model config" config_sha original_sha in
  Ok Reference.{ bundle; contract; descriptor }

let environment generator references =
  let* execution = field "consumer_execution" generator in
  let* () =
    require
      (execution = "not_measured")
      "producer cannot assert consumer execution"
  in
  let* _ = member "routes" generator >>= names in
  let* cpu = member "cpu" generator in
  let* () =
    Err.List.iter
      (fun key ->
        let+ _ = field key cpu in
        ())
      [
        "machine";
        "platform";
        "cpuinfo";
        "capability";
        "parallel";
        "torch_config";
      ]
  in
  let* () =
    Err.List.iter
      (fun key ->
        let* n = member key cpu >>= integer in
        require (n > 0L) ("producer " ^ key))
      [ "threads"; "interop_threads" ]
  in
  Err.List.iter
    (fun reference ->
      let* release =
        member "release_environment" reference.Reference.descriptor
      in
      Err.List.iter
        (fun key ->
          let* actual = member key generator in
          let* expected = member key release in
          equal ~identity:"generator"
            ~field:("release environment " ^ key)
            actual expected)
        [
          "architecture";
          "core_override";
          "core_revision";
          "core_source_sha256";
          "device";
          "lock_sha256";
          "python";
          "tools_sha256";
          "versions";
        ])
    references

module Bundle = struct
  type t = {
    dir : string;
    inventory : F.Manifest.member Smap.t;
    manifest : Jsont.json;
    references : Reference.t list;
    index : F.Cohort.Pin.t;
    archive_pin : F.Cohort.Pin.t;
    manifest_pin : F.Cohort.Pin.t;
  }
end

let read_member t name =
  let* () =
    require (Smap.mem name t.Bundle.inventory) ("undeclared task member " ^ name)
  in
  U.Fetch.read ~max_bytes:0x1000_0000 (Filename.concat t.dir name)

let ensure config cohort request =
  let* manifest, inventory, archive, manifest_pin =
    selected config cohort request
  in
  let* references = member "references" manifest >>= array in
  let* ids = Err.List.map (field "artifact_id") references in
  let* () = unique ids in
  let* () = require (ids <> [] && List.length ids <= 16) "reference count" in
  let* references = Err.List.map (load config cohort) references in
  let* generator = member "generator" manifest in
  let* () = environment generator references in
  let* assets = path [ "recipe"; "assets" ] manifest >>= members in
  let* () =
    Err.List.iter
      (fun (name, pin) ->
        let name = "assets/" ^ name in
        let* actual =
          Err.of_option (`Member_missing name) (Smap.find_opt name inventory)
        in
        let* sha = field "sha256" pin in
        let* size = member "size" pin >>= integer in
        let* _ = field "url" pin in
        require
          (sha = Pt2_sha256.Digest.to_hex actual.sha256 && size = actual.size)
          ("recipe asset bytes " ^ name))
      assets
  in
  let* () =
    Err.List.iter
      (fun r ->
        let* config = member "config" r.Reference.descriptor in
        let* sha = field "sha256" config in
        let* size = member "size" config >>= integer in
        let* actual =
          Err.of_option (`Member_missing "assets/config.json")
            (Smap.find_opt "assets/config.json" inventory)
        in
        require
          (sha = Pt2_sha256.Digest.to_hex actual.sha256 && size = actual.size)
          "task reference config bytes")
      references
  in
  let limits =
    U.Targz.
      {
        max_archive_bytes = 0x3000_0000;
        max_member_bytes = 0x1000_0000;
        max_members = 4096;
      }
  in
  let* dir =
    U.Bundle.ensure_inventory { config with limits } archive inventory
  in
  let bundle =
    Bundle.
      {
        dir;
        inventory;
        manifest;
        references;
        index = request.Request.index;
        archive_pin = archive;
        manifest_pin;
      }
  in
  let* embedded = read_member bundle "task-contract.json" >>= parse in
  let* fields = members manifest in
  let* () =
    equal ~identity:request.Request.fixture_id ~field:"embedded task contract"
      embedded
      (obj
         (List.filter
            (fun (key, _) -> key <> "archive" && key <> "members")
            fields))
  in
  let* environment_json = read_member bundle "environment.json" >>= parse in
  let+ () =
    equal ~identity:request.fixture_id ~field:"environment bytes"
      environment_json generator
  in
  bundle
