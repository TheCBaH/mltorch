open Err.Syntax
open Transformers_metadata.Json_util
module J = Jsont.Json
module F = Pt2_fixture
module U = Pt2_fixture_unix

let of_entry (entry : F.Cohort.entry) =
  let hex = Pt2_sha256.Digest.to_hex in
  obj
    [
      ("artifact_id", J.string entry.artifact_id);
      ("archive", pin entry.archive);
      ("manifest", pin entry.manifest);
      ("graph_sha256", J.string (hex entry.graph_sha256));
      ("contract_sha256", J.string (hex entry.contract_sha256));
      ("map_sha256", J.string (hex entry.map_sha256));
      ("map_member", J.string entry.map_member);
      ("sources", J.list (List.map pin entry.sources));
      ("cases", Identity.strings entry.cases);
    ]

let load config cohort entry =
  let* bundle = U.Bundle.ensure config cohort entry in
  let* bytes = U.Bundle.read_member bundle "contract.json" in
  let* contract = F.Contract.of_string bytes in
  let* cases_bytes = U.Bundle.read_member bundle "cases.json" in
  let* cases = F.Cases.of_string cases_bytes in
  let* () = F.Cases.check contract cases in
  let* () =
    same ~identity:entry.F.Cohort.artifact_id ~field:"contract artifact"
      contract.artifact_id entry.artifact_id
  in
  let* () =
    if List.map (fun c -> c.F.Cases.Case.id) cases.cases = entry.cases then
      Ok ()
    else invalid "cohort/contract case coverage"
  in
  let* json = parse bytes in
  let optional key =
    match Err.payload (member key json) with Ok v -> v | Error _ -> J.null ()
  in
  let* fields = members (of_entry entry) in
  let* outputs = member "outputs" json in
  let* inputs = member "inputs" json in
  let files =
    obj
      (List.map
         (fun (name, (m : F.Manifest.member)) ->
           ( name,
             obj
               [
                 ("sha256", J.string (Pt2_sha256.Digest.to_hex m.sha256));
                 ("size", J.number (Int64.to_float m.size));
               ] ))
         (Schema_runtime.String_map.bindings bundle.manifest.members))
  in
  Ok
    (obj
       (fields
       @ [
           ( "tolerances",
             obj
               [
                 ("atol", J.number contract.atol);
                 ("rtol", J.number contract.rtol);
               ] );
           ("outputs", outputs);
           ("inputs", inputs);
           ("files", files);
           ("producer", optional "producer");
           ("exporter", optional "exporter");
           ("weights", optional "weights");
           ("config_sha256", optional "config_sha256");
           ("recipe_sha256", optional "recipe_sha256");
           ("variant", optional "variant");
         ]))

let load_or_error config cohort entry =
  match Err.payload (load config cohort entry) with
  | Ok json -> json
  | Error error ->
      let fields =
        match Err.payload (members (of_entry entry)) with
        | Ok fields -> fields
        | Error _ -> assert false
      in
      obj
        (fields
        @ [
            ( "metadata_error",
              J.string (Fmt.str "%a" Transformers_metadata.Fault.pp_error error)
            );
          ])

let report_pins expected =
  let* archive = path [ "archive"; "sha256" ] expected in
  let* manifest = path [ "manifest"; "sha256" ] expected in
  let* graph = member "graph_sha256" expected in
  let* contract = member "contract_sha256" expected in
  let* map = member "map_sha256" expected in
  let* sources = member "sources" expected >>= array in
  let+ sources =
    Err.List.map
      (fun row ->
        let* name = field "name" row in
        let+ sha = member "sha256" row in
        ("source:" ^ name, sha))
      sources
  in
  obj
    ([
       ("archive", archive);
       ("manifest", manifest);
       ("graph", graph);
       ("contract", contract);
       ("map", map);
     ]
    @ sources)
