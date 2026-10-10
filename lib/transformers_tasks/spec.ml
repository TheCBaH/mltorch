open Err.Syntax
open Transformers_metadata.Json_util
module J = Jsont.Json
module U = Pt2_fixture_unix
module F = Pt2_fixture
module Smap = Schema_runtime.String_map

let require yes why = if yes then Ok () else invalid why
let names value = array value >>= Err.List.map string

let digest s =
  Err.of_option (`Metadata_invalid "sha256") (Pt2_sha256.Digest.of_hex s)

let check_fields value keys =
  let* fields = members value in
  require
    (List.sort String.compare (List.map fst fields)
    = List.sort String.compare keys)
    "unexpected or missing selection fields"

let publication_pin value =
  let* fields = members value in
  pin_of (obj (("name", J.string "publication.json") :: fields))

let equal_publication value pin =
  let* actual = publication_pin value in
  F.Check.pin F.Fault.Publication actual pin

module Request = struct
  type t = { fixture_id : string; generator : string; index : F.Cohort.Pin.t }

  let of_json value =
    let* () = check_fields value [ "fixture_id"; "generator"; "index" ] in
    let* fixture_id = field "fixture_id" value in
    let* generator = field "generator" value in
    let* () =
      require
        (String.length generator = 40
        && String.for_all
             (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
             generator)
        "full generator SHA"
    in
    let* index = member "index" value >>= pin_of in
    let+ () =
      require
        (index.size > 0L && index.size <= 0x400_0000L)
        "task index size bound"
    in
    { fixture_id; generator; index }
end

let requests value =
  let* () = schema 1 value in
  let* () = check_fields value [ "schema_version"; "fixtures" ] in
  let* rows =
    member "fixtures" value >>= array >>= Err.List.map Request.of_json
  in
  let* () = unique (List.map (fun r -> r.Request.fixture_id) rows) in
  let+ () =
    require (rows <> [] && List.length rows <= 100) "fixture selection size"
  in
  rows

let row_by key id rows =
  let* ids = Err.List.map (field key) rows in
  let* () = unique ids in
  match List.find_opt (fun (name, _) -> name = id) (List.combine ids rows) with
  | Some (_, row) -> Ok row
  | None -> missing (key ^ ": " ^ id)

let inventory manifest =
  let* rows = member "members" manifest >>= members in
  let* () =
    require
      (List.length rows > 0 && List.length rows <= 4096)
      "task member count"
  in
  let* rows =
    Err.List.map
      (fun (name, row) ->
        let* () =
          require (U.Targz.safe_name name) ("unsafe task member " ^ name)
        in
        let* sha256 = field "sha256" row >>= digest in
        let+ size = member "size" row >>= integer in
        (name, F.Manifest.{ sha256; size }))
      rows
  in
  let* _ =
    Err.List.fold_left
      (fun total (_, m) ->
        let* () =
          require
            (m.F.Manifest.size <= 0x1000_0000L)
            "task member exceeds 256 MiB"
        in
        let total = Int64.add total m.size in
        let+ () =
          require (total <= 0x3000_0000L) "task payload exceeds 768 MiB"
        in
        total)
      0L rows
  in
  Ok (Smap.of_seq (List.to_seq rows))

let archive row = path [ "assets"; "archive" ] row >>= pin_of
let manifest row = path [ "assets"; "manifest" ] row >>= pin_of

let fetch config layer pin =
  U.Fetch.ensure ?hash:config.U.Bundle.hash ?transport:config.transport ~layer
    config.cache pin

let selected config (cohort : F.Cohort.t) (request : Request.t) =
  let* file = fetch config F.Fault.Publication request.index in
  let* index = read file in
  let* () = schema 1 index in
  let* publication = member "reference_publication" index in
  let* () = equal_publication publication cohort.publication in
  let* rows = member "fixtures" index >>= array in
  let* row = row_by "fixture_id" request.fixture_id rows in
  let* manifest_pin = manifest row in
  let* archive_pin = archive row in
  let* () =
    require
      (manifest_pin.size <= 0x400_0000L
      && archive_pin.size > 0L
      && archive_pin.size <= 0x2000_0000L)
      "task download size bounds"
  in
  let* file = fetch config F.Fault.Manifest manifest_pin in
  let* manifest = read file in
  let* () = schema 1 manifest in
  let* () =
    Err.List.iter
      (fun key ->
        let* a = member key manifest in
        let* b = member key row in
        equal ~identity:request.fixture_id ~field:key a b)
      [
        "fixture_id";
        "artifact_id";
        "kind";
        "recipe_id";
        "recipe_sha256";
        "expected_cases";
      ]
  in
  let* () =
    member "reference_publication" manifest >>= fun p ->
    equal_publication p cohort.publication
  in
  let* commit = path [ "generator"; "commit" ] manifest >>= string in
  let* () =
    same ~identity:request.fixture_id ~field:"generator SHA" commit
      request.generator
  in
  let* request_sha = field "request_sha256" manifest in
  let* index_request = field "request_sha256" index in
  let* () =
    same ~identity:request.fixture_id ~field:"request SHA" request_sha
      index_request
  in
  let* _ = digest request_sha in
  let* recipe = field "recipe_id" manifest in
  let* recipe_sha = field "recipe_sha256" manifest in
  let* _ = digest recipe_sha in
  let* artifact = field "artifact_id" manifest in
  let identity =
    String.concat "/"
      [
        artifact;
        "task";
        recipe;
        recipe_sha;
        "generator";
        commit;
        "request";
        request_sha;
      ]
  in
  let* () =
    same ~identity:artifact ~field:"full fixture identity" request.fixture_id
      identity
  in
  let* archive = member "archive" manifest in
  let* () =
    Err.List.iter
      (fun key ->
        let* actual = member key archive in
        let* expected = path [ "assets"; "archive"; key ] row in
        equal ~identity:artifact ~field:("archive " ^ key) actual expected)
      [ "name"; "size"; "sha256" ]
  in
  let* format = field "tensor_format" manifest in
  let* () =
    require
      (format = "torch-flat-tensor-map-v1")
      "unsupported task tensor format"
  in
  let* cases = member "cases" manifest >>= array in
  let* case_ids = Err.List.map (field "id") cases in
  let* () = unique case_ids in
  let* expected = member "expected_cases" manifest >>= names in
  let* () =
    require (case_ids = expected && cases <> []) "task case coverage/order"
  in
  let* inventory = inventory manifest in
  Ok (manifest, inventory, archive_pin, manifest_pin)
