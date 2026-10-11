open Err.Syntax
module J = Json
module Pin = Pt2_checkpoint_map.Document.Pin

let ( >>= ) = Err.bind

module Asset = struct
  type t = { format : string; members : string list; pin : Pin.t }
end

module Model = struct
  type t = {
    archive : string;
    evaluation : string;
    graph : Pin.t;
    inputs : string;
    weight_map : Pin.t option;
  }
end

type t = {
  assets : (string * Asset.t) list;
  commit : string;
  models : (string * Model.t) list;
  repo : string;
  tag : string;
}

let pin name j =
  let* sha256 = J.field "sha256" j >>= J.digest in
  let* size = J.member "bytes" j >>= J.integer in
  let+ url = J.field "url" j in
  { Pin.name; sha256; size = Int64.of_int size; url }

let filename s =
  s <> "" && s <> "." && s <> ".."
  && String.for_all
       (function
         | 'A' .. 'Z' | 'a' .. 'z' | '0' .. '9' | '_' | '-' | '.' -> true
         | _ -> false)
       s

let asset j =
  let* name = J.field "filename" j in
  let* () =
    if filename name then Ok () else J.bad J.Failure.Invalid_asset_filename
  in
  let* pin = pin name j in
  let* format = J.field "format" j in
  let* () =
    if List.mem format [ "json"; "torch.save"; "zip" ] then Ok ()
    else J.bad J.Failure.Unsupported_asset_format
  in
  let* members =
    if format = "zip" then
      J.member "members" j >>= J.array >>= Err.List.map J.string
    else Ok []
  in
  let+ () = J.unique members in
  { Asset.format; members; pin }

let model name j =
  let* status = J.field "status" j in
  let* () = J.same "model status" status "current" in
  let* archive = J.field "archive" j in
  let* evaluation = J.field "evaluation" j in
  let* graph = J.member "graph" j >>= pin (name ^ "/graph") in
  let* inputs = J.field "inputs" j in
  let* map = J.member "weight_map" j in
  let+ weight_map =
    match map with
    | Jsont.Null _ -> Ok None
    | j ->
        let+ p = pin (name ^ "/weight-map") j in
        Some p
  in
  { Model.archive; evaluation; graph; inputs; weight_map }

let asset_of t id =
  Err.of_option (`Timm_fixture (J.Failure.Unresolved_asset id))
    (List.assoc_opt id t.assets)

let of_string text =
  let* j = J.parse text in
  let* () = J.version 2 j in
  let* producer = J.member "producer" j in
  let* repo = J.field "repo" producer in
  let* tag = J.field "tag" producer in
  let* commit = J.field "commit" producer in
  let* () =
    if
      String.length commit = 40
      && String.for_all
           (function '0' .. '9' | 'a' .. 'f' -> true | _ -> false)
           commit
    then Ok ()
    else J.bad J.Failure.Invalid_producer_commit
  in
  let* asset_rows = J.member "assets" j >>= J.members in
  let* () =
    if List.length asset_rows > 2048 then J.bad J.Failure.Too_many_assets
    else Ok ()
  in
  let* assets =
    Err.List.map
      (fun (id, j) ->
        let+ a = asset j in
        (id, a))
      asset_rows
  in
  let* () = J.unique (List.map (fun (_, a) -> a.Asset.pin.name) assets) in
  let* rows = J.member "models" j >>= J.members in
  let* () =
    if rows = [] || List.length rows > 512 then
      J.bad J.Failure.Unsupported_model_count
    else Ok ()
  in
  let* models =
    Err.List.map
      (fun (name, j) ->
        let+ m = model name j in
        (name, m))
      rows
  in
  let document = { assets; commit; models; repo; tag } in
  let lookup id =
    Err.of_option (`Timm_fixture (J.Failure.Unresolved_asset id))
      (List.assoc_opt id assets)
  in
  let* () =
    Err.List.iter
      (fun (_, a) ->
        J.same "release asset URL" a.Asset.pin.url
          (Printf.sprintf "https://github.com/%s/releases/download/%s/%s" repo
             tag a.pin.name))
      assets
  in
  let* () =
    Err.List.iter
      (fun (id, name, format) ->
        let* a = lookup id in
        let* () = J.same "shared asset filename" a.pin.name name in
        J.same "shared asset format" a.format format)
      [
        ("decoded_images", "images.pt", "torch.save");
        ("image_metadata", "images-metadata.json", "json");
        ("legacy_images", "images.zip", "zip");
      ]
  in
  let* () =
    Err.List.iter
      (fun (name, m) ->
        let* () =
          Err.List.iter
            (fun (field, value, suffix, format) ->
              let* () =
                J.same ("model " ^ field ^ " asset") value (name ^ "/" ^ field)
              in
              let* a = lookup value in
              let* () =
                J.same "model asset filename" a.pin.name (name ^ suffix)
              in
              J.same "model asset format" a.format format)
            [
              ("archive", m.Model.archive, ".zip", "zip");
              ("evaluation", m.evaluation, "-eval.zip", "zip");
              ("inputs", m.inputs, "-inputs.pt", "torch.save");
            ]
        in
        let base =
          Printf.sprintf "https://raw.githubusercontent.com/%s/%s/models/%s/"
            repo commit name
        in
        let* () = J.same "graph URL" m.graph.url (base ^ "models/model.json") in
        match m.weight_map with
        | None -> Ok ()
        | Some p ->
            J.same "weight map URL" p.url (base ^ "models/safetensors.json"))
      models
  in
  let required =
    List.sort String.compare
      ([ "decoded_images"; "image_metadata"; "legacy_images" ]
      @ List.concat_map
          (fun (_, m) -> [ m.Model.archive; m.evaluation; m.inputs ])
          models)
  in
  let* () =
    J.same "indexed asset coverage" required
      (List.sort String.compare (List.map fst assets))
  in
  let* population = J.member "release_models" j in
  let* count = J.member "count" population >>= J.integer in
  let* names =
    J.member "names" population >>= J.array >>= Err.List.map J.string
  in
  let* () = J.unique names in
  let* () = J.same "release population count" count (List.length names) in
  let* scope = J.field "scope" j in
  let+ () =
    match scope with
    | "full" ->
        J.same "full model coverage"
          (List.sort String.compare names)
          (List.sort String.compare (List.map fst models))
    | "smoke" ->
        if List.for_all (fun (n, _) -> List.mem n names) models then Ok ()
        else J.bad J.Failure.Smoke_model_outside_release_population
    | _ -> J.bad J.Failure.Unsupported_manifest_scope
  in
  document
