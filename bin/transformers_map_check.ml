(* Usage: transformers_map_check.exe [--sources DIR] BUNDLE_DIR...

   Each BUNDLE_DIR is an extracted slim bundle (graph, configs, captures.json,
   contract.json, models/safetensors.v2.json). Prints "ok" or the first fault
   per directory and exits 1 if any failed.

   Without --sources the map is only checked against the graph. With it, the
   files the map declares are read from DIR (by their pinned names), verified
   against their pins, and every capture is produced and hashed. *)

open Err.Syntax
module Map = Pt2_checkpoint_map

(* Just the artifact identity; the full contract is a later layer. *)
let artifact_id_jsont =
  Jsont.Object.map ~kind:"contract" Fun.id
  |> Jsont.Object.mem "artifact_id" Jsont.string
  |> Jsont.Object.skip_unknown |> Jsont.Object.finish

type error =
  [ Map.Fault.error
  | Pt2_archive.error
  | Pt2_checkpoint_map_unix.error
  | `Contract_decode of string ]

let pp_error ppf : error -> unit = function
  | #Map.Fault.error as e -> Map.Fault.pp_error ppf e
  | #Pt2_archive.error as e -> Pt2_archive.pp_error ppf e
  | #Pt2_checkpoint_map_unix.error as e ->
      Pt2_checkpoint_map_unix.pp_error ppf e
  | `Contract_decode m -> Fmt.pf ppf "failed to decode contract.json: %s" m

let check ~sources dir : (string, [> error ]) Err.t =
  let read rel = Pt2_archive.read_file (Filename.concat dir rel) in
  let* contract = read "contract.json" in
  let* artifact_id =
    Jsont_bytesrw.decode_string artifact_id_jsont contract
    |> Err.import ~pos:__POS__ (fun e -> `Contract_decode e)
  in
  let* model_json = read "models/model.json" in
  let* program = Pt2_archive.program_of_json model_json in
  let* weights_json = read "data/weights/model_weights_config.json" in
  let* weights = Pt2_archive.weights_config_of_json weights_json in
  let* constants_json = read "data/constants/model_constants_config.json" in
  let* constants = Pt2_archive.constants_config_of_json constants_json in
  let* captures_json = read "captures.json" in
  let* captures = Map.Captures.of_string captures_json in
  let* map_json = read "models/safetensors.v2.json" in
  let* document = Map.Document.of_string map_json in
  let* () =
    Map.Validate.check document
      {
        Map.Validate.artifact_id;
        captures;
        constants;
        graph_digest = Pt2_sha256.string model_json;
        program;
        weights;
      }
  in
  match sources with
  | None -> Err.return "map matches graph"
  | Some dir ->
      let t0 = Unix.gettimeofday () in
      let* srcs = Pt2_checkpoint_map_unix.sources_in_dir document ~dir in
      let* verified = Map.Prepare.verify_sources document srcs in
      let* set = Map.Prepare.capture_set document verified in
      Err.return
        (Fmt.str
           "%d captures prepared, owned %Ld bytes, sources %Ld bytes, %.1fs"
           (List.length (Map.Prepare.targets set))
           (Map.Prepare.owned_bytes set)
           (Map.Prepare.source_bytes set)
           (Unix.gettimeofday () -. t0))

let () =
  let sources, first =
    match Array.to_list Sys.argv with
    | _ :: "--sources" :: dir :: _ -> (Some dir, 3)
    | _ -> (None, 1)
  in
  let failed = ref false in
  Array.iteri
    (fun i dir ->
      if i >= first then
        match Err.payload (check ~sources dir) with
        | Ok note -> Fmt.pr "ok    %s: %s@." (Filename.basename dir) note
        | Error e ->
            failed := true;
            Fmt.pr "FAIL  %s: %a@." (Filename.basename dir) pp_error e)
    Sys.argv;
  if !failed then exit 1
