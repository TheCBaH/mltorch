(* Usage: transformers_map_check.exe BUNDLE_DIR...

   Each directory is an extracted slim bundle (graph, configs, captures.json,
   contract.json, models/safetensors.v2.json). Prints "ok" or the first fault
   per directory and exits 1 if any failed. Reads no checkpoint source. *)

open Err.Syntax
module Map = Pt2_checkpoint_map

(* Just the artifact identity; the full contract is a later layer. *)
let artifact_id_jsont =
  Jsont.Object.map ~kind:"contract" Fun.id
  |> Jsont.Object.mem "artifact_id" Jsont.string
  |> Jsont.Object.skip_unknown |> Jsont.Object.finish

type error = [ Map.Fault.error | Pt2_archive.error | `Contract_decode of string ]

let pp_error ppf : error -> unit = function
  | #Map.Fault.error as e -> Map.Fault.pp_error ppf e
  | #Pt2_archive.error as e -> Pt2_archive.pp_error ppf e
  | `Contract_decode m -> Fmt.pf ppf "failed to decode contract.json: %s" m

let check dir : (unit, [> error ]) Err.t =
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
  Map.Validate.check document
    {
      Map.Validate.artifact_id;
      captures;
      constants;
      graph_digest = Pt2_sha256.string model_json;
      program;
      weights;
    }

let () =
  let failed = ref false in
  Array.iteri
    (fun i dir ->
      if i > 0 then
        match Err.payload (check dir) with
        | Ok () -> Fmt.pr "ok    %s@." (Filename.basename dir)
        | Error e ->
            failed := true;
            Fmt.pr "FAIL  %s: %a@." (Filename.basename dir) pp_error e)
    Sys.argv;
  if !failed then exit 1
