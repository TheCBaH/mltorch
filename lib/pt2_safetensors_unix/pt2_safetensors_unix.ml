open Err.Syntax

type error =
  [ Pt2_archive.error
  | Pt2_safetensors.error
  | `Checkpoint of Hf_hub_safetensors.error
  | `Source of Hf_hub.Error.t ]

let pp_error ppf : error -> unit = function
  | #Pt2_archive.error as e -> Pt2_archive.pp_error ppf e
  | #Pt2_safetensors.error as e -> Pt2_safetensors.pp_error ppf e
  | `Checkpoint e -> Hf_hub_safetensors.pp_error ppf e
  | `Source e -> Fmt.pf ppf "invalid checkpoint source: %a" Hf_hub.Error.pp e

let read_map dir =
  let* map_json =
    Pt2_archive.read_file (Filename.concat dir "models/safetensors.json")
  in
  let* map = Pt2_safetensors.map_of_string map_json in
  (* Refuse before any download: a map that cannot run needs no checkpoint. *)
  match map.unmapped with
  | [] -> Err.return map
  | names -> Err.fail (`Unmapped_constants names)

(* The pinned checkpoint, mapped, after its sha256 and size agree with the pin. *)
let resolve ?env ?http (source : Pt2_safetensors.Checkpoint_map.Source.t) =
  let* repo =
    Hf_hub.Repo_id.of_string source.repo_id
    |> Err.import ~pos:__POS__ (fun e -> `Source e)
  in
  let* revision =
    Hf_hub.Revision.of_string source.revision
    |> Err.import ~pos:__POS__ (fun e -> `Source e)
  in
  let* memory, blob =
    Hf_hub_safetensors.open_mmap ?env ?http ~revision ~repo
      ~filename:source.filename ()
    |> Err.import ~pos:__POS__ (fun e -> `Checkpoint e)
  in
  let size = (Unix.LargeFile.stat blob.path).st_size in
  let* () = Pt2_safetensors.check_source source ~etag:blob.etag ~size in
  Err.return (memory, blob)

let checkpoint_path ?env ?http dir =
  let* map = read_map dir in
  let* _, blob = resolve ?env ?http map.source in
  Err.return blob.path

let open_dir ?env ?http dir =
  let file rel = Filename.concat dir rel in
  let* program_json = Pt2_archive.read_file (file "models/model.json") in
  let* program = Pt2_archive.program_of_json program_json in
  let* weights_json =
    Pt2_archive.read_file (file "data/weights/model_weights_config.json")
  in
  let* weights = Pt2_archive.weights_config_of_json weights_json in
  let constants_path = file "data/constants/model_constants_config.json" in
  let* constants =
    if Sys.file_exists constants_path then
      let* json = Pt2_archive.read_file constants_path in
      Pt2_archive.constants_config_of_json json
    else Err.return Pt2_archive.no_constants
  in
  let* map = read_map dir in
  let* memory, _ = resolve ?env ?http map.source in
  Pt2_safetensors.of_parts ~map ~program ~weights ~constants memory
