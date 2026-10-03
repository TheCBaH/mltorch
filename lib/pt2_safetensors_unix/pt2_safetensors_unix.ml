open Err.Syntax

module Source_mismatch = struct
  type t =
    | Sha256 of { actual : string option; expected : string }
    | Size of { actual : int64; expected : int64 }
end

type error =
  [ Pt2_archive.error
  | Pt2_safetensors.error
  | `Checkpoint of Hf_hub_safetensors.error
  | `Source of Hf_hub.Error.t
  | `Source_mismatch of Source_mismatch.t ]

let pp_error ppf : error -> unit = function
  | #Pt2_archive.error as e -> Pt2_archive.pp_error ppf e
  | #Pt2_safetensors.error as e -> Pt2_safetensors.pp_error ppf e
  | `Checkpoint e -> Hf_hub_safetensors.pp_error ppf e
  | `Source e -> Fmt.pf ppf "invalid checkpoint source: %a" Hf_hub.Error.pp e
  | `Source_mismatch (Sha256 { actual; expected }) ->
      Fmt.pf ppf "checkpoint sha256 is %a, safetensors.json pins %s"
        Fmt.(option ~none:(any "unknown") string)
        actual expected
  | `Source_mismatch (Size { actual; expected }) ->
      Fmt.pf ppf "checkpoint is %Ld bytes, safetensors.json pins %Ld" actual
        expected

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
  let* map_json = Pt2_archive.read_file (file "models/safetensors.json") in
  let* map = Pt2_safetensors.map_of_string map_json in
  (* Refuse before any download: a map that cannot run needs no checkpoint. *)
  let* () =
    match map.unmapped with
    | [] -> Err.return ()
    | names -> Err.fail (`Unmapped_constants names)
  in
  let source = map.source in
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
  let* () =
    if blob.Hf_hub.Blob.etag = Some source.sha256 then Err.return ()
    else
      Err.fail
        (`Source_mismatch
           (Source_mismatch.Sha256
              { actual = blob.etag; expected = source.sha256 }))
  in
  let* () =
    match Unix.LargeFile.stat blob.path with
    | { Unix.LargeFile.st_size; _ } when Int64.equal st_size source.size ->
        Err.return ()
    | { st_size; _ } ->
        Err.fail
          (`Source_mismatch
             (Source_mismatch.Size { actual = st_size; expected = source.size }))
  in
  Pt2_safetensors.of_parts ~map ~program ~weights ~constants memory
