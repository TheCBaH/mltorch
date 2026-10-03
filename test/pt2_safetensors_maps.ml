(* Check every committed models/<name>/models/safetensors.json in the model
   submodule against that model's own weight and constant configs, with no
   checkpoint and no network: every captured tensor mapped, dtype and shape
   equal, layout dense. The three models whose non-persistent buffers the
   checkpoint lacks are expected to be refused, and show here as such.
   argv: <models directory> *)

let ( let* ) = Result.bind

let check dir =
  let file rel = Filename.concat dir rel in
  let render f e = Format.asprintf "%a" f (Err.Error.kind e) in
  let* map_json =
    Pt2_archive.read_file (file "models/safetensors.json")
    |> Result.map_error (render Pt2_archive.pp_error)
  in
  let* map =
    Pt2_safetensors.map_of_string map_json
    |> Result.map_error (render Pt2_safetensors.pp_error)
  in
  let* weights_json =
    Pt2_archive.read_file (file "data/weights/model_weights_config.json")
    |> Result.map_error (render Pt2_archive.pp_error)
  in
  let* weights =
    Pt2_archive.weights_config_of_json weights_json
    |> Result.map_error (render Pt2_archive.pp_error)
  in
  let constants_path = file "data/constants/model_constants_config.json" in
  let* constants =
    if Sys.file_exists constants_path then
      let* json =
        Pt2_archive.read_file constants_path
        |> Result.map_error (render Pt2_archive.pp_error)
      in
      Pt2_archive.constants_config_of_json json
      |> Result.map_error (render Pt2_archive.pp_error)
    else Ok Pt2_archive.no_constants
  in
  let* () =
    Pt2_safetensors.check_graph ~map ~weights ~constants
    |> Result.map_error (render Pt2_safetensors.pp_error)
  in
  Ok (Schema_runtime.String_map.cardinal map.tensors)

let () =
  let root = Sys.argv.(1) in
  let names = Sys.readdir root in
  Array.sort String.compare names;
  Array.iter
    (fun name ->
      let dir = Filename.concat root name in
      if Sys.file_exists (Filename.concat dir "models/safetensors.json") then
        match check dir with
        | Ok n -> Printf.printf "%s: ok, %d tensors\n" name n
        | Error msg -> Printf.printf "%s: refused, %s\n" name msg)
    names
