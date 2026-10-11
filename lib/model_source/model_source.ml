module U = Pt2_fixture_unix

type t =
  | Fixture of { id : string; json : string option }
  | Timm of { name : string; dir : string; json : string }

type dirs = { cache : string; cohort : string; models : string }

let words name =
  String.split_on_char '.' name
  |> List.concat_map (String.split_on_char '/')
  |> List.filter (fun w -> w <> "")

let candidates ids name =
  if List.mem name ids then [ name ]
  else
    let words = words name in
    List.filter
      (fun id ->
        let parts = String.split_on_char '/' id in
        words <> [] && List.for_all (fun w -> List.mem w parts) words)
      ids

let resolve dirs name =
  let dir = Filename.concat dirs.models name in
  let json = Filename.concat dir "models/model.json" in
  if Sys.file_exists json then Ok (Timm { name; dir; json })
  else
    let ( let* ) = Result.bind in
    let* cohort =
      Err.payload
        (let open Err.Syntax in
         let* text = U.Fetch.read dirs.cohort in
         Pt2_fixture.Cohort.of_string text)
      |> Result.map_error (fun e ->
          Fmt.str "%s: %a" dirs.cohort U.Fixture.pp_error e)
    in
    let ids =
      List.map
        (fun (e : Pt2_fixture.Cohort.entry) -> e.artifact_id)
        cohort.entries
    in
    match candidates ids name with
    | [ id ] ->
        let entry = Option.get (Pt2_fixture.Cohort.find cohort id) in
        let json =
          Filename.concat
            (Filename.concat
               (Filename.concat dirs.cache "bundles")
               (Pt2_sha256.Digest.to_hex entry.archive.sha256))
            "models/model.json"
        in
        Ok
          (Fixture
             { id; json = (if Sys.file_exists json then Some json else None) })
    | [] ->
        Error
          (Printf.sprintf
             "%s is neither a model of %s nor a cohort artifact of %s" name
             dirs.models dirs.cohort)
    | many ->
        Error
          (Printf.sprintf "%s is ambiguous in %s:\n%s" name dirs.cohort
             (String.concat "\n" (List.map (fun id -> "  " ^ id) many)))

let checkpoint dir =
  Err.payload (Pt2_safetensors_unix.checkpoint_path dir)
  |> Result.map_error (fun e ->
      Fmt.str "%s: %a" dir Pt2_safetensors_unix.pp_error e)
