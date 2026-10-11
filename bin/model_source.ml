(* Where a model comes from, for the Makefile's model.* rules.

   model_source resolve MODELS_DIR COHORT CACHE_DIR NAME
     prints "KIND ID MODEL_JSON": a timm model of the producer submodule
     ("timm NAME MODELS_DIR/NAME/models/model.json") or a released Transformers
     fixture ("fixture ARTIFACT_ID BUNDLE/models/model.json", "-" while the
     fixture is not in the cache). A fixture NAME is an artifact id or the dotted
     words that single one out (smolvlm-256m.connector): each must be one
     /-separated component of exactly one id. An unknown or ambiguous name lists
     the candidates on stderr and exits 2.
   model_source checkpoint MODEL_DIR
     prints the path of a timm model's pinned Hub checkpoint, fetching it into
     HF_HUB_CACHE if absent (HF_HUB_OFFLINE=1 forbids the request). *)

module U = Pt2_fixture_unix

let die fmt =
  Format.kasprintf
    (fun s ->
      prerr_endline s;
      exit 2)
    fmt

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

let resolve models cohort_path cache_dir name =
  let json =
    Filename.concat (Filename.concat models name) "models/model.json"
  in
  if Sys.file_exists json then Printf.printf "timm %s %s\n" name json
  else
    let cohort =
      match
        Err.payload
          (let open Err.Syntax in
           let* text = U.Fetch.read cohort_path in
           Pt2_fixture.Cohort.of_string text)
      with
      | Ok c -> c
      | Error e -> die "%s: %a" cohort_path U.Fixture.pp_error e
    in
    let ids =
      List.map
        (fun (e : Pt2_fixture.Cohort.entry) -> e.artifact_id)
        cohort.entries
    in
    match candidates ids name with
    | [ id ] ->
        let entry = Option.get (Pt2_fixture.Cohort.find cohort id) in
        let bundle =
          Filename.concat
            (Filename.concat cache_dir "bundles")
            (Pt2_sha256.Digest.to_hex entry.archive.sha256)
        in
        let json = Filename.concat bundle "models/model.json" in
        Printf.printf "fixture %s %s\n" id
          (if Sys.file_exists json then json else "-")
    | [] ->
        die "%s is neither a model of %s nor a cohort artifact of %s" name
          models cohort_path
    | many ->
        die "%s is ambiguous in %s:\n%s" name cohort_path
          (String.concat "\n" (List.map (fun id -> "  " ^ id) many))

let checkpoint dir =
  match Err.payload (Pt2_safetensors_unix.checkpoint_path dir) with
  | Ok path -> print_endline path
  | Error e -> die "%s: %a" dir Pt2_safetensors_unix.pp_error e

let () =
  match Array.to_list Sys.argv with
  | [ _; "resolve"; models; cohort; cache; name ] ->
      resolve models cohort cache name
  | [ _; "checkpoint"; dir ] -> checkpoint dir
  | _ ->
      prerr_endline
        "usage: model_source resolve MODELS_DIR COHORT CACHE_DIR NAME\n\
        \       model_source checkpoint MODEL_DIR";
      exit 2
