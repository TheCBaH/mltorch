(* Offline verification of immutable replay runs against current source and
   verified release contracts. Legacy reports remain historical. *)
open Err.Syntax
open Transformers_metadata.Json_util
module R = Transformers_reports

let () =
  let source = ref "modules/devcontainer.transformers"
  and json_path = ref "_build/transformers-matrix.json"
  and markdown_path = ref "_build/transformers-matrix.md" in
  let rec flags = function
    | "--source" :: path :: rest ->
        source := path;
        flags rest
    | "--json" :: path :: rest ->
        json_path := path;
        flags rest
    | "--markdown" :: path :: rest ->
        markdown_path := path;
        flags rest
    | x :: _ when String.starts_with ~prefix:"--" x ->
        invalid ("unknown/incomplete option: " ^ x)
    | x :: rest ->
        let+ rest = flags rest in
        x :: rest
    | [] -> Ok []
  in
  let work () =
    let* args = flags (List.tl (Array.to_list Sys.argv)) in
    match args with
    | cohort_path :: cache_dir :: (_ :: _ as roots) ->
        let* bytes = Pt2_fixture_unix.Fetch.read cohort_path in
        let* cohort = Pt2_fixture.Cohort.of_string bytes in
        let* cache = Pt2_fixture_unix.Cache.create cache_dir in
        let config = Pt2_fixture_unix.Bundle.config cache in
        let* identity =
          R.Identity.capture ~consumer_root:"." ~source:!source
            ~cohort:cohort_path
        in
        let expected =
          List.map
            (fun entry ->
              ( entry.Pt2_fixture.Cohort.artifact_id,
                R.Expectation.load_or_error config cohort entry ))
            cohort.entries
        in
        let matrix = R.Matrix.generate ~identity ~expected roots in
        let* markdown = R.Matrix.markdown matrix in
        let* () = write !json_path matrix in
        Out_channel.with_open_bin !markdown_path (fun oc ->
            output_string oc markdown);
        Fmt.pr "%s" markdown;
        let+ passed = member "passed" matrix >>= bool in
        if passed then 0 else 1
    | _ ->
        invalid
          "usage: transformers_matrix COHORT CACHE REPORT_DIR... [--source \
           DIR] [--json FILE] [--markdown FILE]"
  in
  try
    match Err.payload (work ()) with
    | Ok code -> exit code
    | Error error ->
        Fmt.epr "%a@." Transformers_metadata.Fault.pp_error error;
        exit 2
  with
  | Unix.Unix_error (e, fn, arg) ->
      Fmt.epr "%s %s: %s@." fn arg (Unix.error_message e);
      exit 2
  | Sys_error error ->
      Fmt.epr "%s@." error;
      exit 2
