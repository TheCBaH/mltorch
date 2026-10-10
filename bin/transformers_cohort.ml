(* Offline regeneration/check by default; fetch is an explicit network mode.
   No producer execution, tensor conversion or reference computation. *)
open Err.Syntax
module T = Transformers_metadata
module U = Pt2_fixture_unix

let run mode selection_file source cache_dir cohort_file report_file =
  let* selection_json = T.Json_util.read selection_file in
  let* selection = T.Selection.of_json selection_json in
  let* producer =
    T.Producer.load ~consumer_root:(Unix.getcwd ()) ~root:source
  in
  let* cache = U.Cache.create cache_dir in
  let transport = if mode = "fetch" then Some U.Transport.curl else None in
  let* generated, report =
    T.Selection.generate (U.Bundle.config ?transport cache) selection producer
  in
  let* () =
    if mode = "check" then
      let* existing = T.Json_util.read cohort_file in
      T.Json_util.equal ~identity:cohort_file ~field:"regenerated cohort"
        existing generated
    else T.Json_util.write cohort_file generated
  in
  let* () = T.Json_util.write report_file report in
  Fmt.pr "verified %d selected artifacts; source %s; release producer %s@."
    (List.length selection.artifacts)
    producer.commit selection.producer_commit;
  Ok ()

let () =
  let result =
    try
      match Array.to_list Sys.argv with
      | [
       _;
       (("generate" | "check" | "fetch") as mode);
       selection;
       source;
       cache;
       cohort;
       report;
      ] ->
          run mode selection source cache cohort report
      | _ ->
          T.Json_util.invalid
            "usage: transformers_cohort (generate|check|fetch) SELECTION \
             SOURCE CACHE COHORT REPORT"
    with
    | Sys_error e -> Err.fail (`File_io ("metadata", e))
    | Unix.Unix_error (e, op, path) ->
        Err.fail (`File_io (path, op ^ ": " ^ Unix.error_message e))
  in
  match result with
  | Ok () -> ()
  | Error e ->
      Fmt.epr "transformers_cohort: %a@." T.Fault.pp_error (Err.Error.kind e);
      exit 2
