(* Validate full-cohort policy choices. This does not assert numerical success. *)
open Err.Syntax
open Transformers_metadata.Json_util

let () =
  let run () =
    match Array.to_list Sys.argv with
    | [ _; cohort; policies ] ->
        let* cohort_bytes = Pt2_fixture_unix.Fetch.read cohort in
        let* json = read policies in
        let+ entries =
          Transformers_reports.Policy_selection.of_json ~cohort_bytes json
        in
        Fmt.pr
          "%d explicit component policies, %d required core gates; numerical \
           results remain separate@."
          (List.length entries)
          (List.length (Transformers_reports.Policy_selection.core entries))
    | _ -> invalid "usage: transformers_policies COHORT POLICY_SELECTION"
  in
  match Err.payload (run ()) with
  | Ok () -> ()
  | Error error ->
      Fmt.epr "%a@." Transformers_metadata.Fault.pp_error error;
      exit 2
