(* Offline, complete published-case replay. Every invocation retains its run
   manifest, reports (including acquisition errors) and completion digest. *)
open Err.Syntax
open Transformers_metadata.Json_util
module Fixture = Pt2_fixture_unix.Fixture
module Report = Pt2_fixture.Report
module R = Transformers_reports

let fail error =
  Fmt.epr "%a@." Transformers_metadata.Fault.pp_error error;
  exit 2

let () =
  let report_dir = ref "_build/transformers-reports"
  and source = ref "modules/devcontainer.transformers"
  and dots = ref Direct.Binary64
  and casts = ref Direct.Checked in
  let rec flags = function
    | "--report-dir" :: dir :: rest ->
        report_dir := dir;
        flags rest
    | "--source" :: dir :: rest ->
        source := dir;
        flags rest
    | "--dots" :: "binary32-sequential" :: rest ->
        dots := Direct.Binary32_sequential;
        flags rest
    | "--dots" :: "exact" :: rest ->
        dots := Direct.Binary64;
        flags rest
    | "--casts" :: "saturating" :: rest ->
        casts := Direct.Saturating;
        flags rest
    | "--casts" :: "checked" :: rest ->
        casts := Direct.Checked;
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
    | cohort_path :: cache_dir :: ids ->
        let* () = unique ids in
        let* cohort_text = Pt2_fixture_unix.Fetch.read cohort_path in
        let* cohort = Pt2_fixture.Cohort.of_string cohort_text in
        let* cache = Pt2_fixture_unix.Cache.create cache_dir in
        let config = Pt2_fixture_unix.Bundle.config cache in
        let* () =
          Err.List.iter
            (fun id ->
              if
                List.exists
                  (fun e -> e.Pt2_fixture.Cohort.artifact_id = id)
                  cohort.entries
              then Ok ()
              else missing ("requested artifact " ^ id))
            ids
        in
        let entries =
          List.filter
            (fun e -> ids = [] || List.mem e.Pt2_fixture.Cohort.artifact_id ids)
            cohort.entries
        in
        let* () =
          if entries = [] then invalid "empty cohort selection" else Ok ()
        in
        let capture () =
          R.Identity.capture ~consumer_root:"." ~source:!source
            ~cohort:cohort_path
        in
        let* identity = capture () in
        let* consumer = path [ "consumer"; "commit" ] identity >>= string in
        let policy =
          R.Run.Policy.
            {
              casts =
                (match !casts with
                | Direct.Checked -> "checked"
                | Direct.Saturating -> "saturating");
              dots =
                (match !dots with
                | Direct.Binary64 -> "exact"
                | Direct.Binary32_sequential -> "binary32-sequential");
            }
        in
        let expected =
          List.map (R.Expectation.load_or_error config cohort) entries
        in
        let* run = R.Run.create ~root:!report_dir ~identity ~policy ~expected in
        Fmt.pr "Run: %s@." run.dir;
        let failed = ref false in
        let* () =
          Err.List.iter
            (fun (entry, expected) ->
              let t0 = Unix.gettimeofday () in
              let replay =
                let* f = Fixture.open_ config cohort entry in
                Pt2_fixture_replay.replay ~dots:!dots ~casts:!casts ~consumer f
              in
              let* json =
                match Err.payload replay with
                | Error error ->
                    failed := true;
                    let error = Fmt.str "%a" Fixture.pp_error error in
                    Fmt.pr "ERROR %s: %s@." entry.Pt2_fixture.Cohort.artifact_id
                      error;
                    R.Run.acquisition_error ~consumer ~policy ~expected error
                | Ok report ->
                    if report.Report.status <> Report.Passed then failed := true;
                    Fmt.pr "%-8s %s (%d cases, %.1fs)@."
                      (match report.status with
                      | Report.Passed -> "passed"
                      | Report.Failed -> "failed"
                      | Report.Refused -> "refused")
                      report.artifact_id (List.length report.cases)
                      (Unix.gettimeofday () -. t0);
                    List.iter
                      (fun (c : Report.case) ->
                        Option.iter
                          (fun e -> Fmt.pr "  %s: %s@." c.id e)
                          c.error;
                        List.iter
                          (fun (o : Pt2_fixture.Compare.t) ->
                            if not (Pt2_fixture.Compare.passed o) then
                              Fmt.pr
                                "  %s %s: %Ld/%Ld mismatches, max abs %g, max \
                                 rel %g@."
                                c.id o.name o.mismatches o.elements
                                o.max_abs_error o.max_rel_error)
                          c.outputs)
                      report.cases;
                    Option.iter
                      (fun e -> Fmt.pr "  refusal: %s@." e)
                      report.refusal;
                    parse (Report.to_string report)
              in
              R.Run.add run json)
            (List.combine entries expected)
        in
        let* final_identity = capture () in
        let+ () = R.Run.finish run ~identity:final_identity in
        if !failed then 1 else 0
    | _ ->
        invalid
          "usage: transformers_replay COHORT CACHE [--source DIR] \
           [--report-dir DIR] [--dots exact|binary32-sequential] [--casts \
           checked|saturating] [ARTIFACT_ID...]"
  in
  try
    match Err.payload (work ()) with
    | Ok code -> exit code
    | Error error -> fail error
  with
  | Unix.Unix_error (e, fn, arg) ->
      Fmt.epr "%s %s: %s@." fn arg (Unix.error_message e);
      exit 2
  | Sys_error error ->
      Fmt.epr "%s@." error;
      exit 2
