(* Usage: transformers_replay.exe COHORT.json CACHE_DIR [--report-dir DIR] [--dots exact|binary32-sequential] [ARTIFACT_ID...]

   Offline. Opens each selected artifact from CACHE_DIR (every layer verified,
   every capture proved), runs all of its published cases through Native direct
   execution and compares every output element. Prints one line per artifact and,
   with --report-dir, writes its full report as JSON. Exits 1 unless every
   selected artifact passed; a refusal (the engine cannot run the graph) is a
   failure of the run, reported as such. *)

open Err.Syntax
module Fixture = Pt2_fixture_unix.Fixture
module Report = Pt2_fixture.Report

let command line =
  let ic = Unix.open_process_in line in
  let out = In_channel.input_all ic in
  ignore (Unix.close_process_in ic);
  String.trim out

let consumer () =
  let head = command "git rev-parse --short=12 HEAD 2>/dev/null" in
  let dirty = command "git status --porcelain 2>/dev/null | wc -l" in
  Printf.sprintf "%s%s" head
    (if dirty = "0" || dirty = "" then ""
     else "+workspace(" ^ dirty ^ " changes)")

let flat id = String.concat "--" (String.split_on_char '/' id)

let () =
  let args = List.tl (Array.to_list Sys.argv) in
  let report_dir = ref None and dots = ref Direct.Binary64 in
  let rec flags = function
    | "--report-dir" :: dir :: rest ->
        report_dir := Some dir;
        flags rest
    | "--dots" :: "binary32-sequential" :: rest ->
        dots := Direct.Binary32_sequential;
        flags rest
    | "--dots" :: "exact" :: rest ->
        dots := Direct.Binary64;
        flags rest
    | x :: rest -> x :: flags rest
    | [] -> []
  in
  let args = flags args in
  let report_dir = !report_dir and dots = !dots in
  match args with
  | cohort_path :: cache_dir :: ids -> (
      let setup =
        let* text = Pt2_fixture_unix.Fetch.read cohort_path in
        let* cohort = Pt2_fixture.Cohort.of_string text in
        let* cache = Pt2_fixture_unix.Cache.create cache_dir in
        Err.return (cohort, Pt2_fixture_unix.Bundle.config cache)
      in
      match Err.payload setup with
      | Error e ->
          Fmt.epr "%a@." Fixture.pp_error e;
          exit 2
      | Ok (cohort, config) ->
          let consumer = consumer () in
          let entries =
            match ids with
            | [] -> cohort.entries
            | ids ->
                List.filter
                  (fun (e : Pt2_fixture.Cohort.entry) ->
                    List.mem e.artifact_id ids)
                  cohort.entries
          in
          let failed = ref false in
          List.iter
            (fun (entry : Pt2_fixture.Cohort.entry) ->
              let t0 = Unix.gettimeofday () in
              let run =
                let* f = Fixture.open_ config cohort entry in
                Pt2_fixture_replay.replay ~dots ~consumer f
              in
              match Err.payload run with
              | Error e ->
                  failed := true;
                  Fmt.pr "ERROR %s: %a@." entry.artifact_id Fixture.pp_error e
              | Ok (report : Report.t) ->
                  let name =
                    match report.status with
                    | Passed -> "PASS"
                    | Failed -> "FAIL"
                    | Refused -> "REFUSED"
                  in
                  if report.status <> Passed then failed := true;
                  Fmt.pr "%-8s %s (%d cases, %.1fs)@." name entry.artifact_id
                    (List.length report.cases)
                    (Unix.gettimeofday () -. t0);
                  List.iter
                    (fun (c : Report.case) ->
                      Option.iter
                        (fun e -> Fmt.pr "           %s: %s@." c.id e)
                        c.error;
                      List.iter
                        (fun (o : Pt2_fixture.Compare.t) ->
                          if not (Pt2_fixture.Compare.passed o) then
                            Fmt.pr
                              "           %s %s: %Ld/%Ld mismatches, max abs \
                               %g, max rel %g@."
                              c.id o.name o.mismatches o.elements
                              o.max_abs_error o.max_rel_error)
                        c.outputs)
                    report.cases;
                  Option.iter
                    (fun r -> Fmt.pr "           refusal: %s@." r)
                    report.refusal;
                  Option.iter
                    (fun dir ->
                      (try Unix.mkdir dir 0o755
                       with Unix.Unix_error (Unix.EEXIST, _, _) -> ());
                      Out_channel.with_open_bin
                        (Filename.concat dir
                           (flat entry.artifact_id
                           ^ (match dots with
                             | Direct.Binary64 -> ""
                             | Direct.Binary32_sequential ->
                                 ".binary32-sequential")
                           ^ ".replay.json"))
                        (fun oc -> output_string oc (Report.to_string report)))
                    report_dir)
            entries;
          if !failed then exit 1)
  | _ ->
      prerr_endline
        "usage: transformers_replay COHORT.json CACHE_DIR [--report-dir DIR] \
         [--dots exact|binary32-sequential] [ARTIFACT_ID...]";
      exit 2
