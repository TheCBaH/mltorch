(* Usage: transformers_fixture.exe (fetch|open) COHORT.json CACHE_DIR [ARTIFACT_ID...]

   fetch  download every layer of each artifact into CACHE_DIR, verifying each
          file against its pin; with no ARTIFACT_ID, every cohort artifact.
   open   offline: open each artifact from CACHE_DIR alone (bundle, sources,
          every capture proved) and report. Exits 1 if any artifact fails.

   Both run the same chain; the only difference is whether a transport exists. *)

open Err.Syntax
module Fixture = Pt2_fixture_unix.Fixture

let run ~online cohort_path cache_dir ids =
  let* text = Pt2_fixture_unix.Fetch.read cohort_path in
  let* cohort = Pt2_fixture.Cohort.of_string text in
  let* cache = Pt2_fixture_unix.Cache.create cache_dir in
  let transport =
    if online then Some Pt2_fixture_unix.Transport.curl else None
  in
  let config = Pt2_fixture_unix.Bundle.config ?transport cache in
  let entries =
    match ids with
    | [] -> cohort.entries
    | ids ->
        List.filter
          (fun (e : Pt2_fixture.Cohort.entry) -> List.mem e.artifact_id ids)
          cohort.entries
  in
  Err.return (cohort, config, entries)

let () =
  match Array.to_list Sys.argv with
  | _ :: (("fetch" | "open") as mode) :: cohort :: cache :: ids -> (
      match Err.payload (run ~online:(mode = "fetch") cohort cache ids) with
      | Error e ->
          Fmt.epr "%a@." Fixture.pp_error e;
          exit 2
      | Ok (cohort, config, entries) ->
          let failed = ref false in
          List.iter
            (fun (entry : Pt2_fixture.Cohort.entry) ->
              let t0 = Unix.gettimeofday () in
              match Err.payload (Fixture.open_ config cohort entry) with
              | Ok f ->
                  Fmt.pr
                    "ok    %s: %d captures, owned %Ld bytes, sources %Ld \
                     bytes, %.1fs@."
                    entry.artifact_id
                    (List.length
                       (Pt2_checkpoint_map.Prepare.targets f.captures))
                    (Pt2_checkpoint_map.Prepare.owned_bytes f.captures)
                    (Pt2_checkpoint_map.Prepare.source_bytes f.captures)
                    (Unix.gettimeofday () -. t0)
              | Error e ->
                  failed := true;
                  Fmt.pr "FAIL  %s: %a@." entry.artifact_id Fixture.pp_error e)
            entries;
          if !failed then exit 1)
  | _ ->
      prerr_endline
        "usage: transformers_fixture (fetch|open) COHORT.json CACHE_DIR \
         [ARTIFACT_ID...]";
      exit 2
