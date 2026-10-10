(* Producer-owned tensor fixtures are data. Fetch is explicit; check and
   diagnostics verify pins and complete tensor sets offline, never Python.
   Immutable reports distinguish fixture integrity from model acceptance. *)
open Err.Syntax
open Transformers_metadata.Json_util
module T = Transformers_tasks
module U = Pt2_fixture_unix
module J = Jsont.Json

let () =
  let work () =
    match Array.to_list Sys.argv with
    | [
     _;
     (("fetch" | "check" | "diagnostics" | "adapters" | "models" | "generation")
      as mode);
     selection;
     cohort_path;
     cache_path;
     output;
    ] ->
        let* requests = read selection >>= T.Spec.requests in
        let* bytes = U.Fetch.read cohort_path in
        let* cohort = Pt2_fixture.Cohort.of_string bytes in
        let* cache = U.Cache.create cache_path in
        let config =
          U.Bundle.config
            ?transport:(if mode = "fetch" then Some U.Transport.curl else None)
            cache
        in
        let context () =
          T.Runtime.context
            ~model_execution:(mode = "models" || mode = "generation")
            ~cohort:cohort_path ~selection ~executable:Sys.executable_name ()
        in
        let* initial = context () in
        let* run = T.Runtime.create output initial in
        Fmt.pr "Run: %s@." run.dir;
        let failed = ref false in
        let* () =
          Err.List.iter
            (fun request ->
              let result =
                let* bundle = T.Reference.ensure config cohort request in
                let* result =
                  if mode = "diagnostics" then T.Diagnostic.run bundle
                  else if
                    mode = "adapters" || mode = "models" || mode = "generation"
                  then
                    let* producer =
                      Transformers_metadata.Producer.load ~consumer_root:"."
                        ~root:"modules/devcontainer.transformers"
                    in
                    if mode = "generation" then
                      T.Generation.run config producer bundle
                    else
                      T.Acceptance.run config producer ~models:(mode = "models")
                        bundle
                  else
                    let* cases = member "cases" bundle.manifest >>= array in
                    let+ () =
                      Err.List.iter (T.Tensors.verify_case bundle) cases
                    in
                    obj
                      [
                        ("schema_version", J.int 1);
                        ( "fixture_id",
                          J.string request.T.Spec.Request.fixture_id );
                        ("status", J.string "verified fixture bytes and tensors");
                        ("consumer_acceptance", J.string "not measured");
                        ("manifest", bundle.manifest);
                      ]
                in
                let* fields = members result in
                Ok
                  (obj
                     (fields
                     @ [
                         ( "pins",
                           obj
                             [
                               ("index", pin bundle.index);
                               ("manifest", pin bundle.manifest_pin);
                               ("archive", pin bundle.archive_pin);
                               ("reference_publication", pin cohort.publication);
                             ] );
                       ]))
              in
              let result =
                match Err.payload result with
                | Ok result ->
                    if
                      (mode = "adapters" || mode = "models"
                     || mode = "generation")
                      && not (T.Acceptance.passed result)
                    then failed := true;
                    Fmt.pr "%s %s@." mode request.fixture_id;
                    result
                | Error error ->
                    failed := true;
                    let message = Fmt.str "%a" T.Fault.pp error in
                    Fmt.epr "%s@." message;
                    obj
                      [
                        ("schema_version", J.int 1);
                        ("fixture_id", J.string request.fixture_id);
                        ("status", J.string "failed");
                        ("error", J.string message);
                        ("consumer_acceptance", J.string "not measured");
                      ]
              in
              T.Runtime.add run request.fixture_id result)
            requests
        in
        let* final = context () in
        let+ () = T.Runtime.finish run final in
        if !failed then 1 else 0
    | _ ->
        invalid
          "usage: transformers_tasks \
           (fetch|check|diagnostics|adapters|models|generation) SELECTION \
           COHORT CACHE OUTPUT_DIR"
  in
  let result =
    try work () with
    | Sys_error e -> Err.fail (`File_io ("task fixture", e))
    | Unix.Unix_error (e, op, path) ->
        Err.fail (`File_io (path, op ^ ": " ^ Unix.error_message e))
  in
  match result with
  | Ok code -> exit code
  | Error e ->
      Fmt.epr "%a@." T.Fault.pp (Err.Error.kind e);
      exit 2
