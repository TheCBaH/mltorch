(* The dedicated inference comparison command: validate, plan, run, replay.
   Exit codes: 0 complete agreement; 1 completed numerical disagreement; 2
   invalid input, map or policy; 3 refused or failed preparation, transfer,
   execution or resource limit; 4 incomplete coverage, an agreement that could
   not be established, or an interrupted run. *)

open Cmdliner
module H = Inference_compare_host
module S = Inference_compare_schema

let experiment_arg =
  Arg.(
    required
    & opt (some file) None
    & info [ "experiment" ] ~docv:"FILE" ~doc:"The experiment manifest.")

let evidence_root_arg =
  Arg.(
    required
    & opt (some string) None
    & info [ "evidence-root" ] ~docv:"DIR"
        ~doc:"Directory receiving run directories.")

let run_id_arg =
  Arg.(
    value
    & opt (some string) None
    & info [ "run-id" ] ~docv:"ID"
        ~doc:"Run identifier (default: derived from the experiment and plan).")

let fail_load e =
  Format.eprintf "error: %a@." H.Session.pp_error e;
  2

let validate_cmd =
  let run experiment =
    match H.Session.load experiment with
    | Error e -> fail_load e
    | Ok loaded -> (
        match H.Session.validate loaded with
        | Error e -> fail_load e
        | Ok v ->
            Format.printf
              "valid: %d unit(s), %d comparison site(s), coverage %s@."
              (List.length
                 (Inference_compare_plan.Unit_order.order
                    (Inference_compare_plan.Structural.order v.structural)))
              (List.length v.sites)
              (match
                 Inference_compare_plan.Structural.coverage v.structural
               with
              | Complete -> "complete"
              | Partial _ -> "partial");
            0)
  in
  Cmd.v
    (Cmd.info "validate"
       ~doc:
         "Validate the mapping, recipes and policies without running inference.")
    Term.(const run $ experiment_arg)

let plan_cmd =
  let out_arg =
    Arg.(
      value
      & opt (some string) None
      & info [ "out" ] ~docv:"FILE" ~doc:"Write the sealed plan here.")
  in
  let run experiment out =
    match H.Session.load experiment with
    | Error e -> fail_load e
    | Ok loaded -> (
        match H.Session.validate loaded with
        | Error e -> fail_load e
        | Ok v -> (
            match H.Session.prepare v with
            | Error e -> fail_load e
            | Ok p ->
                let plan = Inference_compare_core.Sealed_plan.plan p.sealed in
                let text =
                  S.Wire_text.to_string (S.Execution_plan.encode plan) ^ "\n"
                in
                (match out with
                | Some f -> (
                    match H.File_io.write_atomic f text with
                    | Ok () ->
                        Format.printf "plan %s written to %s@."
                          (S.Digest.Plan.to_string plan.plan_digest)
                          f
                    | Error e ->
                        Format.eprintf "error: %a@." H.File_io.pp_error e)
                | None -> print_string text);
                0))
  in
  Cmd.v
    (Cmd.info "plan"
       ~doc:"Seal and print the execution plan; no engine is prepared or run.")
    Term.(const run $ experiment_arg $ out_arg)

let run_cmd =
  let run experiment root run_id =
    match H.Session.load experiment with
    | Error e -> fail_load e
    | Ok loaded -> (
        match H.Session.validate loaded with
        | Error e -> fail_load e
        | Ok v -> (
            match H.Session.prepare v with
            | Error e -> fail_load e
            | Ok p -> (
                let run_id =
                  Option.bind run_id (fun s ->
                      Result.to_option (S.Id.Run.of_string s))
                in
                match H.Run.run ~root ?run_id p with
                | Error (H.Evidence.Duplicate_run d) ->
                    Format.eprintf
                      "error: run directory %s already exists (use --run-id)@."
                      d;
                    2
                | Error e ->
                    Format.eprintf "error: %a@." H.Evidence.pp_error e;
                    3
                | Ok o -> (
                    let r = o.result in
                    Format.printf "run %s@." o.directory;
                    Format.printf "execution=%s numerical=%s units=%d@."
                      (match r.completion.execution with
                      | Complete -> "complete"
                      | Failed -> "failed"
                      | Refused -> "refused"
                      | Interrupted -> "interrupted")
                      (match r.completion.numerical with
                      | Agreement -> "agreement"
                      | Disagreement -> "disagreement"
                      | Not_established -> "not_established")
                      (List.length r.executed);
                    (match r.failure with
                    | Some f ->
                        Format.printf "failure: %a@." Fmt.string
                          (match S.Failure.encode_code f.code with
                          | Jsont.String (s, _) -> s
                          | _ -> "?")
                    | None -> ());
                    match o.finalized with
                    | Ok () -> H.Run.exit_code r
                    | Error e ->
                        Format.eprintf "error: evidence not finalized: %a@."
                          H.Evidence.pp_error e;
                        4))))
  in
  Cmd.v
    (Cmd.info "run" ~doc:"Run the experiment and write immutable evidence.")
    Term.(const run $ experiment_arg $ evidence_root_arg $ run_id_arg)

let replay_cmd =
  let run_dir =
    Arg.(
      required
      & opt (some dir) None
      & info [ "run" ] ~docv:"DIR" ~doc:"A finished run directory.")
  in
  let unit_arg =
    Arg.(
      value
      & opt (some string) None
      & info [ "unit" ] ~docv:"UNIT"
          ~doc:
            "Re-run this unit from its retained inputs (default: the whole \
             experiment).")
  in
  let case_arg =
    Arg.(
      value
      & opt (some string) None
      & info [ "case" ] ~docv:"CASE" ~doc:"The case the run recorded.")
  in
  let run run_dir unit case =
    ignore case;
    let report =
      match unit with
      | None -> Some (H.Replay.full ~run_dir)
      | Some u -> (
          match S.Id.Unit.of_string u with
          | Ok u -> Some (H.Replay.unit ~run_dir u)
          | Error _ ->
              Format.eprintf "error: %S is not a valid unit identifier@." u;
              None)
    in
    match report with
    | None -> 2
    | Some report -> (
        match report with
        | Error e -> (
            Format.eprintf "error: %a@." H.Replay.pp_error e;
            match e with
            | H.Replay.Evidence _ -> 4
            | H.Replay.Missing_payload _ -> 4
            | _ -> 2)
        | Ok r ->
            Format.printf "replay %s: %s@."
              (match r.scope with
              | `Full -> "full"
              | `Unit u -> "unit " ^ S.Id.Unit.to_string u)
              (if r.identical then "identical to the record"
               else "DIFFERS from the record");
            List.iter (Format.printf "  %s@.") r.differences;
            List.iter (Format.printf "note: %s@.") r.notes;
            if r.identical then 0 else 1)
  in
  Cmd.v
    (Cmd.info "replay"
       ~doc:
         "Re-execute a recorded run, or one unit of it, and compare with the \
          record.")
    Term.(const run $ run_dir $ unit_arg $ case_arg)

let () =
  let info =
    Cmd.info "inference_compare"
      ~doc:"Shared-frontier comparison of two inference engines."
  in
  exit
    (Cmd.eval' (Cmd.group info [ validate_cmd; plan_cmd; run_cmd; replay_cmd ]))
