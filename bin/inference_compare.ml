(* The dedicated inference comparison command: certify, export-pt2, plan,
   replay, run, validate.
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

let certify_cmd =
  let workflow =
    Arg.(
      required
      & opt (some file) None
      & info [ "workflow" ] ~docv:"FILE"
          ~doc:"The validation workflow (experiment, contract, fidelity).")
  in
  let cert_id =
    Arg.(
      value
      & opt (some string) None
      & info [ "id" ] ~docv:"ID"
          ~doc:"Certification identifier (default: the experiment's).")
  in
  let run workflow_path root cert_id =
    match H.Certify.run ~root ?cert_id ~workflow_path () with
    | Ok c ->
        Format.printf "certified %s (%s)@." c.directory
          (match c.record.qualification with
          | Exact -> "exact"
          | Within_declared_contract -> "within the declared contract");
        0
    | Error f ->
        Format.eprintf "not certified: %a@." H.Certify.pp_failure f;
        H.Certify.exit_code f
  in
  Cmd.v
    (Cmd.info "certify"
       ~doc:
         "Run the two-pass certification: reference trajectory, tested \
          trajectory, the tested engine's complete graph and the contract's \
          end-to-end checks; write an immutable golden record only if every \
          stage passes.")
    Term.(const run $ workflow $ evidence_root_arg $ cert_id)

let export_pt2_cmd =
  let model =
    Arg.(
      required
      & opt (some file) None
      & info [ "model" ] ~docv:"FILE" ~doc:"The exported .pt2 model.")
  in
  let inputs =
    Arg.(
      required
      & opt (some file) None
      & info [ "inputs" ] ~docv:"FILE" ~doc:"The model's input map (inputs.pt).")
  in
  let index =
    Arg.(
      value & opt int 0
      & info [ "sample" ] ~docv:"N" ~doc:"Which sample of the input map.")
  in
  let tolerance =
    Arg.(
      value & opt float 1e-4
      & info [ "tolerance" ] ~docv:"T"
          ~doc:"Absolute and relative tolerance of every binary32 site.")
  in
  let out =
    Arg.(
      required
      & opt (some string) None
      & info [ "out" ] ~docv:"DIR" ~doc:"Directory receiving experiment.json.")
  in
  let right =
    Arg.(
      value
      & opt
          (enum
             [
               ("native", Inference_compare_pt2_export.Pt2_export.Native);
               ("native4d", Inference_compare_pt2_export.Pt2_export.Native4d);
             ])
          Inference_compare_pt2_export.Pt2_export.Native
      & info [ "right" ] ~docv:"ENGINE"
          ~doc:"The engine compared with ATen: native or native4d.")
  in
  let certify =
    Arg.(
      value
      & opt
          (some
             (enum
                [
                  ("exact", S.Partition_fidelity.Bitwise_required);
                  ("declared", S.Partition_fidelity.Declared_contract);
                ]))
          None
      & info [ "certify" ] ~docv:"FIDELITY"
          ~doc:
            "Also write a contract and validation workflow for the two-pass \
             certification (ATen is the reference), requiring the tested \
             engine's segmented run to match its complete graph bitwise \
             (exact) or within the declared policy (declared).")
  in
  let tested =
    Arg.(
      value
      & opt
          (enum [ ("left", S.Side.Left); ("right", S.Side.Right) ])
          S.Side.Right
      & info [ "tested" ] ~docv:"SIDE"
          ~doc:"With --certify: which side is under test (default right).")
  in
  let run model inputs index tolerance out right certify tested =
    let certify =
      Option.map
        (fun fidelity ->
          { Inference_compare_pt2_export.Pt2_export.tested; fidelity })
        certify
    in
    match
      Inference_compare_pt2_export.Pt2_export.write_experiment ~right ?certify
        ~dir:out ~model ~inputs ~index ~tolerance ()
    with
    | Ok () ->
        Format.printf "wrote %s@." (Filename.concat out "experiment.json");
        0
    | Error m ->
        Format.eprintf "error: %s@." m;
        2
  in
  Cmd.v
    (Cmd.info "export-pt2"
       ~doc:
         "Write an experiment comparing the ATen interpreter with the Native \
          or Native4D evaluator on one exported model and sample.")
    Term.(
      const run $ model $ inputs $ index $ tolerance $ out $ right $ certify
      $ tested)

let () =
  let info =
    Cmd.info "inference_compare"
      ~doc:"Shared-frontier comparison of two inference engines."
  in
  exit
    (Cmd.eval'
       (Cmd.group info
          [
            certify_cmd;
            export_pt2_cmd;
            plan_cmd;
            replay_cmd;
            run_cmd;
            validate_cmd;
          ]))
