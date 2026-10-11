(* The two-pass certification workflow. Pass 1 runs the shared frontier with
   the reference side as the uniform donor; only a complete agreement admits
   pass 2, which starts from fresh resources with the very same identities and
   commits the tested side's values. The tested engine's complete graph is then
   run on the same case and compared with pass 2's final outputs under the
   declared partition fidelity, the contract's required end-to-end checks are
   applied, and only then is an immutable golden record written. Every stage
   is persisted before the next starts, and a refusal, error or mismatch at any
   point ends the workflow with no golden. *)

open Inference_compare_schema
include Certify_base
module L = Codecs.Logical_value
module Core = Inference_compare_core
module Pol = Inference_compare_policy
module M = Inference_compare_metrics
open Certify_contract
open Certify_outputs

(* ---- running one pass -------------------------------------------------------- *)

let prepare_pass ~experiment_path ~donor ?(transform = fun l -> Ok l) stage =
  let map r = Result.map_error (load_error stage) r in
  let* loaded = map (Session.load experiment_path) in
  let* loaded =
    Result.map_error
      (fun m -> { stage; kind = Invalid; detail = m; run = None })
      (transform
         {
           loaded with
           experiment =
             { loaded.experiment with default_donor = donor; overrides = [] };
         })
  in
  let* v = map (Session.validate loaded) in
  map (Session.prepare v)

let failure_text (r : Core.Coordinator.result) =
  match r.failure with
  | None -> "no failure record"
  | Some f ->
      Printf.sprintf "%s in the %s phase"
        (W.canonical_string (Failure.encode_code f.code))
        (W.canonical_string (Phase.encode f.phase))

let pass_ok stage (o : Run.outcome) =
  let run = Some o.directory in
  let r = o.result in
  match o.finalized with
  | Error e ->
      fail ?run stage Failed
        (Format.asprintf "the run's evidence was not completed: %a"
           Evidence.pp_error e)
  | Ok () -> (
      match
        (r.completion.execution, r.completion.numerical, r.completion.coverage)
      with
      | Complete, Agreement, Complete -> Ok ()
      | Complete, Disagreement, _ ->
          fail ?run stage Disagreement "a local comparison disagrees"
      | Complete, _, _ ->
          fail ?run stage Incomplete "agreement could not be established"
      | (Failed | Refused), _, _ -> fail ?run stage Failed (failure_text r)
      | Interrupted, _, _ ->
          fail ?run stage Incomplete "the run was interrupted")

let run_evidence_digest ~algorithm stage (o : Run.outcome) =
  let* text =
    io stage (File_io.read (Filename.concat o.directory "completion.json"))
  in
  make_digest Digest.Run_evidence.make algorithm text stage

let exact_local (r : Core.Coordinator.result) =
  List.for_all
    (fun (c : Comparison_result.t) ->
      c.verdict = Pass
      && Int64.equal c.elements_checked c.elements_expected
      &&
      match c.element with
      | Some e -> Int64.equal e.bitwise_mismatches 0L
      | None -> false)
    r.comparisons

(* What a pass must not change for the next to count: the plan, both engines'
   identities and every input byte. *)
let pins_of (p : Session.prepared) =
  let l = p.validated.loaded in
  let plan = Core.Sealed_plan.plan p.sealed in
  ( Digest.Plan.hex plan.plan_digest,
    canonical_text Engine_identity.encode p.left_identity,
    canonical_text Engine_identity.encode p.right_identity,
    l.texts,
    List.map (fun (k, b) -> (Id.Canonical_value.to_string k, b)) l.case_texts )

(* ---- the workflow ------------------------------------------------------------- *)

let write_stage ~dir ~seq stage fields =
  let json =
    W.obj
      (W.field "stage" W.enc_string (stage_name stage)
      :: W.field "status" W.enc_string "passed"
      :: fields)
  in
  let* () = io stage (File_io.mkdir_p (Filename.concat dir "stages")) in
  io stage
    (File_io.write_exclusive
       (Filename.concat dir
          (Printf.sprintf "stages/%d-%s.json" seq (stage_name stage)))
       (Wire_text.to_string json ^ "\n"))

let drop_finals finals = List.iter (fun (_, v) -> L.drop v) finals

let selection_for ~(mapping : Mapping_artifact.t)
    ~(contract : Computation_contract.t) : Policy_selection.t =
  {
    default = None;
    explicit =
      List.filter_map
        (fun (o : Output_binding.t) ->
          List.find_map
            (fun (f : Final_output.t) ->
              if Id.Canonical_value.equal f.canonical o.key then
                List.find_map
                  (fun (occ, p) ->
                    if Id.Output_occurrence.equal occ f.occurrence then
                      Some (o.comparison_id, p)
                    else None)
                  contract.final_policies
              else None)
            mapping.frontier.final_outputs)
        mapping.frontier.output_bindings;
    rules = [];
  }

let whole_transform ~contract (l : Session.loaded) =
  match Whole_graph.of_mapping l.mapping with
  | Error m -> Error m
  | Ok mapping ->
      let selection = selection_for ~mapping ~contract in
      let text enc v = Wire_text.to_string (enc v) ^ "\n" in
      let swap k v =
        List.map (fun (n, t) -> if n = k then (n, v) else (n, t))
      in
      Ok
        {
          l with
          mapping;
          selection;
          texts =
            l.texts
            |> swap "mapping" (text Mapping_artifact.encode mapping)
            |> swap "selection" (text Policy_selection.encode selection);
        }

let default_id (l : Session.loaded) =
  Id.Experiment.to_string l.experiment.experiment_id

let run_id stage cert suffix =
  match Id.Run.of_string (cert ^ "-" ^ suffix) with
  | Ok r -> Ok r
  | Error _ -> invalid stage "the certification id is not a valid run name"

let read_frozen ~dir ~ctx ~(mapping : Mapping_artifact.t)
    (frozen : Workflow_file.frozen list) =
  List.fold_left
    (fun acc (f : Workflow_file.frozen) ->
      let* acc = acc in
      match
        List.find_opt
          (fun (o : Final_output.t) ->
            Id.Output_occurrence.equal o.occurrence f.occurrence)
          mapping.frontier.final_outputs
      with
      | None -> invalid Contract "a frozen output names no final output"
      | Some o -> (
          match
            List.find_opt
              (fun (c : Canonical_spec.t) ->
                Id.Canonical_value.equal c.key o.canonical)
              mapping.frontier.canonical_values
          with
          | None -> invalid Contract "a final output has no canonical spec"
          | Some c ->
              let* bytes =
                io Contract (File_io.read (File_io.resolve ~base:dir f.file))
              in
              let* v =
                Result.map_error
                  (fun m ->
                    { stage = Contract; kind = Invalid; detail = m; run = None })
                  (value ctx (c.spec, bytes))
              in
              Ok ((f.occurrence, v) :: acc)))
    (Ok []) frozen

let side_graph (l : Session.loaded) side =
  match (side : Side.t) with
  | Left -> l.left_graph.key
  | Right -> l.right_graph.key

let side_identity (p : Session.prepared) side =
  match (side : Side.t) with
  | Left -> p.left_identity
  | Right -> p.right_identity

let by_occurrence finals occ =
  List.find_map
    (fun (o, v) -> if Id.Output_occurrence.equal o occ then Some v else None)
    finals

let observed_value ctx table (o : Final_output.t) side =
  match Hashtbl.find_opt table (observed_key o.canonical side) with
  | None -> None
  | Some x -> Result.to_option (value ctx x)

let stage_fields ~(o : Run.outcome) ~run_digest (p : Session.prepared) =
  let plan = Core.Sealed_plan.plan p.sealed in
  [
    W.field "run" W.enc_string (Filename.basename o.directory);
    W.field "run_digest" Wire_digest.encode_run_evidence run_digest;
    W.field "plan_digest" Wire_digest.encode_plan plan.plan_digest;
    W.field "left_engine" Engine_identity.encode p.left_identity;
    W.field "right_engine" Engine_identity.encode p.right_identity;
  ]

let run ?(hooks = no_hooks) ?cert_id ~root ~workflow_path () :
    (certified, failure) result =
  let pending = ref [] in
  let hold finals = pending := finals :: !pending in
  let workflow () =
    let base = Filename.dirname workflow_path in
    let wmap stage r = Result.map_error (load_error stage) r in
    let* wf, _ =
      wmap Contract (Session.decode_file workflow_path Workflow_file.decode)
    in
    let w = wf.workflow in
    let experiment_path = File_io.resolve ~base wf.experiment in
    let* contract, _ =
      wmap Contract
        (Session.decode_file
           (File_io.resolve ~base wf.contract)
           Computation_contract.decode)
    in
    let* loaded = wmap Contract (Session.load experiment_path) in
    let algorithm = loaded.experiment.digest_algorithm in
    let* cdigest =
      make_digest Digest.Contract.make algorithm
        (Computation_contract.preimage contract)
        Contract
    in
    let* () =
      if Digest.Contract.equal cdigest w.contract_digest then Ok ()
      else
        invalid Contract "the workflow's contract digest is not the contract's"
    in
    let ctx = loaded.initial_ctx in
    let* frozen_values =
      read_frozen ~dir:base ~ctx ~mapping:loaded.mapping wf.frozen_outputs
    in
    let* () = check_contract ~wf ~contract ~loaded ~frozen:wf.frozen_outputs in
    let finals = loaded.mapping.frontier.final_outputs in
    let final_keys =
      List.map (fun (o : Final_output.t) -> o.canonical) finals
    in
    let cert = match cert_id with Some c -> c | None -> default_id loaded in
    let dir = Filename.concat root ("cert-" ^ cert) in
    let* () = io Contract (File_io.mkdir_p root) in
    let* () =
      match File_io.mkdir_exclusive dir with
      | Ok () -> Ok ()
      | Error e ->
          fail Contract Invalid
            (Format.asprintf "certification directory: %a" File_io.pp_error e)
    in
    (* pass 1: the reference trajectory *)
    hooks.before Reference_trajectory;
    let* p1 =
      prepare_pass ~experiment_path ~donor:w.reference_side Reference_trajectory
    in
    let* id1 = run_id Reference_trajectory cert "p1" in
    let* o1 =
      match Run.run ~root ~run_id:id1 ~keep_finals:true p1 with
      | Ok o -> Ok o
      | Error e ->
          fail Reference_trajectory Failed
            (Format.asprintf "%a" Evidence.pp_error e)
    in
    hold o1.result.finals;
    let* () = pass_ok Reference_trajectory o1 in
    let* d1 = run_evidence_digest ~algorithm Reference_trajectory o1 in
    let* () =
      write_stage ~dir ~seq:1 Reference_trajectory
        (stage_fields ~o:o1 ~run_digest:d1 p1)
    in
    (* pass 2: the tested trajectory, from fresh resources *)
    hooks.before Tested_trajectory;
    let* p2 =
      prepare_pass ~experiment_path ~donor:w.tested_side Tested_trajectory
    in
    let* () =
      if pins_of p1 = pins_of p2 then Ok ()
      else
        fail Identity_continuity Invalid
          "the plan, the engines or the inputs changed between the passes"
    in
    let trusted : observed = Hashtbl.create 8 in
    let* id2 = run_id Tested_trajectory cert "p2" in
    let* o2 =
      match
        Run.run ~root ~run_id:id2 ~keep_finals:true
          ~observe:(observer final_keys trusted)
          p2
      with
      | Ok o -> Ok o
      | Error e ->
          fail Tested_trajectory Failed
            (Format.asprintf "%a" Evidence.pp_error e)
    in
    hold o2.result.finals;
    let* () = pass_ok Tested_trajectory o2 in
    let* d2 = run_evidence_digest ~algorithm Tested_trajectory o2 in
    let* () =
      write_stage ~dir ~seq:2 Tested_trajectory
        (stage_fields ~o:o2 ~run_digest:d2 p2)
    in
    (* the tested engine's complete graph, on the same case *)
    hooks.before Full_graph_confirmation;
    let* p3 =
      prepare_pass ~experiment_path ~donor:w.tested_side
        ~transform:(whole_transform ~contract)
        Full_graph_confirmation
    in
    let engines_of p = pins_of p |> fun (_, l, r, t, c) -> (l, r, t, c) in
    let* () =
      let l1, r1, t1, c1 = engines_of p1 and l3, r3, t3, c3 = engines_of p3 in
      let keep =
        List.filter (fun (n, _) -> n <> "mapping" && n <> "selection")
      in
      if l1 = l3 && r1 = r3 && keep t1 = keep t3 && c1 = c3 then Ok ()
      else
        fail Identity_continuity Invalid
          "the engines or the inputs of the complete-graph run differ from the \
           passes'"
    in
    let reference_whole : observed = Hashtbl.create 8 in
    let* id3 = run_id Full_graph_confirmation cert "full" in
    let* o3 =
      match
        Run.run ~root ~run_id:id3 ~keep_finals:true
          ~observe:(observer final_keys reference_whole)
          p3
      with
      | Ok o -> Ok o
      | Error e ->
          fail Full_graph_confirmation Failed
            (Format.asprintf "%a" Evidence.pp_error e)
    in
    hold o3.result.finals;
    let* () =
      match pass_ok Full_graph_confirmation o3 with
      | Ok () -> Ok ()
      | Error f -> Error f
    in
    let how =
      match w.partition_fidelity with
      | Bitwise_required -> Bitwise
      | Declared_contract -> Policy
    in
    let* fidelity =
      compare_outputs Full_graph_confirmation ~ctx ~algorithm ~contract ~finals
        ~how
        ~left:(by_occurrence o2.result.finals)
        ~right:(by_occurrence o3.result.finals)
        ~what:"partition fidelity of the tested engine"
    in
    let* reference_fidelity =
      if w.end_to_end_reference_check then
        compare_outputs Full_graph_confirmation ~ctx ~algorithm ~contract
          ~finals ~how
          ~left:(by_occurrence o1.result.finals)
          ~right:(fun occ ->
            List.find_map
              (fun (o : Final_output.t) ->
                if Id.Output_occurrence.equal o.occurrence occ then
                  observed_value ctx reference_whole o w.reference_side
                else None)
              finals)
          ~what:"partition fidelity of the reference engine"
      else Ok { exact = true }
    in
    let* d3 = run_evidence_digest ~algorithm Full_graph_confirmation o3 in
    let* () =
      write_stage ~dir ~seq:3 Full_graph_confirmation
        (stage_fields ~o:o3 ~run_digest:d3 p3)
    in
    (* the contract's required end-to-end checks *)
    hooks.before Required_end_to_end;
    let* acceptance =
      if contract.required_full_graph_acceptance then
        compare_outputs Required_end_to_end ~ctx ~algorithm ~contract ~finals
          ~how:Policy
          ~left:(by_occurrence o3.result.finals)
          ~right:(by_occurrence frozen_values)
          ~what:"full-graph acceptance against the frozen reference"
      else Ok { exact = true }
    in
    let* independent =
      if w.end_to_end_reference_check then
        compare_outputs Required_end_to_end ~ctx ~algorithm ~contract ~finals
          ~how:Policy
          ~left:(by_occurrence o3.result.finals)
          ~right:(fun occ ->
            List.find_map
              (fun (o : Final_output.t) ->
                if Id.Output_occurrence.equal o.occurrence occ then
                  observed_value ctx reference_whole o w.reference_side
                else None)
              finals)
          ~what:"the tested output against the reference's complete graph"
      else Ok { exact = true }
    in
    let* () = write_stage ~dir ~seq:4 Required_end_to_end [] in
    (* golden materialization *)
    hooks.before Golden;
    let tested_outputs =
      List.map
        (fun (o : Final_output.t) ->
          (o, by_occurrence o3.result.finals o.occurrence))
        finals
    in
    let* payloads =
      List.fold_left
        (fun acc (o, v) ->
          let* acc = acc in
          match v with
          | Some v -> Ok (acc @ [ (o, v) ])
          | None -> fail Golden Failed "a tested output is missing")
        (Ok []) tested_outputs
    in
    let gdir = Filename.concat dir "golden" in
    let* () = io Golden (File_io.mkdir_exclusive gdir) in
    let* files =
      List.fold_left
        (fun acc (i, ((o : Final_output.t), v)) ->
          let* acc = acc in
          let name = Printf.sprintf "output-%d.bin" i in
          let* () =
            io Golden
              (File_io.write_exclusive
                 (Filename.concat gdir name)
                 (L.contents v))
          in
          let* d =
            make_digest Digest.Payload.make algorithm (L.contents v) Golden
          in
          Ok (acc @ [ (o, name, d) ]))
        (Ok [])
        (List.mapi (fun i x -> (i, x)) payloads)
    in
    let* outputs_specs_digest =
      make_digest Digest.Output_specs.make algorithm
        (canonical_text
           (W.enc_list Tensor_spec.encode)
           (List.map (fun (_, v) -> L.spec v) payloads))
        Golden
    in
    let exact =
      exact_local o1.result && exact_local o2.result && fidelity.exact
      && reference_fidelity.exact && acceptance.exact && independent.exact
    in
    let qualification : Golden_record.qualification =
      if exact then Exact else Within_declared_contract
    in
    let record : Golden_record.t =
      {
        case_id = loaded.experiment.case_id;
        ordered_output_payload_digest = List.map (fun (_, _, d) -> d) files;
        output_specs_digest = outputs_specs_digest;
        tested_graph = side_graph loaded w.tested_side;
        tested_engine = side_identity p2 w.tested_side;
        reference_engine = side_identity p1 w.reference_side;
        first_pass_run_digest = d1;
        second_pass_run_digest = d2;
        full_graph_confirmation_digest = d3;
        contract_digest = cdigest;
        qualification;
      }
    in
    let* () =
      io Golden
        (File_io.write_exclusive
           (Filename.concat gdir "outputs.json")
           (Wire_text.to_string
              (W.enc_list
                 (fun ((o : Final_output.t), name, d) ->
                   W.obj
                     [
                       W.field "occurrence"
                         (fun x ->
                           W.enc_string (Id.Output_occurrence.to_string x))
                         o.occurrence;
                       W.field "file" W.enc_string name;
                       W.field "digest" Wire_digest.encode_payload d;
                     ])
                 files)
           ^ "\n"))
    in
    let* () =
      io Golden
        (File_io.write_exclusive
           (Filename.concat gdir "golden.json")
           (Wire_text.to_string (Golden_record.encode record) ^ "\n"))
    in
    Ok { directory = dir; record }
  in
  let finish r =
    List.iter drop_finals !pending;
    r
  in
  match workflow () with
  | r -> finish r
  | exception e ->
      ignore (finish (Ok ()));
      raise e
