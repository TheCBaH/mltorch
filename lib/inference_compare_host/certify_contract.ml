(* What a certification requires of its contract, workflow and mapping before
   anything runs. *)

open Inference_compare_schema
module W = Wire
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module Core = Inference_compare_core
module Pol = Inference_compare_policy
module M = Inference_compare_metrics
open Certify_base

let conversions_of (m : Mapping_artifact.t) =
  let f = m.frontier in
  let recipes =
    List.map (fun (b : Input_binding.t) -> b.from_common) f.input_bindings
    @ List.map
        (fun (c : Input_check.t) -> c.from_imported_inputs)
        f.input_checks
    @ List.concat_map
        (fun (o : Output_binding.t) ->
          [ o.from_left_outputs; o.from_right_outputs ])
        f.output_bindings
  in
  List.concat_map
    (fun (r : Recipe.t) ->
      List.filter_map
        (fun (n : Recipe.node) ->
          match n.operation with
          | Conversion_step.Codec_operation { implementation_digest; _ } ->
              Some implementation_digest
          | _ -> None)
        r.nodes)
    recipes

let lossy_comparison (m : Mapping_artifact.t) =
  List.exists
    (fun (o : Output_binding.t) ->
      (not o.from_left_outputs.lossless) || not o.from_right_outputs.lossless)
    m.frontier.output_bindings

let same_selection a b =
  String.equal
    (canonical_text Policy_selection.encode (Policy_selection.canonicalize a))
    (canonical_text Policy_selection.encode (Policy_selection.canonicalize b))

let check_contract ~(wf : Workflow_file.t) ~(contract : Computation_contract.t)
    ~(loaded : Session.loaded) ~frozen =
  let w = wf.workflow and e = loaded.experiment in
  let stage = Contract in
  let* () =
    if w.reference_side = w.tested_side then
      invalid stage "the reference and the tested side must be different"
    else Ok ()
  in
  let* () =
    match (contract.reference, w.reference_side) with
    | Left_reference, Left | Right_reference, Right -> Ok ()
    | (Left_reference | Right_reference), _ ->
        invalid stage
          "the contract's reference is not the workflow's reference side"
    | Peers, _ ->
        invalid stage
          "a peer-only comparison cannot certify correctness: there is no \
           trusted side"
    | Oracle _, _ ->
        invalid stage
          "an external oracle reference is not executable by this workflow"
  in
  let* () =
    if e.overrides <> [] then
      invalid stage
        "donor overrides are diagnostic and cannot be part of a certification"
    else Ok ()
  in
  let* () =
    match w.cases with
    | [ c ] when Id.Case.equal c e.case_id -> Ok ()
    | _ ->
        invalid stage
          "the workflow's cases must be exactly the experiment's case"
  in
  let* () =
    if same_selection contract.local_policy_selection loaded.selection then
      Ok ()
    else
      invalid stage
        "the experiment's policy selection is not the contract's local \
         selection"
  in
  let* () =
    if lossy_comparison loaded.mapping then
      invalid stage
        "a lossy comparison conversion cannot certify the unconverted outputs"
    else Ok ()
  in
  let* () =
    match
      List.find_opt
        (fun d ->
          not
            (List.exists
               (fun a ->
                 String.equal
                   (Digest.Conversion.to_string a)
                   (Digest.Implementation.to_string d))
               contract.allowed_conversion_digest))
        (conversions_of loaded.mapping)
    with
    | None -> Ok ()
    | Some _ ->
        invalid stage
          "the mapping uses a conversion the contract does not allow"
  in
  let* () =
    match
      List.find_opt
        (fun (o : Final_output.t) ->
          not
            (List.exists
               (fun (occ, _) -> Id.Output_occurrence.equal occ o.occurrence)
               contract.final_policies))
        loaded.mapping.frontier.final_outputs
    with
    | None -> Ok ()
    | Some o ->
        invalid stage
          (Printf.sprintf "final output %s has no frozen final policy"
             (Id.Output_occurrence.to_string o.occurrence))
  in
  if
    contract.required_full_graph_acceptance
    && List.exists
         (fun (o : Final_output.t) ->
           not
             (List.exists
                (fun (f : Workflow_file.frozen) ->
                  Id.Output_occurrence.equal f.occurrence o.occurrence)
                frozen))
         loaded.mapping.frontier.final_outputs
  then
    invalid stage
      "the contract requires a full-graph acceptance and the workflow freezes \
       no reference output for some final"
  else Ok ()
