(* Wire codecs for every digest domain. Kept apart from [Digest] so that
   module stays free of the wire layer. *)

module W = Wire

let make (type t) (of_string : string -> (t, Digest_error.t) Err.t)
    (to_string : t -> string) =
  let decode : t W.decoder =
   fun j ->
    let ( let* ) = W.( let* ) in
    let* s = W.string j in
    Err.map_error
      (fun e -> { W.path = []; reason = W.Invalid_digest e })
      (of_string s)
  in
  (decode, fun v -> W.enc_string (to_string v))

let build, encode_build = make Digest.Build.of_string Digest.Build.to_string

let configuration, encode_configuration =
  make Digest.Configuration.of_string Digest.Configuration.to_string

let comparison_policy, encode_comparison_policy =
  make Digest.Comparison_policy.of_string Digest.Comparison_policy.to_string

let contract, encode_contract =
  make Digest.Contract.of_string Digest.Contract.to_string

let conversion, encode_conversion =
  make Digest.Conversion.of_string Digest.Conversion.to_string

let derivation, encode_derivation =
  make Digest.Derivation.of_string Digest.Derivation.to_string

let dtype_parameters, encode_dtype_parameters =
  make Digest.Dtype_parameters.of_string Digest.Dtype_parameters.to_string

let engine_artifact, encode_engine_artifact =
  make Digest.Engine_artifact.of_string Digest.Engine_artifact.to_string

let environment, encode_environment =
  make Digest.Environment.of_string Digest.Environment.to_string

let extracted_graph_payload, encode_extracted_graph_payload =
  make Digest.Extracted_graph_payload.of_string
    Digest.Extracted_graph_payload.to_string

let frontier_mapping, encode_frontier_mapping =
  make Digest.Frontier_mapping.of_string Digest.Frontier_mapping.to_string

let graph, encode_graph = make Digest.Graph.of_string Digest.Graph.to_string

let implementation, encode_implementation =
  make Digest.Implementation.of_string Digest.Implementation.to_string

let input_domain, encode_input_domain =
  make Digest.Input_domain.of_string Digest.Input_domain.to_string

let logical_value, encode_logical_value =
  make Digest.Logical_value.of_string Digest.Logical_value.to_string

let mapping, encode_mapping =
  make Digest.Mapping.of_string Digest.Mapping.to_string

let metric_support, encode_metric_support =
  make Digest.Metric_support.of_string Digest.Metric_support.to_string

let output_specs, encode_output_specs =
  make Digest.Output_specs.of_string Digest.Output_specs.to_string

let payload, encode_payload =
  make Digest.Payload.of_string Digest.Payload.to_string

let plan, encode_plan = make Digest.Plan.of_string Digest.Plan.to_string

let policy_bindings, encode_policy_bindings =
  make Digest.Policy_bindings.of_string Digest.Policy_bindings.to_string

let policy_selection, encode_policy_selection =
  make Digest.Policy_selection.of_string Digest.Policy_selection.to_string

let run_evidence, encode_run_evidence =
  make Digest.Run_evidence.of_string Digest.Run_evidence.to_string
