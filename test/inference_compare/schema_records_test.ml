(* The remaining schema records: policies, selectors, contracts, results and
   evidence, plus the canonical ordering of digestable records. *)

open Inference_compare_schema
module F = Schema_fixtures

let roundtrip name decode encode v =
  let text = Wire_text.to_string (encode v) in
  match Wire_text.decode_string decode text with
  | Error e ->
      Format.printf "%s: DECODE ERROR %a@." name
        (Err.Error.pp_kind Wire.pp_error)
        e
  | Ok v' ->
      Format.printf "%s: %s@." name
        (if String.equal text (Wire_text.to_string (encode v')) then "roundtrip"
         else "DIFFERS")

let f32_policy : Float_policy.t =
  {
    atol = 1e-5;
    rtol = 1e-4;
    scale = Symmetric_max_scale;
    arithmetic = Binary32;
    nan = Reject;
    infinity = Same_sign;
    signed_zero = Equal;
    required = Tolerance;
  }

let statistics_policy : Difference_statistics.policy =
  { direction = Right_minus_left; variance = Population }

let metric id kind partition role : Tensor_metric.t =
  { id = F.text (Id.Tensor_metric.of_string id); kind; partition; role }

let axis i = Option.get (Tensor_axis.of_index i)

let policy : Comparison_policy.t =
  {
    element = Some (Floating f32_policy);
    statistics = statistics_policy;
    relative_scale = Symmetric_max_scale;
    tensor_metrics =
      [
        metric "rmse" Root_mean_square_error Whole_tensor
          (Required (At_most 1e-3));
        metric "cos"
          (Cosine_similarity { zero_norm = Undefined })
          (Reduce_axes [ axis 2 ])
          (Required (At_least 0.999));
        metric "w1"
          (Wasserstein_1d
             (Histogram_masses
                {
                  support =
                    {
                      id = F.text (Id.Metric_support.of_string "bins");
                      payload_digest =
                        F.digest
                          (Digest.Metric_support.make F.algorithm
                             ~hex:(F.sha 'e'));
                      positions = 8L;
                    };
                  mass = Require_unit_mass { absolute_tolerance = 1e-6 };
                }))
          Whole_tensor Diagnostic_only;
      ];
  }

let selector : Metric_selector.t =
  let g = F.left_graph in
  let consumer : Tensor_selector.Consumer.t =
    {
      node = { node = None; operator = Some (F.operator "aten" "add") };
      port = Input_port.of_index 0;
    }
  in
  All_of
    [
      On_side
        {
          side = Left;
          anchors = Every_anchor;
          tensor =
            All_of
              [
                Produced_by
                  {
                    scope = Within_cluster;
                    producer =
                      {
                        node =
                          { node = Some (F.node g "n_mul"); operator = None };
                        port = Output_port.of_index 0;
                      };
                  };
                Not (Boundary Graph_output);
                Flows_to
                  {
                    scope = Endpoint_graph;
                    through =
                      [
                        {
                          node =
                            {
                              node = None;
                              operator = Some (F.operator "aten" "reshape");
                            };
                          input_port = None;
                          output_port = None;
                        };
                      ];
                    sink = consumer;
                  };
                Fan_out
                  {
                    scope = Endpoint_graph;
                    count = { minimum = 2L; maximum = None };
                  };
              ];
        };
      Not (Unit (F.unit_id "u9"));
    ]

let rule : Metric_rule.t =
  {
    id = F.text (Id.Metric_rule.of_string "residual_join");
    description =
      F.text (Prose.Description.of_string "matmul into residual add");
    selector;
    priority = Rule_priority.of_int 10;
    action = Add_metrics policy.tensor_metrics;
    empty_match = Reject;
  }

let selection : Policy_selection.t =
  {
    default = Some policy;
    explicit = [ (F.comparison "cmp_y", policy) ];
    rules = [ rule ];
  }

let contract : Computation_contract.t =
  {
    reference = Left_reference;
    input_domain =
      {
        description = F.text (Prose.Description.of_string "unit normal");
        specification_digest =
          F.digest (Digest.Input_domain.make F.algorithm ~hex:(F.sha 'f'));
      };
    local_policy_selection = selection;
    final_policies =
      [ (F.text (Id.Output_occurrence.of_string "out0"), policy) ];
    allowed_conversion_digest =
      [ F.digest (Digest.Conversion.make F.algorithm ~hex:(F.sha '1')) ];
    required_full_graph_acceptance = true;
  }

let%expect_test "policy, selector and contract roundtrip" =
  roundtrip "policy" Comparison_policy.decode Comparison_policy.encode policy;
  roundtrip "selector" Metric_selector.decode Metric_selector.encode selector;
  roundtrip "rule" Metric_rule.decode Metric_rule.encode rule;
  roundtrip "selection" Policy_selection.decode Policy_selection.encode
    selection;
  roundtrip "contract" Computation_contract.decode Computation_contract.encode
    contract;
  [%expect
    {|
    policy: roundtrip
    selector: roundtrip
    rule: roundtrip
    selection: roundtrip
    contract: roundtrip |}]

let comparison_result : Comparison_result.t =
  let scalar_none : Difference_statistics.scalar =
    Unavailable Empty_finite_set
  in
  let magnitudes : Magnitude_summary.t =
    {
      finite_values = 5L;
      zero_over_zero = 1L;
      unbounded = 1L;
      nonfinite_operands = 2L;
      arithmetic_overflow = 0L;
      minimum_finite = Some 0.0;
      maximum_finite = Some 0.5;
      largest_finite_index = Some [ 1L; 2L ];
      mean_finite = Finite_value 0.125;
      standard_deviation_finite = scalar_none;
    }
  in
  {
    comparison_id = F.comparison "cmp_y";
    policy_digest =
      F.digest (Digest.Comparison_policy.make F.algorithm ~hex:(F.sha '2'));
    elements_expected = 9L;
    elements_checked = 9L;
    element =
      Some
        {
          statistics =
            {
              status = Computed;
              finite_pairs = 7L;
              excluded_nonfinite_pairs = 2L;
              positive_errors = 3L;
              negative_errors = 2L;
              zero_errors = 2L;
              minimum_signed_error =
                Some { index = [ 0L ]; value = Finite_value (-0.5) };
              maximum_signed_error = None;
              largest_absolute_error = None;
              bias = Finite_value 0.0;
              standard_deviation = scalar_none;
              mean_absolute_error = Finite_value 0.25;
              root_mean_square_error = Unavailable Numeric_overflow;
            };
          bitwise_mismatches = 4L;
          tolerance_violations = Some 1L;
          first_bitwise_index = Some [ 0L; 1L ];
          first_violation_index = None;
          absolute = magnitudes;
          relative = magnitudes;
          nonfinite = { Nonfinite_summary.zero with both_nan_same_bits = 1L };
          difference_samples =
            [
              { index = [ 0L ]; absolute = Finite 0.5; relative = Unbounded };
              {
                index = [ 1L ];
                absolute = Unavailable Nonfinite_operands;
                relative = Unavailable Zero_over_zero;
              };
            ];
          max_ulp_distance =
            Option.map Ulp_distance.of_unsigned_bits (Some (-1L));
          finite_pairs = 7L;
          nan_pairs = 1L;
          infinity_pairs = 1L;
        };
    tensor_metrics =
      [
        {
          metric = F.text (Id.Tensor_metric.of_string "rmse");
          status = Nonfinite;
          elements_used = 9L;
          groups_expected = 1L;
          groups_checked = 0L;
          minimum = No_finite_value;
          maximum = Overflow;
          violations = None;
          first_violation_group = None;
          samples = [ ([ 0L ], 0.25) ];
        };
      ];
    verdict = Metric_unavailable;
  }

let%expect_test "exceptional measurements stay tagged" =
  roundtrip "comparison_result" Comparison_result.decode
    Comparison_result.encode comparison_result;
  (match comparison_result.element with
  | Some { max_ulp_distance = Some u; _ } ->
      Format.printf "max ulp: %s@." (Ulp_distance.to_decimal_string u)
  | _ -> ());
  let text = Wire_text.to_string (Comparison_result.encode comparison_result) in
  let has sub =
    let n = String.length sub in
    let rec go i =
      i + n <= String.length text && (String.sub text i n = sub || go (i + 1))
    in
    go 0
  in
  Format.printf "NaN literal: %b, Infinity literal: %b, unbounded tag: %b@."
    (has "NaN") (has "Infinity")
    (has {|"tag":"unbounded"|});
  [%expect
    {|
    comparison_result: roundtrip
    max ulp: 18446744073709551615
    NaN literal: false, Infinity literal: false, unbounded tag: true |}]

let%expect_test "evidence records roundtrip" =
  let instance s = F.text (Id.Value_instance.of_string s) in
  let event : Event.t =
    {
      sequence = 7L;
      run_id = F.text (Id.Run.of_string "run0");
      case_id = F.case_id "case0";
      experiment_id = F.text (Id.Experiment.of_string "follow_left");
      unit = Some (F.unit_id "u1");
      payload =
        Frontier_committed
          [
            {
              key = F.canonical "h";
              donor = Left;
              selected_instance = instance "v1";
              converted_instance = instance "v2";
              recipe = F.text (Id.Recipe.of_string "r_h");
            };
          ];
    }
  in
  let failure : Failure.t =
    {
      code = Lossy_common_input;
      phase = Planning;
      side = Some Right;
      unit = Some (F.unit_id "u1");
      node = None;
      tensor = Some (F.tensor F.right_graph "x2");
      rule = None;
      comparison = None;
      expected = Some (F.text (Prose.Diagnostic_text.of_string "f32"));
      actual = Some (F.text (Prose.Diagnostic_text.of_string "f16"));
    }
  in
  roundtrip "event" Event.decode Event.encode event;
  roundtrip "failed event" Event.decode Event.encode
    { event with payload = Failed failure };
  roundtrip "completion" Completion.decode Completion.encode
    {
      execution = Complete;
      coverage = Partial [ F.text (Prose.Description.of_string "one site") ];
      numerical = Not_established;
      evidence = Shared_frontier;
    };
  [%expect
    {|
    event: roundtrip
    failed event: roundtrip
    completion: roundtrip |}]

let%expect_test "workflow and golden roundtrip" =
  let engine : Engine_identity.t =
    {
      engine = F.text (Id.Engine.of_string "native");
      route = F.text (Id.Engine_route.of_string "eval_direct");
      build_digest = F.digest (Digest.Build.make F.algorithm ~hex:(F.sha '3'));
      configuration_digest =
        F.digest (Digest.Configuration.make F.algorithm ~hex:(F.sha '4'));
      artifact_digest = None;
      environment_digest =
        F.digest (Digest.Environment.make F.algorithm ~hex:(F.sha '5'));
    }
  in
  let contract_digest =
    F.digest (Digest.Contract.make F.algorithm ~hex:(F.sha '6'))
  in
  let run_digest ch =
    F.digest (Digest.Run_evidence.make F.algorithm ~hex:(F.sha ch))
  in
  roundtrip "workflow" Validation_workflow.decode Validation_workflow.encode
    {
      reference_side = Left;
      tested_side = Right;
      contract_digest;
      cases = [ F.case_id "case0" ];
      partition_fidelity = Bitwise_required;
      end_to_end_reference_check = true;
    };
  roundtrip "golden" Golden_record.decode Golden_record.encode
    {
      case_id = F.case_id "case0";
      ordered_output_payload_digest = [ F.payload_digest '7' ];
      output_specs_digest =
        F.digest (Digest.Output_specs.make F.algorithm ~hex:(F.sha '8'));
      tested_graph = F.right_graph;
      tested_engine = engine;
      reference_engine = engine;
      first_pass_run_digest = run_digest '9';
      second_pass_run_digest = run_digest 'a';
      full_graph_confirmation_digest = run_digest 'b';
      contract_digest;
      qualification = Within_declared_contract;
    };
  [%expect {|
    workflow: roundtrip
    golden: roundtrip |}]

let%expect_test "capabilities and limits roundtrip" =
  roundtrip "codec capability" Capability.decode_codec Capability.encode_codec
    {
      codec = F.text (Id.Codec.of_string "f32");
      element = Float_binary32;
      byte_width = 4;
      element_inspection = Supported;
      exact_values = Supported;
    };
  roundtrip "limits" Resource_limits.decode Resource_limits.encode
    {
      max_live_bytes = 1_073_741_824L;
      max_conversion_scratch_bytes = 4096L;
      max_snapshot_bytes = 1_073_741_824L;
      max_retained_disk_bytes = 0L;
      max_mismatch_samples = 16L;
      retention = First_mismatch;
      retention_requirement = Best_effort;
      verified_spill = false;
      max_nodes = 100_000L;
      max_edges = 400_000L;
      max_units = 10_000L;
      max_recipe_nodes = 100_000L;
      max_selector_nodes = 1_000L;
      max_selector_depth = 32L;
      max_path_length = 8L;
      max_retained_witnesses = 1_000L;
    };
  [%expect {|
    codec capability: roundtrip
    limits: roundtrip |}]

let%expect_test "canonical ordering of digestable records" =
  let m = F.mapping in
  let shuffled =
    {
      m with
      Mapping_artifact.units = List.rev m.units;
      tensor_relations = List.rev m.tensor_relations;
      frontier =
        { m.frontier with input_bindings = List.rev m.frontier.input_bindings };
    }
  in
  let same = Mapping_artifact.preimage m = Mapping_artifact.preimage shuffled in
  Format.printf "set order is not part of the preimage: %b@." same;
  (* A cluster's output list is occurrence order, so swapping it must change
     the preimage. *)
  let swap (u : Matched_unit.t) =
    {
      u with
      left = { u.left with outputs = List.rev u.left.outputs @ u.left.inputs };
    }
  in
  let reordered = { m with units = List.map swap m.units } in
  Format.printf "output order is part of the preimage: %b@."
    (Mapping_artifact.preimage m <> Mapping_artifact.preimage reordered);
  let j = Mapping_artifact.encode m in
  Format.printf "domains separate: %b@."
    (Digest_preimage.of_json Mapping j <> Digest_preimage.of_json Plan j);
  [%expect
    {|
    set order is not part of the preimage: true
    output order is part of the preimage: true
    domains separate: true |}]

let%expect_test "rule order carries no precedence in the preimage" =
  let other : Metric_rule.t =
    {
      rule with
      id = F.text (Id.Metric_rule.of_string "another");
      priority = Rule_priority.of_int 1;
    }
  in
  let a = { selection with rules = [ rule; other ] } in
  let b = { selection with rules = [ other; rule ] } in
  Format.printf "rule order ignored: %b@."
    (Policy_selection.preimage a = Policy_selection.preimage b);
  let c =
    {
      selection with
      rules = [ rule; { other with priority = Rule_priority.of_int 2 } ];
    }
  in
  Format.printf "priority is part of the preimage: %b@."
    (Policy_selection.preimage a <> Policy_selection.preimage c);
  let g = F.graph_desc in
  let g' = { g with nodes = List.rev g.nodes; sources = List.rev g.sources } in
  Format.printf "graph table order ignored: %b@."
    (Graph_desc.preimage g = Graph_desc.preimage g');
  let swapped = { g with final_outputs = List.rev g.final_outputs } in
  Format.printf "final output order kept: %b@."
    (Graph_desc.preimage g <> Graph_desc.preimage swapped);
  [%expect
    {|
    rule order ignored: true
    priority is part of the preimage: true
    graph table order ignored: true
    final output order kept: true |}]
