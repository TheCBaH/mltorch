module W = Wire
module Unit_id = W.Text (Id.Unit)
module Rule_id = W.Text (Id.Metric_rule)
module Comparison = W.Text (Id.Comparison)
module Diagnostic = W.Text (Prose.Diagnostic_text)

type code =
  | Ambiguous_route
  | Boundary_mismatch
  | Conflicting_initial_value
  | Duplicate_owner
  | Empty_rule_match
  | Evidence_failed
  | Execution_failed
  | Input_transfer_failed
  | Invalid_selector
  | Lossy_common_input
  | Missing_common_input
  | Missing_owner
  | Missing_policy
  | Missing_tensor_pair
  | Output_inventory_failed
  | Output_order_mismatch
  | Policy_conflict
  | Preparation_refused
  | Resource_exhausted
  | Stale_graph
  | Unavailable_comparison_site
  | Unit_cycle
  | Unknown_member
  | Unknown_override
  | Unsupported_effect
  | Unsupported_policy

type t = {
  code : code;
  phase : Phase.t;
  side : Side.t option;
  unit : Id.Unit.t option;
  node : Node_ref.t option;
  tensor : Tensor_ref.t option;
  rule : Id.Metric_rule.t option;
  comparison : Id.Comparison.t option;
  expected : Prose.Diagnostic_text.t option;
  actual : Prose.Diagnostic_text.t option;
}

let decode_code, encode_code =
  W.enum ~kind:"failure_code"
    [
      (Ambiguous_route, "ambiguous_route");
      (Boundary_mismatch, "boundary_mismatch");
      (Conflicting_initial_value, "conflicting_initial_value");
      (Duplicate_owner, "duplicate_owner");
      (Empty_rule_match, "empty_rule_match");
      (Evidence_failed, "evidence_failed");
      (Execution_failed, "execution_failed");
      (Input_transfer_failed, "input_transfer_failed");
      (Invalid_selector, "invalid_selector");
      (Lossy_common_input, "lossy_common_input");
      (Missing_common_input, "missing_common_input");
      (Missing_owner, "missing_owner");
      (Missing_policy, "missing_policy");
      (Missing_tensor_pair, "missing_tensor_pair");
      (Output_inventory_failed, "output_inventory_failed");
      (Output_order_mismatch, "output_order_mismatch");
      (Policy_conflict, "policy_conflict");
      (Preparation_refused, "preparation_refused");
      (Resource_exhausted, "resource_exhausted");
      (Stale_graph, "stale_graph");
      (Unavailable_comparison_site, "unavailable_comparison_site");
      (Unit_cycle, "unit_cycle");
      (Unknown_member, "unknown_member");
      (Unknown_override, "unknown_override");
      (Unsupported_effect, "unsupported_effect");
      (Unsupported_policy, "unsupported_policy");
    ]

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* code = W.Obj.required o "code" decode_code in
      let* phase = W.Obj.required o "phase" Phase.decode in
      let* side = W.Obj.optional o "side" Side.decode in
      let* unit = W.Obj.optional o "unit" Unit_id.decode in
      let* node = W.Obj.optional o "node" Node_ref.decode in
      let* tensor = W.Obj.optional o "tensor" Tensor_ref.decode in
      let* rule = W.Obj.optional o "rule" Rule_id.decode in
      let* comparison = W.Obj.optional o "comparison" Comparison.decode in
      let* expected = W.Obj.optional o "expected" Diagnostic.decode in
      let* actual = W.Obj.optional o "actual" Diagnostic.decode in
      Ok
        {
          code;
          phase;
          side;
          unit;
          node;
          tensor;
          rule;
          comparison;
          expected;
          actual;
        })

let encode f =
  W.obj
    [
      W.field "code" encode_code f.code;
      W.field "phase" Phase.encode f.phase;
      W.field "side" (W.enc_option Side.encode) f.side;
      W.field "unit" (W.enc_option Unit_id.encode) f.unit;
      W.field "node" (W.enc_option Node_ref.encode) f.node;
      W.field "tensor" (W.enc_option Tensor_ref.encode) f.tensor;
      W.field "rule" (W.enc_option Rule_id.encode) f.rule;
      W.field "comparison" (W.enc_option Comparison.encode) f.comparison;
      W.field "expected" (W.enc_option Diagnostic.encode) f.expected;
      W.field "actual" (W.enc_option Diagnostic.encode) f.actual;
    ]
