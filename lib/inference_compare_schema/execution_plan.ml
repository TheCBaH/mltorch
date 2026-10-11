module W = Wire
module Unit_id = W.Text (Id.Unit)
module Canonical = W.Text (Id.Canonical_value)
module Comparison = W.Text (Id.Comparison)

type t = {
  mapping_digest : Digest.Mapping.t;
  frontier_mapping_digest : Digest.Frontier_mapping.t;
  policy_selection_digest : Digest.Policy_selection.t;
  policy_bindings_digest : Digest.Policy_bindings.t;
  comparison_policies : (Id.Comparison.t * Digest.Comparison_policy.t) list;
  order : Planned_unit.t list;
  final_values : Id.Canonical_value.t list;
  consumers : (Id.Canonical_value.t * Id.Unit.t list) list;
  coverage : Coverage.t;
  plan_digest : Digest.Plan.t;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* mapping_digest =
        W.Obj.required o "mapping_digest" Wire_digest.mapping
      in
      let* frontier_mapping_digest =
        W.Obj.required o "frontier_mapping_digest" Wire_digest.frontier_mapping
      in
      let* policy_selection_digest =
        W.Obj.required o "policy_selection_digest" Wire_digest.policy_selection
      in
      let* policy_bindings_digest =
        W.Obj.required o "policy_bindings_digest" Wire_digest.policy_bindings
      in
      let* comparison_policies =
        W.Obj.required o "comparison_policies"
          (W.list (W.pair Comparison.decode Wire_digest.comparison_policy))
      in
      let* order = W.Obj.required o "order" (W.list Planned_unit.decode) in
      let* final_values =
        W.Obj.required o "final_values" (W.list Canonical.decode)
      in
      let* consumers =
        W.Obj.required o "consumers"
          (W.list (W.pair Canonical.decode (W.list Unit_id.decode)))
      in
      let* coverage = W.Obj.required o "coverage" Coverage.decode in
      let* plan_digest = W.Obj.required o "plan_digest" Wire_digest.plan in
      Ok
        {
          mapping_digest;
          frontier_mapping_digest;
          policy_selection_digest;
          policy_bindings_digest;
          comparison_policies;
          order;
          final_values;
          consumers;
          coverage;
          plan_digest;
        })

let encode p =
  W.obj
    [
      W.field "mapping_digest" Wire_digest.encode_mapping p.mapping_digest;
      W.field "frontier_mapping_digest" Wire_digest.encode_frontier_mapping
        p.frontier_mapping_digest;
      W.field "policy_selection_digest" Wire_digest.encode_policy_selection
        p.policy_selection_digest;
      W.field "policy_bindings_digest" Wire_digest.encode_policy_bindings
        p.policy_bindings_digest;
      W.field "comparison_policies"
        (W.enc_list
           (W.enc_pair Comparison.encode Wire_digest.encode_comparison_policy))
        p.comparison_policies;
      W.field "order" (W.enc_list Planned_unit.encode) p.order;
      W.field "final_values" (W.enc_list Canonical.encode) p.final_values;
      W.field "consumers"
        (W.enc_list (W.enc_pair Canonical.encode (W.enc_list Unit_id.encode)))
        p.consumers;
      W.field "coverage" Coverage.encode p.coverage;
      W.field "plan_digest" Wire_digest.encode_plan p.plan_digest;
    ]
