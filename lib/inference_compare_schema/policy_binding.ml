module W = Wire
module Rule_id = W.Text (Id.Metric_rule)

type base = Default | Explicit | Rules of Id.Metric_rule.t list

module Rule_match = struct
  type t = {
    rule : Id.Metric_rule.t;
    selected_as_base : bool;
    witnesses : Topology_witness.t list;
  }

  let decode : t W.decoder =
    W.record (fun o ->
        let ( let* ) = W.( let* ) in
        let* rule = W.Obj.required o "rule" Rule_id.decode in
        let* selected_as_base = W.Obj.required o "selected_as_base" W.bool in
        let* witnesses =
          W.Obj.required o "witnesses" (W.list Topology_witness.decode)
        in
        Ok { rule; selected_as_base; witnesses })

  let encode m =
    W.obj
      [
        W.field "rule" Rule_id.encode m.rule;
        W.field "selected_as_base" W.enc_bool m.selected_as_base;
        W.field "witnesses" (W.enc_list Topology_witness.encode) m.witnesses;
      ]
end

type t = {
  site : Comparison_site.t;
  policy : Comparison_policy.t;
  policy_digest : Digest.Comparison_policy.t;
  base : base;
  matches : Rule_match.t list;
}

let decode_base : base W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "default" -> Ok Default
      | "explicit" -> Ok Explicit
      | "rules" ->
          let* rs = W.Obj.required o "rules" (W.list Rule_id.decode) in
          Ok (Rules rs)
      | tag -> W.unknown_tag ~kind:"policy_base" tag)

let encode_base = function
  | Default -> W.tagged "default" []
  | Explicit -> W.tagged "explicit" []
  | Rules rs ->
      W.tagged "rules" [ W.field "rules" (W.enc_list Rule_id.encode) rs ]

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* site = W.Obj.required o "site" Comparison_site.decode in
      let* policy = W.Obj.required o "policy" Comparison_policy.decode in
      let* policy_digest =
        W.Obj.required o "policy_digest" Wire_digest.comparison_policy
      in
      let* base = W.Obj.required o "base" decode_base in
      let* matches = W.Obj.required o "matches" (W.list Rule_match.decode) in
      Ok { site; policy; policy_digest; base; matches })

let encode b =
  W.obj
    [
      W.field "site" Comparison_site.encode b.site;
      W.field "policy" Comparison_policy.encode b.policy;
      W.field "policy_digest" Wire_digest.encode_comparison_policy
        b.policy_digest;
      W.field "base" encode_base b.base;
      W.field "matches" (W.enc_list Rule_match.encode) b.matches;
    ]
