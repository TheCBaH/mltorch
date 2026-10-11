module W = Wire
module Comparison = W.Text (Id.Comparison)

type t = {
  default : Comparison_policy.t option;
  explicit : (Id.Comparison.t * Comparison_policy.t) list;
  rules : Metric_rule.t list;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* default = W.Obj.optional o "default" Comparison_policy.decode in
      let* explicit =
        W.Obj.required o "explicit"
          (W.unique
             ~key:(fun (c, _) -> Id.Comparison.to_string c)
             (W.list (W.pair Comparison.decode Comparison_policy.decode)))
      in
      let* rules =
        W.Obj.required o "rules"
          (W.unique
             ~key:(fun (r : Metric_rule.t) -> Id.Metric_rule.to_string r.id)
             (W.list Metric_rule.decode))
      in
      Ok { default; explicit; rules })

let encode s =
  W.obj
    [
      W.field "default" (W.enc_option Comparison_policy.encode) s.default;
      W.field "explicit"
        (W.enc_list (W.enc_pair Comparison.encode Comparison_policy.encode))
        s.explicit;
      W.field "rules" (W.enc_list Metric_rule.encode) s.rules;
    ]

(* List order carries no precedence, so rules and explicit bindings sort by
   identifier; priorities, not position, order the rules. *)
let canonicalize s =
  let rule (r : Metric_rule.t) =
    let action : Metric_rule.action =
      match r.action with
      | Select_policy p -> Select_policy (Comparison_policy.canonicalize p)
      | Add_metrics ms -> Add_metrics (Comparison_policy.sort_metrics ms)
    in
    { r with Metric_rule.action }
  in
  {
    default = Option.map Comparison_policy.canonicalize s.default;
    explicit =
      List.stable_sort
        (fun (a, _) (b, _) -> Id.Comparison.compare a b)
        (List.map
           (fun (c, p) -> (c, Comparison_policy.canonicalize p))
           s.explicit);
    rules =
      List.stable_sort
        (fun (a : Metric_rule.t) (b : Metric_rule.t) ->
          Id.Metric_rule.compare a.id b.id)
        (List.map rule s.rules);
  }

let preimage s =
  Digest_preimage.of_json Policy_selection (encode (canonicalize s))
