module W = Wire
module Rule_id = W.Text (Id.Metric_rule)
module Description = W.Text (Prose.Description)

type action =
  | Add_metrics of Tensor_metric.t list
  | Select_policy of Comparison_policy.t

type empty_match = Allow | Reject

type t = {
  id : Id.Metric_rule.t;
  description : Prose.Description.t;
  selector : Metric_selector.t;
  priority : Rule_priority.t;
  action : action;
  empty_match : empty_match;
}

let decode_empty_match, encode_empty_match =
  W.enum ~kind:"empty_match" [ (Allow, "allow"); (Reject, "reject") ]

let decode_action : action W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "add_metrics" ->
          let* ms = W.Obj.required o "metrics" (W.list Tensor_metric.decode) in
          Ok (Add_metrics ms)
      | "select_policy" ->
          let* p = W.Obj.required o "policy" Comparison_policy.decode in
          Ok (Select_policy p)
      | tag -> W.unknown_tag ~kind:"metric_rule_action" tag)

let encode_action = function
  | Add_metrics ms ->
      W.tagged "add_metrics"
        [ W.field "metrics" (W.enc_list Tensor_metric.encode) ms ]
  | Select_policy p ->
      W.tagged "select_policy" [ W.field "policy" Comparison_policy.encode p ]

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* id = W.Obj.required o "id" Rule_id.decode in
      let* description = W.Obj.required o "description" Description.decode in
      let* selector = W.Obj.required o "selector" Metric_selector.decode in
      let* priority = W.Obj.required o "priority" Rule_priority.decode in
      let* action = W.Obj.required o "action" decode_action in
      let* empty_match = W.Obj.required o "empty_match" decode_empty_match in
      Ok { id; description; selector; priority; action; empty_match })

let encode r =
  W.obj
    [
      W.field "id" Rule_id.encode r.id;
      W.field "description" Description.encode r.description;
      W.field "selector" Metric_selector.encode r.selector;
      W.field "priority" Rule_priority.encode r.priority;
      W.field "action" encode_action r.action;
      W.field "empty_match" encode_empty_match r.empty_match;
    ]
