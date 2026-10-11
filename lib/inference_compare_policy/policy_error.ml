open Inference_compare_schema

module Limit = struct
  type t = Path_length | Selector_depth | Selector_nodes | Witnesses
end

type detail =
  | Config of Inference_compare_metrics.Metric_config_error.t
  | Limit of { limit : Limit.t; bound : int64; actual : int64 }
  | No_detail
  | Plan of Inference_compare_plan.Plan_error.t

type t = {
  code : Failure.code;
  rule : Id.Metric_rule.t option;
  comparison : Id.Comparison.t option;
  unit : Id.Unit.t option;
  metric : Id.Tensor_metric.t option;
  detail : detail;
}

let make ?rule ?comparison ?unit ?metric ?(detail = No_detail) code =
  { code; rule; comparison; unit; metric; detail }

let pp ppf e =
  Fmt.string ppf
    (match Failure.encode_code e.code with
    | Jsont.String (s, _) -> s
    | _ -> "?");
  Option.iter (Fmt.pf ppf " rule=%a" Id.Metric_rule.pp) e.rule;
  Option.iter (Fmt.pf ppf " comparison=%a" Id.Comparison.pp) e.comparison;
  Option.iter (Fmt.pf ppf " unit=%a" Id.Unit.pp) e.unit;
  Option.iter (Fmt.pf ppf " metric=%a" Id.Tensor_metric.pp) e.metric;
  match e.detail with
  | No_detail -> ()
  | Config c ->
      Fmt.pf ppf " (%a)" Inference_compare_metrics.Metric_config_error.pp c
  | Limit { bound; actual; _ } ->
      Fmt.pf ppf " limit=%Ld actual=%Ld" bound actual
  | Plan p -> Fmt.pf ppf " (%a)" Inference_compare_plan.Plan_error.pp p

let fail e = Err.fail ~pp_error:pp e
