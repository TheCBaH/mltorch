(* The lowering inspector's whole-graph overlay data. See the .mli. *)

module String_map = Map.Make (String)

type t = { metric : Lowering_key.Metric.t; values : float option String_map.t }
type error = Me_limits.over_limit_error

let pp_error fmt : [< error ] -> unit = function
  | `Over_limit o -> Me_limits.Over_limit.pp fmt o

let check_count ~limits n =
  Me_limits.check ~scope:Me_limits.Scope.Graph Me_limits.Field.Node_data_results
    n ~ceiling:limits.Me_limits.Limits.max_node_data_results_per_graph

let create ~limits ~metric values =
  let open Err.Syntax in
  let+ () = check_count ~limits (List.length values) in
  let finite = function
    | Some v when Float.is_finite v -> Some v
    | Some _ | None -> None
  in
  {
    metric;
    values =
      List.fold_left
        (fun m (node, v) -> String_map.add (Me_ids.op_node node) (finite v) m)
        String_map.empty values;
  }

let metric_jsont =
  Jsont.enum ~kind:"lowering metric"
    (List.map
       (fun m -> (Lowering_key.Metric.name m, m))
       Lowering_key.Metric.all)

let jsont =
  Jsont.Object.map ~kind:"loweringSummary" (fun metric values ->
      { metric; values })
  |> Jsont.Object.mem "metric" metric_jsont ~enc:(fun t -> t.metric)
  |> Jsont.Object.mem "values"
       (Jsont.Object.as_string_map (Jsont.option Jsont.number))
       ~enc:(fun t -> t.values)
  |> Jsont.Object.finish
