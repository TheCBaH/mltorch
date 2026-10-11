open Inference_compare_schema
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value

type error =
  | Config of Metric_config_error.t
  | Element of Element_compare.error
  | Metric of Tensor_metrics.error

let pp_error ppf = function
  | Config c -> Metric_config_error.pp ppf c
  | Element e -> Element_compare.pp_error ppf e
  | Metric e -> Tensor_metrics.pp_error ppf e

let lift f = function Ok v -> Ok v | Error e -> Error (Err.Error.map_kind f e)

let compare ?(max_samples = 8) ctx env (policy : Comparison_policy.t)
    ~comparison_id ~policy_digest ~left ~right =
  let ( let* ) = Err.bind in
  let* () =
    lift
      (fun c -> Config c)
      (Policy_check.validate ~codecs:ctx.L.Context.codecs env (L.spec left)
         policy)
  in
  let expected = Int64.of_int (L.element_count left) in
  let result ?element ?(metrics = []) ~checked verdict : Comparison_result.t =
    {
      comparison_id;
      policy_digest;
      elements_expected = expected;
      elements_checked = checked;
      element;
      tensor_metrics = metrics;
      verdict;
    }
  in
  if not (Tensor_spec.equal (L.spec left) (L.spec right)) then
    (* Different shape or dtype is a metadata error, never a zero-element
       success. *)
    Ok (result ~checked:0L Metadata_error)
  else
    let registered_element =
      match policy.element with Some (Registered _) -> true | _ -> false
    in
    let* element =
      if registered_element then Ok None
      else
        lift
          (fun e -> Element e)
          (Element_compare.compare ~max_samples ctx policy ~left ~right)
        |> Result.map Option.some
    in
    let metrics_spec = Policy_check.distinct_metrics policy in
    let rec run acc = function
      | [] -> Ok (List.rev acc)
      | (m : Tensor_metric.t) :: rest ->
          let* r =
            lift
              (fun e -> Metric e)
              (Tensor_metrics.compute ~max_samples ctx env m ~left ~right)
          in
          run ((m, r) :: acc) rest
    in
    let* computed = run [] metrics_spec in
    let unavailable =
      List.exists
        (fun (m, r) ->
          match (m : Tensor_metric.t).role with
          | Diagnostic_only -> false
          | Required _ ->
              r.Metric_result.status <> Complete
              || not (Int64.equal r.groups_checked r.groups_expected))
        computed
    in
    let metric_violation =
      List.exists
        (fun (m, (r : Metric_result.t)) ->
          match (m : Tensor_metric.t).role with
          | Diagnostic_only -> false
          | Required _ -> (
              match r.violations with
              | Some v -> Int64.compare v 0L > 0
              | None -> false))
        computed
    in
    let element_violation =
      match element with
      | Some e -> Element_compare.rule_passed e = Some false
      | None -> false
    in
    let verdict : Comparison_result.verdict =
      if registered_element then Unsupported
      else if unavailable then Metric_unavailable
      else if element_violation || metric_violation then Mismatch
      else Pass
    in
    Ok
      (result ?element ~metrics:(List.map snd computed)
         ~checked:(if registered_element then 0L else expected)
         verdict)
