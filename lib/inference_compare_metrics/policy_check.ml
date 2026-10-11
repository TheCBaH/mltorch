open Inference_compare_schema
module E = Inference_compare_codecs.Element_codec

let ( let* ) = Err.bind
let cfg e = Err.fail ~pp_error:Metric_config_error.pp e

let distinct_metrics (p : Comparison_policy.t) =
  List.fold_left
    (fun acc (m : Tensor_metric.t) ->
      if
        List.exists
          (fun (x : Tensor_metric.t) -> Id.Tensor_metric.equal x.id m.id)
          acc
      then acc
      else acc @ [ m ])
    [] p.tensor_metrics

let required_checks (p : Comparison_policy.t) =
  (match p.element with Some _ -> 1 | None -> 0)
  + List.length
      (List.filter
         (fun (m : Tensor_metric.t) -> m.role <> Diagnostic_only)
         (distinct_metrics p))

let validate ~codecs env (spec : Tensor_spec.t) (p : Comparison_policy.t) =
  let codec = spec.dtype.codec in
  let* c =
    match E.Registry.find codecs codec with
    | Some c -> Ok c
    | None -> cfg (Metric_config_error.Unsupported_codec codec)
  in
  let* () =
    match (p.element, c.element) with
    | None, (Boolean | Signed_integer | Unsigned_integer) ->
        cfg (Metric_config_error.Exact_rule_required codec)
    | None, _ -> Ok ()
    | Some (Floating _), (Float_binary32 | Float_binary64) -> Ok ()
    | Some Exact_boolean, Boolean -> Ok ()
    | Some Exact_integer, (Signed_integer | Unsigned_integer) -> Ok ()
    | Some (Registered _), _ -> Ok ()
    | Some _, _ -> cfg (Metric_config_error.Element_rule_codec_mismatch codec)
  in
  let* () =
    match p.element with
    | Some (Floating f) when f.scale <> p.relative_scale ->
        cfg Metric_config_error.Element_rule_scale_mismatch
    | _ -> Ok ()
  in
  let* () =
    let rec dup = function
      | [] -> Ok ()
      | (m : Tensor_metric.t) :: rest ->
          if
            List.exists
              (fun (x : Tensor_metric.t) ->
                Id.Tensor_metric.equal x.id m.id && x <> m)
              rest
          then cfg (Metric_config_error.Duplicate_metric_id m.id)
          else dup rest
    in
    dup p.tensor_metrics
  in
  let* () =
    let rec each = function
      | [] -> Ok ()
      | m :: rest ->
          let* () = Tensor_metrics.validate env spec m in
          each rest
    in
    each (distinct_metrics p)
  in
  if required_checks p = 0 then cfg Metric_config_error.No_required_check
  else Ok ()
