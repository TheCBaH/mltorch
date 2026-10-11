(* Configuration faults of a metric or a policy: all detectable before any
   engine runs, so planning rejects them rather than a run discovering them. *)

type t =
  | Axis_out_of_range of { axis : int; rank : int }
  | Cosine_bound_out_of_range
  | Duplicate_axis of int
  | Duplicate_metric_id of Inference_compare_schema.Id.Tensor_metric.t
  | Element_rule_scale_mismatch
  | Element_rule_codec_mismatch of Inference_compare_schema.Id.Codec.t
  | Empty_axes
  | Exact_rule_required of Inference_compare_schema.Id.Codec.t
  | Invalid_mass_tolerance
  | Negative_distance_bound
  | No_required_check
  | Nonfinite_bound
  | Payload_too_large
  | Support_positions_mismatch of { declared : int64; group_size : int64 }
  | Unregistered_comparator of Inference_compare_schema.Id.Comparator.t
  | Unsupported_codec of Inference_compare_schema.Id.Codec.t
  | Wrong_threshold_direction

let pp ppf = function
  | Axis_out_of_range { axis; rank } ->
      Fmt.pf ppf "axis %d out of range for rank %d" axis rank
  | Cosine_bound_out_of_range -> Fmt.string ppf "cosine bound outside [-1, 1]"
  | Duplicate_axis a -> Fmt.pf ppf "duplicate axis %d" a
  | Duplicate_metric_id id ->
      Fmt.pf ppf "metric id %a reused with a different specification"
        Inference_compare_schema.Id.Tensor_metric.pp id
  | Element_rule_scale_mismatch ->
      Fmt.string ppf "relative scale differs from the element rule's scale"
  | Element_rule_codec_mismatch c ->
      Fmt.pf ppf "element rule does not fit codec %a"
        Inference_compare_schema.Id.Codec.pp c
  | Empty_axes -> Fmt.string ppf "empty reduction axis list"
  | Exact_rule_required c ->
      Fmt.pf ppf "codec %a requires an exact element rule"
        Inference_compare_schema.Id.Codec.pp c
  | Invalid_mass_tolerance ->
      Fmt.string ppf "mass tolerance must be finite and nonnegative"
  | Negative_distance_bound ->
      Fmt.string ppf "distance bound must be nonnegative"
  | No_required_check -> Fmt.string ppf "policy has no required check"
  | Nonfinite_bound -> Fmt.string ppf "threshold must be finite"
  | Payload_too_large -> Fmt.string ppf "tensor too large to index"
  | Support_positions_mismatch { declared; group_size } ->
      Fmt.pf ppf "support has %Ld positions, group size is %Ld" declared
        group_size
  | Unregistered_comparator c ->
      Fmt.pf ppf "unregistered comparator %a"
        Inference_compare_schema.Id.Comparator.pp c
  | Unsupported_codec c ->
      Fmt.pf ppf "metric unsupported for codec %a"
        Inference_compare_schema.Id.Codec.pp c
  | Wrong_threshold_direction ->
      Fmt.string ppf "threshold direction does not fit the metric"
