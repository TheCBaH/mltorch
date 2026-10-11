module W = Wire
module Metric_id = W.Text (Id.Tensor_metric)
module Comparator = W.Text (Id.Comparator)

type partition = Reduce_axes of Tensor_axis.t list | Whole_tensor
type zero_norm = Both_zero_equal | Undefined

type mass_policy =
  | Normalize_positive_total
  | Require_unit_mass of { absolute_tolerance : float }

type distribution =
  | Empirical_values
  | Histogram_masses of { support : Metric_support.t; mass : mass_policy }

type kind =
  | Absolute_bias of Difference_statistics.direction
  | Cosine_similarity of { zero_norm : zero_norm }
  | Error_standard_deviation of Difference_statistics.policy
  | Mean_absolute_error
  | Normalized_l2 of { scale : Float_policy.scale; zero_norm : zero_norm }
  | Registered of {
      comparator : Id.Comparator.t;
      implementation_digest : Digest.Implementation.t;
      configuration_digest : Digest.Comparison_policy.t;
    }
  | Root_mean_square_error
  | Wasserstein_1d of distribution

type threshold = At_least of float | At_most of float
type role = Diagnostic_only | Required of threshold

type t = {
  id : Id.Tensor_metric.t;
  kind : kind;
  partition : partition;
  role : role;
}

let decode_zero_norm, encode_zero_norm =
  W.enum ~kind:"zero_norm"
    [ (Both_zero_equal, "both_zero_equal"); (Undefined, "undefined") ]

let decode_mass : mass_policy W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "normalize_positive_total" -> Ok Normalize_positive_total
      | "require_unit_mass" ->
          let* absolute_tolerance =
            W.Obj.required o "absolute_tolerance" W.nonneg_float
          in
          Ok (Require_unit_mass { absolute_tolerance })
      | tag -> W.unknown_tag ~kind:"mass_policy" tag)

let encode_mass = function
  | Normalize_positive_total -> W.tagged "normalize_positive_total" []
  | Require_unit_mass { absolute_tolerance } ->
      W.tagged "require_unit_mass"
        [ W.field "absolute_tolerance" W.enc_float absolute_tolerance ]

let decode_distribution : distribution W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "empirical_values" -> Ok Empirical_values
      | "histogram_masses" ->
          let* support = W.Obj.required o "support" Metric_support.decode in
          let* mass = W.Obj.required o "mass" decode_mass in
          Ok (Histogram_masses { support; mass })
      | tag -> W.unknown_tag ~kind:"distribution" tag)

let encode_distribution = function
  | Empirical_values -> W.tagged "empirical_values" []
  | Histogram_masses { support; mass } ->
      W.tagged "histogram_masses"
        [
          W.field "support" Metric_support.encode support;
          W.field "mass" encode_mass mass;
        ]

let decode_kind : kind W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "absolute_bias" ->
          let* d =
            W.Obj.required o "direction" Difference_statistics.decode_direction
          in
          Ok (Absolute_bias d)
      | "cosine_similarity" ->
          let* zero_norm = W.Obj.required o "zero_norm" decode_zero_norm in
          Ok (Cosine_similarity { zero_norm })
      | "error_standard_deviation" ->
          let* p =
            W.Obj.required o "policy" Difference_statistics.decode_policy
          in
          Ok (Error_standard_deviation p)
      | "mean_absolute_error" -> Ok Mean_absolute_error
      | "normalized_l2" ->
          let* scale = W.Obj.required o "scale" Float_policy.decode_scale in
          let* zero_norm = W.Obj.required o "zero_norm" decode_zero_norm in
          Ok (Normalized_l2 { scale; zero_norm })
      | "registered" ->
          let* comparator = W.Obj.required o "comparator" Comparator.decode in
          let* implementation_digest =
            W.Obj.required o "implementation_digest" Wire_digest.implementation
          in
          let* configuration_digest =
            W.Obj.required o "configuration_digest"
              Wire_digest.comparison_policy
          in
          Ok
            (Registered
               { comparator; implementation_digest; configuration_digest })
      | "root_mean_square_error" -> Ok Root_mean_square_error
      | "wasserstein_1d" ->
          let* d = W.Obj.required o "distribution" decode_distribution in
          Ok (Wasserstein_1d d)
      | tag -> W.unknown_tag ~kind:"metric_kind" tag)

let encode_kind = function
  | Absolute_bias d ->
      W.tagged "absolute_bias"
        [ W.field "direction" Difference_statistics.encode_direction d ]
  | Cosine_similarity { zero_norm } ->
      W.tagged "cosine_similarity"
        [ W.field "zero_norm" encode_zero_norm zero_norm ]
  | Error_standard_deviation p ->
      W.tagged "error_standard_deviation"
        [ W.field "policy" Difference_statistics.encode_policy p ]
  | Mean_absolute_error -> W.tagged "mean_absolute_error" []
  | Normalized_l2 { scale; zero_norm } ->
      W.tagged "normalized_l2"
        [
          W.field "scale" Float_policy.encode_scale scale;
          W.field "zero_norm" encode_zero_norm zero_norm;
        ]
  | Registered { comparator; implementation_digest; configuration_digest } ->
      W.tagged "registered"
        [
          W.field "comparator" Comparator.encode comparator;
          W.field "implementation_digest" Wire_digest.encode_implementation
            implementation_digest;
          W.field "configuration_digest" Wire_digest.encode_comparison_policy
            configuration_digest;
        ]
  | Root_mean_square_error -> W.tagged "root_mean_square_error" []
  | Wasserstein_1d d ->
      W.tagged "wasserstein_1d" [ W.field "distribution" encode_distribution d ]

let decode_partition : partition W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "reduce_axes" ->
          let* axes = W.Obj.required o "axes" (W.list Tensor_axis.decode) in
          Ok (Reduce_axes axes)
      | "whole_tensor" -> Ok Whole_tensor
      | tag -> W.unknown_tag ~kind:"partition" tag)

let encode_partition = function
  | Reduce_axes axes ->
      W.tagged "reduce_axes"
        [ W.field "axes" (W.enc_list Tensor_axis.encode) axes ]
  | Whole_tensor -> W.tagged "whole_tensor" []

let decode_role : role W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "diagnostic_only" -> Ok Diagnostic_only
      | "required_at_least" ->
          let* v = W.Obj.required o "bound" W.float in
          Ok (Required (At_least v))
      | "required_at_most" ->
          let* v = W.Obj.required o "bound" W.float in
          Ok (Required (At_most v))
      | tag -> W.unknown_tag ~kind:"metric_role" tag)

let encode_role = function
  | Diagnostic_only -> W.tagged "diagnostic_only" []
  | Required (At_least v) ->
      W.tagged "required_at_least" [ W.field "bound" W.enc_float v ]
  | Required (At_most v) ->
      W.tagged "required_at_most" [ W.field "bound" W.enc_float v ]

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* id = W.Obj.required o "id" Metric_id.decode in
      let* kind = W.Obj.required o "kind" decode_kind in
      let* partition = W.Obj.required o "partition" decode_partition in
      let* role = W.Obj.required o "role" decode_role in
      Ok { id; kind; partition; role })

let encode m =
  W.obj
    [
      W.field "id" Metric_id.encode m.id;
      W.field "kind" encode_kind m.kind;
      W.field "partition" encode_partition m.partition;
      W.field "role" encode_role m.role;
    ]
