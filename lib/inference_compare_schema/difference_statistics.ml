module W = Wire

type direction = Left_minus_right | Right_minus_left
type variance = Population | Sample
type policy = { direction : direction; variance : variance }

type unavailable =
  | Empty_finite_set
  | Insufficient_samples
  | Numeric_overflow
  | Unsupported_dtype

type scalar = Finite_value of float | Unavailable of unavailable
type extremum = { index : int64 list; value : scalar }
type status = Computed | Not_supported | Overflowed

type t = {
  status : status;
  finite_pairs : int64;
  excluded_nonfinite_pairs : int64;
  positive_errors : int64;
  negative_errors : int64;
  zero_errors : int64;
  minimum_signed_error : extremum option;
  maximum_signed_error : extremum option;
  largest_absolute_error : extremum option;
  bias : scalar;
  standard_deviation : scalar;
  mean_absolute_error : scalar;
  root_mean_square_error : scalar;
}

let decode_direction, encode_direction =
  W.enum ~kind:"direction"
    [
      (Left_minus_right, "left_minus_right");
      (Right_minus_left, "right_minus_left");
    ]

let decode_variance, encode_variance =
  W.enum ~kind:"variance" [ (Population, "population"); (Sample, "sample") ]

let decode_unavailable, encode_unavailable =
  W.enum ~kind:"statistic_unavailable"
    [
      (Empty_finite_set, "empty_finite_set");
      (Insufficient_samples, "insufficient_samples");
      (Numeric_overflow, "numeric_overflow");
      (Unsupported_dtype, "unsupported_dtype");
    ]

let decode_status, encode_status =
  W.enum ~kind:"statistics_status"
    [
      (Computed, "computed");
      (Not_supported, "not_supported");
      (Overflowed, "overflowed");
    ]

let decode_policy : policy W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* direction = W.Obj.required o "direction" decode_direction in
      let* variance = W.Obj.required o "variance" decode_variance in
      Ok { direction; variance })

let encode_policy p =
  W.obj
    [
      W.field "direction" encode_direction p.direction;
      W.field "variance" encode_variance p.variance;
    ]

let decode_scalar : scalar W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "finite" ->
          let* v = W.Obj.required o "value" W.float in
          Ok (Finite_value v)
      | "unavailable" ->
          let* r = W.Obj.required o "reason" decode_unavailable in
          Ok (Unavailable r)
      | tag -> W.unknown_tag ~kind:"scalar" tag)

let encode_scalar = function
  | Finite_value v -> W.tagged "finite" [ W.field "value" W.enc_float v ]
  | Unavailable r ->
      W.tagged "unavailable" [ W.field "reason" encode_unavailable r ]

let decode_extremum : extremum W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* index = W.Obj.required o "index" (W.list W.nonneg_int64) in
      let* value = W.Obj.required o "value" decode_scalar in
      Ok { index; value })

let encode_extremum e =
  W.obj
    [
      W.field "index" (W.enc_list W.enc_int64) e.index;
      W.field "value" encode_scalar e.value;
    ]

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let count name = W.Obj.required o name W.nonneg_int64 in
      let extremum name = W.Obj.optional o name decode_extremum in
      let scalar name = W.Obj.required o name decode_scalar in
      let* status = W.Obj.required o "status" decode_status in
      let* finite_pairs = count "finite_pairs" in
      let* excluded_nonfinite_pairs = count "excluded_nonfinite_pairs" in
      let* positive_errors = count "positive_errors" in
      let* negative_errors = count "negative_errors" in
      let* zero_errors = count "zero_errors" in
      let* minimum_signed_error = extremum "minimum_signed_error" in
      let* maximum_signed_error = extremum "maximum_signed_error" in
      let* largest_absolute_error = extremum "largest_absolute_error" in
      let* bias = scalar "bias" in
      let* standard_deviation = scalar "standard_deviation" in
      let* mean_absolute_error = scalar "mean_absolute_error" in
      let* root_mean_square_error = scalar "root_mean_square_error" in
      Ok
        {
          status;
          finite_pairs;
          excluded_nonfinite_pairs;
          positive_errors;
          negative_errors;
          zero_errors;
          minimum_signed_error;
          maximum_signed_error;
          largest_absolute_error;
          bias;
          standard_deviation;
          mean_absolute_error;
          root_mean_square_error;
        })

let encode s =
  W.obj
    [
      W.field "status" encode_status s.status;
      W.field "finite_pairs" W.enc_int64 s.finite_pairs;
      W.field "excluded_nonfinite_pairs" W.enc_int64 s.excluded_nonfinite_pairs;
      W.field "positive_errors" W.enc_int64 s.positive_errors;
      W.field "negative_errors" W.enc_int64 s.negative_errors;
      W.field "zero_errors" W.enc_int64 s.zero_errors;
      W.field "minimum_signed_error"
        (W.enc_option encode_extremum)
        s.minimum_signed_error;
      W.field "maximum_signed_error"
        (W.enc_option encode_extremum)
        s.maximum_signed_error;
      W.field "largest_absolute_error"
        (W.enc_option encode_extremum)
        s.largest_absolute_error;
      W.field "bias" encode_scalar s.bias;
      W.field "standard_deviation" encode_scalar s.standard_deviation;
      W.field "mean_absolute_error" encode_scalar s.mean_absolute_error;
      W.field "root_mean_square_error" encode_scalar s.root_mean_square_error;
    ]
