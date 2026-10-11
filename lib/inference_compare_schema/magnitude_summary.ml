module W = Wire

type t = {
  finite_values : int64;
  zero_over_zero : int64;
  unbounded : int64;
  nonfinite_operands : int64;
  arithmetic_overflow : int64;
  minimum_finite : float option;
  maximum_finite : float option;
  largest_finite_index : int64 list option;
  mean_finite : Difference_statistics.scalar;
  standard_deviation_finite : Difference_statistics.scalar;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let count n = W.Obj.required o n W.nonneg_int64 in
      let* finite_values = count "finite_values" in
      let* zero_over_zero = count "zero_over_zero" in
      let* unbounded = count "unbounded" in
      let* nonfinite_operands = count "nonfinite_operands" in
      let* arithmetic_overflow = count "arithmetic_overflow" in
      let* minimum_finite = W.Obj.optional o "minimum_finite" W.float in
      let* maximum_finite = W.Obj.optional o "maximum_finite" W.float in
      let* largest_finite_index =
        W.Obj.optional o "largest_finite_index" (W.list W.nonneg_int64)
      in
      let* mean_finite =
        W.Obj.required o "mean_finite" Difference_statistics.decode_scalar
      in
      let* standard_deviation_finite =
        W.Obj.required o "standard_deviation_finite"
          Difference_statistics.decode_scalar
      in
      Ok
        {
          finite_values;
          zero_over_zero;
          unbounded;
          nonfinite_operands;
          arithmetic_overflow;
          minimum_finite;
          maximum_finite;
          largest_finite_index;
          mean_finite;
          standard_deviation_finite;
        })

let encode s =
  W.obj
    [
      W.field "finite_values" W.enc_int64 s.finite_values;
      W.field "zero_over_zero" W.enc_int64 s.zero_over_zero;
      W.field "unbounded" W.enc_int64 s.unbounded;
      W.field "nonfinite_operands" W.enc_int64 s.nonfinite_operands;
      W.field "arithmetic_overflow" W.enc_int64 s.arithmetic_overflow;
      W.field "minimum_finite" (W.enc_option W.enc_float) s.minimum_finite;
      W.field "maximum_finite" (W.enc_option W.enc_float) s.maximum_finite;
      W.field "largest_finite_index"
        (W.enc_option (W.enc_list W.enc_int64))
        s.largest_finite_index;
      W.field "mean_finite" Difference_statistics.encode_scalar s.mean_finite;
      W.field "standard_deviation_finite" Difference_statistics.encode_scalar
        s.standard_deviation_finite;
    ]
