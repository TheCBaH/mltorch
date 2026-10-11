module W = Wire

type t = {
  statistics : Difference_statistics.t;
  bitwise_mismatches : int64;
  tolerance_violations : int64 option;
  first_bitwise_index : int64 list option;
  first_violation_index : int64 list option;
  absolute : Magnitude_summary.t;
  relative : Magnitude_summary.t;
  nonfinite : Nonfinite_summary.t;
  difference_samples : Difference_measurement.sample list;
  max_ulp_distance : Ulp_distance.t option;
  finite_pairs : int64;
  nan_pairs : int64;
  infinity_pairs : int64;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let index n = W.Obj.optional o n (W.list W.nonneg_int64) in
      let count n = W.Obj.required o n W.nonneg_int64 in
      let* statistics =
        W.Obj.required o "statistics" Difference_statistics.decode
      in
      let* bitwise_mismatches = count "bitwise_mismatches" in
      let* tolerance_violations =
        W.Obj.optional o "tolerance_violations" W.nonneg_int64
      in
      let* first_bitwise_index = index "first_bitwise_index" in
      let* first_violation_index = index "first_violation_index" in
      let* absolute = W.Obj.required o "absolute" Magnitude_summary.decode in
      let* relative = W.Obj.required o "relative" Magnitude_summary.decode in
      let* nonfinite = W.Obj.required o "nonfinite" Nonfinite_summary.decode in
      let* difference_samples =
        W.Obj.required o "difference_samples"
          (W.list Difference_measurement.decode_sample)
      in
      let* max_ulp_distance =
        W.Obj.optional o "max_ulp_distance" Ulp_distance.decode
      in
      let* finite_pairs = count "finite_pairs" in
      let* nan_pairs = count "nan_pairs" in
      let* infinity_pairs = count "infinity_pairs" in
      Ok
        {
          statistics;
          bitwise_mismatches;
          tolerance_violations;
          first_bitwise_index;
          first_violation_index;
          absolute;
          relative;
          nonfinite;
          difference_samples;
          max_ulp_distance;
          finite_pairs;
          nan_pairs;
          infinity_pairs;
        })

let encode r =
  W.obj
    [
      W.field "statistics" Difference_statistics.encode r.statistics;
      W.field "bitwise_mismatches" W.enc_int64 r.bitwise_mismatches;
      W.field "tolerance_violations" (W.enc_option W.enc_int64)
        r.tolerance_violations;
      W.field "first_bitwise_index"
        (W.enc_option (W.enc_list W.enc_int64))
        r.first_bitwise_index;
      W.field "first_violation_index"
        (W.enc_option (W.enc_list W.enc_int64))
        r.first_violation_index;
      W.field "absolute" Magnitude_summary.encode r.absolute;
      W.field "relative" Magnitude_summary.encode r.relative;
      W.field "nonfinite" Nonfinite_summary.encode r.nonfinite;
      W.field "difference_samples"
        (W.enc_list Difference_measurement.encode_sample)
        r.difference_samples;
      W.field "max_ulp_distance"
        (W.enc_option Ulp_distance.encode)
        r.max_ulp_distance;
      W.field "finite_pairs" W.enc_int64 r.finite_pairs;
      W.field "nan_pairs" W.enc_int64 r.nan_pairs;
      W.field "infinity_pairs" W.enc_int64 r.infinity_pairs;
    ]
