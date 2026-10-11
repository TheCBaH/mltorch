module W = Wire

type t = {
  both_nan_same_bits : int64;
  both_nan_different_bits : int64;
  left_nan_only : int64;
  right_nan_only : int64;
  same_sign_infinities : int64;
  opposite_sign_infinities : int64;
  left_infinite_right_finite : int64;
  right_infinite_left_finite : int64;
  signed_zero_differences : int64;
}

let zero =
  {
    both_nan_same_bits = 0L;
    both_nan_different_bits = 0L;
    left_nan_only = 0L;
    right_nan_only = 0L;
    same_sign_infinities = 0L;
    opposite_sign_infinities = 0L;
    left_infinite_right_finite = 0L;
    right_infinite_left_finite = 0L;
    signed_zero_differences = 0L;
  }

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let count n = W.Obj.required o n W.nonneg_int64 in
      let* both_nan_same_bits = count "both_nan_same_bits" in
      let* both_nan_different_bits = count "both_nan_different_bits" in
      let* left_nan_only = count "left_nan_only" in
      let* right_nan_only = count "right_nan_only" in
      let* same_sign_infinities = count "same_sign_infinities" in
      let* opposite_sign_infinities = count "opposite_sign_infinities" in
      let* left_infinite_right_finite = count "left_infinite_right_finite" in
      let* right_infinite_left_finite = count "right_infinite_left_finite" in
      let* signed_zero_differences = count "signed_zero_differences" in
      Ok
        {
          both_nan_same_bits;
          both_nan_different_bits;
          left_nan_only;
          right_nan_only;
          same_sign_infinities;
          opposite_sign_infinities;
          left_infinite_right_finite;
          right_infinite_left_finite;
          signed_zero_differences;
        })

let encode s =
  W.obj
    [
      W.field "both_nan_same_bits" W.enc_int64 s.both_nan_same_bits;
      W.field "both_nan_different_bits" W.enc_int64 s.both_nan_different_bits;
      W.field "left_nan_only" W.enc_int64 s.left_nan_only;
      W.field "right_nan_only" W.enc_int64 s.right_nan_only;
      W.field "same_sign_infinities" W.enc_int64 s.same_sign_infinities;
      W.field "opposite_sign_infinities" W.enc_int64 s.opposite_sign_infinities;
      W.field "left_infinite_right_finite" W.enc_int64
        s.left_infinite_right_finite;
      W.field "right_infinite_left_finite" W.enc_int64
        s.right_infinite_left_finite;
      W.field "signed_zero_differences" W.enc_int64 s.signed_zero_differences;
    ]
