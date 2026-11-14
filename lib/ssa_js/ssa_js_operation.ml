(* What has no JavaScript form, as a case rather than a sentence. *)

(* @alphabetical *)
type t =
  | Attention
  | Bfloat16_rounding
  | Binary32_erf
  | Binary64_fma
  | Contraction_extent_past_31_bits
  | Contraction_format
  | Contraction_paired
  | Contraction_stride_past_31_bits
  | Int64_to_binary32
  | Lane_wise
  | Vector
  | Vector_or_mask

let pp ppf = function
  | Attention -> Fmt.string ppf "an attention"
  | Bfloat16_rounding -> Fmt.string ppf "a bfloat16 rounding as an operation"
  | Binary32_erf -> Fmt.string ppf "a binary32 erf"
  | Binary64_fma -> Fmt.string ppf "a binary64 fused multiply-add"
  | Contraction_extent_past_31_bits ->
      Fmt.string ppf "a contraction extent past 31 bits"
  | Contraction_format ->
      Fmt.string ppf "a contraction over other than binary32"
  | Contraction_paired -> Fmt.string ppf "a paired contraction"
  | Contraction_stride_past_31_bits ->
      Fmt.string ppf "a contraction stride past 31 bits"
  | Int64_to_binary32 -> Fmt.string ppf "an int64 to binary32 conversion"
  | Lane_wise -> Fmt.string ppf "a lane-wise operation"
  | Vector -> Fmt.string ppf "a vector operation"
  | Vector_or_mask -> Fmt.string ppf "a vector or mask"
