(** Each scalar binary32 [Ffma] as the operations of {!F32_math.fma32}: the
    exact product and TwoSum in binary64, rounded to odd through a bitcast, then
    to binary32, bit-equal to a hardware [fmaf]. For a target with no fused
    multiply-add. A vector or binary64 [Ffma] is left as it is. *)

val program : Mir_verify.Generic.t -> Mir_verify.Generic.t
