(** The C text shared by every generated translation unit. *)

module Name : sig
  type t =
    | Bf16_to_f32  (** the binary32 value of a [bf16] cell, no binary64 step *)
    | Bf16_to_float
    | Coord_failure
    | Cos_f32  (** {!F32_math.Value.cos}'s sequence in [float] *)
    | Erf
    | Erf_f32  (** {!F32_math.Value.erf}'s sequence in [float] *)
    | Exp_f32  (** {!F32_math.Value.exp}'s sequence in [float] *)
    | F16_to_f32  (** the binary32 value of an [f16] cell, no binary64 step *)
    | F16_to_float
    | F32_prelude
        (** [FLT_EVAL_METHOD == 0] asserted: what every fp32 kernel needs *)
    | Float_max
    | Float_max_f32
    | Floor_div
    | I64_div
    | I64_from_float
    | I64_from_float_failure
    | Idx_clamp_low
    | Idx_max
    | Idx_min
    | Log_f32  (** {!F32_math.Value.log}'s sequence in [float] *)
    | Sin_f32  (** {!F32_math.Value.sin}'s sequence in [float] *)
    | Vector_prelude
        (** the generic-vector types and lane helpers of vectorized C; its users
            also need {!Float_max} and {!Erf} *)
    | Vector_prelude_f32
        (** the binary32 counterpart, {!f32_lanes} floats per vector; its users
            also need {!Float_max_f32}, {!Cos_f32}, {!Erf_f32}, {!Exp_f32},
            {!Log_f32}, {!Sin_f32} and {!F32_prelude} *)
    | Vector_prelude_fma_f32
        (** [vs_fma], a fused multiply-add per lane through [fmaf]; its users
            also need {!Vector_prelude_f32} *)

  val all : t list
  val to_string : t -> string
end

val f32_lanes : int
(** The logical width of a binary32 vector loop in C: the lane count
    {!Loop_target.f32} plans for. *)

val error_words : int
(** The [v] slots of [struct model_error]. *)

val kind_index : Loop_js_failure.Kind.t -> int
(** A failure kind's number: its position in {!Loop_js_failure.Kind.all}. Part
    of the ABI, so it changes only with that (alphabetical, closed) list. *)

val prelude : string
(** Includes, [struct model_error] and [fail_set]. *)

val prelude_in : Loop_c_dialect.t -> string
(** {!prelude} in a dialect; [prelude_in Gnu] is {!prelude}. *)

val helpers : ?dialect:Loop_c_dialect.t -> Name.t list -> string
(** The definitions of exactly these helpers, in {!Name.all} order. *)
