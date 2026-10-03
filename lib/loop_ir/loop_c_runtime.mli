(** The C text shared by every generated translation unit. *)

module Name : sig
  type t =
    | Bf16_to_float
    | Coord_failure
    | Erf
    | F16_to_float
    | Float_max
    | Floor_div
    | I64_div
    | I64_from_float
    | I64_from_float_failure
    | Idx_clamp_low
    | Idx_max
    | Idx_min
    | Vector_prelude
        (** the generic-vector types and lane helpers of vectorized C; its users
            also need {!Float_max} and {!Erf} *)

  val all : t list
  val to_string : t -> string
end

val error_words : int
(** The [v] slots of [struct model_error]. *)

val kind_index : Loop_js_failure.Kind.t -> int
(** A failure kind's number: its position in {!Loop_js_failure.Kind.all}. Part
    of the ABI, so it changes only with that (alphabetical, closed) list. *)

val prelude : string
(** Includes, [struct model_error] and [fail_set]. *)

val helpers : Name.t list -> string
(** The definitions of exactly these helpers, in {!Name.all} order. *)
