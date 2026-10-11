(** Element codecs: how a logical element is laid out in row-major,
    little-endian bytes, and what each codec declares about itself.

    Integers stay integers and floats keep their bit patterns: NaN payloads,
    signed zeros and I64 values above 2^53 pass through unchanged. *)

open Inference_compare_schema

type t = { id : Id.Codec.t; element : Capability.element; byte_width : int }

module Registry : sig
  type codec := t
  type t

  val builtin : t
  (** [bool], [f32], [f64], [i32], [i64] and [u8]. *)

  val add : t -> codec -> (t, [> `Duplicate_codec of Id.Codec.t ]) Err.t
  val find : t -> Id.Codec.t -> codec option

  val capabilities : t -> Capability.Codec.t list
  (** Every codec built in element-inspecting and exact. *)
end

val id : string -> Id.Codec.t
(** A builtin codec id by its wire name; raises [Invalid_argument] on a name
    that is not a valid identifier (the builtin names all are). *)

val bool : Id.Codec.t
val f32 : Id.Codec.t
val f64 : Id.Codec.t
val i32 : Id.Codec.t
val i64 : Id.Codec.t
val u8 : Id.Codec.t
val is_float : t -> bool
val is_integer : t -> bool

val bits : t -> string -> int -> int64
(** The element's little-endian bytes zero-extended into an [int64]. The index
    is an element index into [string], already bounded by the caller. *)

val set_bits : t -> Bytes.t -> int -> int64 -> unit
(** Stores the low [byte_width] bytes. *)

val integer_value : t -> string -> int -> int64
(** Sign- or zero-extended according to the codec; [bool] and [u8] are
    zero-extended. Integer codecs only. *)

val float_value : t -> string -> int -> float
(** The element as a double, exact for [f32] and [f64]. Float codecs only. *)
