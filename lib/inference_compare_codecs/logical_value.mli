(** A logical tensor value: a spec, row-major little-endian element bytes owned
    immutably, and a digest of both. The digest is an index and an integrity
    check; exact equality of two values always compares the bytes. *)

open Inference_compare_schema

module Context : sig
  type t = {
    codecs : Element_codec.Registry.t;
    hashers : Payload_hasher.Registry.t;
    algorithm : Id.Digest_algorithm.t;
        (** The algorithm for logical-value digests. *)
    ledger : Owned_blob.Ledger.t;
  }
end

type t

val make : Context.t -> Tensor_spec.t -> string -> (t, Codec_error.t) Err.t
(** Validates the codec, the checked element count and the exact byte length,
    and computes the digest. Takes ownership of the bytes. *)

val spec : t -> Tensor_spec.t
val digest : t -> Digest.Logical_value.t
val contents : t -> string
val storage : t -> Owned_blob.t

val element_count : t -> int
(** Bounded by [Sys.max_string_length] at construction. *)

val codec : Context.t -> t -> Element_codec.t
(** The element codec of a made value; raises [Invalid_argument] only if the
    context's registry no longer holds it. *)

val equal : t -> t -> bool
(** Same spec and byte-for-byte equal storage. *)

val drop : t -> unit

val of_integers :
  Context.t -> Tensor_spec.t -> int64 list -> (t, Codec_error.t) Err.t
(** Little-endian truncation to the codec's width: for building fixtures. *)

val of_floats :
  Context.t -> Tensor_spec.t -> float list -> (t, Codec_error.t) Err.t
(** [f32] rounds to nearest even; for building fixtures. *)

val digest_of_parts :
  Context.t ->
  Tensor_spec.t ->
  string ->
  (Digest.Logical_value.t, Codec_error.t) Err.t
(** The digest [make] would record, without taking ownership. *)
