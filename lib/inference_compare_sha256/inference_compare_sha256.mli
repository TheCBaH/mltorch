(** SHA-256 as a {!Inference_compare_codecs.Payload_hasher.BACKEND}. *)

open Inference_compare_schema

val algorithm : Id.Digest_algorithm.t
(** [sha256]. *)

module Backend : Inference_compare_codecs.Payload_hasher.BACKEND

val registry : Inference_compare_codecs.Payload_hasher.Registry.t
(** A registry holding only this backend. *)
