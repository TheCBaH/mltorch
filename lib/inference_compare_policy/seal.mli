(** Sealing: concrete per-site policy digests, the selection digest and the
    bindings digest, all computed under the algorithm the manifest pins for
    these domains. A sealed binding set is what both donor passes of a run
    share; any change to a relevant graph, mapping, conversion, support or
    policy changes a digest here. *)

open Inference_compare_schema

type error = Hasher of Inference_compare_codecs.Payload_hasher.error

val pp_error : Format.formatter -> error -> unit

val seal :
  hashers:Inference_compare_codecs.Payload_hasher.Registry.t ->
  algorithm:Id.Digest_algorithm.t ->
  selection:Policy_selection.t ->
  Policy_binding.t list ->
  (Policy_bindings.t, error) Err.t
(** Replaces each binding's [policy_digest] with the digest of its policy's
    canonical preimage, then digests the declarative selection and the whole
    binding set. *)

val placeholder : Id.Digest_algorithm.t -> Digest.Comparison_policy.t
(** A digest of zeros, for bindings not yet sealed. *)

val selection_digest :
  hashers:Inference_compare_codecs.Payload_hasher.Registry.t ->
  algorithm:Id.Digest_algorithm.t ->
  Policy_selection.t ->
  (Digest.Policy_selection.t, error) Err.t
