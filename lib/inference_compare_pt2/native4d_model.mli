(** The Native4D side of the pair: the lowered Native graph through the
    canonical passes and the checked four-axis lowering, each step an ordinary
    graph with its own description and its own correspondence written as a
    mapping artifact. A model outside the four-axis domain is a refusal here,
    never a fall back to the Native route. *)

open Inference_compare_schema

type t = {
  n0 : Graph_ir.op Graph_model.t;  (** The lowered Native graph. *)
  n1 : Graph_ir.op Graph_model.t;  (** After the canonical passes. *)
  canonical_bytes : Graph_ir.Tensor_id.t -> (string, string) result;
      (** The exact value of each source of the canonical graph. *)
  n4 : Native4d.Op.t Graph_model.t;  (** The four-axis graph. *)
  constants : (Tensor.packed Graph_ir.Tensor_id.Map.t, string) result Lazy.t;
      (** Every constant the four-axis graph reads, materialized on demand. *)
  canonicalize : (Pass_map.t, string) result Lazy.t;
      (** Lowered Native to canonical Native; read on demand because it hashes
          every source value, which an endpoint alone does not need. *)
  lower : (Pass_map.t, string) result Lazy.t;
      (** Canonical Native to Native4D. *)
}

val make :
  hashers:Inference_compare_codecs.Payload_hasher.Registry.t ->
  algorithm:Id.Digest_algorithm.t ->
  ctx:Inference_compare_codecs.Logical_value.Context.t ->
  archive:Pt2_archive.t ->
  native:Native_model.t ->
  user_bytes:(string, string) result ->
  (t, string) result
(** [user_bytes] is the value of the model's one input, which the archive does
    not hold; it is read only when a pass map is. *)

val refusal : Native4d.Error.t Err.Error.t -> string
(** The text of a lowering failure: the graph is outside the domain. *)
