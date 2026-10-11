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
  n4_bytes : Graph_ir.Tensor_id.t -> (string, string) result;
      (** The exact value of each source of the four-axis graph. *)
  constants : (Tensor.packed Graph_ir.Tensor_id.Map.t, string) result Lazy.t;
      (** Every constant the four-axis graph reads, materialized on demand. *)
  canonicalize : (Pass_map.t, string) result Lazy.t;
      (** Lowered Native to canonical Native; read on demand because it hashes
          every source value, which an endpoint alone does not need. *)
  lower : (Pass_map.t, string) result Lazy.t;
      (** Canonical Native to Native4D. *)
  canonicalize_reverse : (Pass_map.t, string) result Lazy.t;
      (** Canonical Native to lowered Native. *)
  lower_reverse : (Pass_map.t, string) result Lazy.t;
      (** Native4D to canonical Native. *)
}

(** The Native views alone: the lowered and canonical graphs and the pass map
    between them, which exist for every model Native supports. *)
module Canonical : sig
  type four := t

  type t = {
    n0 : Graph_ir.op Graph_model.t;  (** The lowered Native graph. *)
    n1 : Graph_ir.op Graph_model.t;  (** After the canonical passes. *)
    n0_bytes : Graph_ir.Tensor_id.t -> (string, string) result;
        (** The exact value of each source of the lowered graph. *)
    canonical_bytes : Graph_ir.Tensor_id.t -> (string, string) result;
        (** The exact value of each source of the canonical graph. *)
    canonicalize : (Pass_map.t, string) result Lazy.t;
        (** Lowered Native to canonical Native. *)
    canonicalize_reverse : (Pass_map.t, string) result Lazy.t;
        (** Canonical Native to lowered Native. *)
    lowering : (four, string) result Lazy.t;
        (** The four-axis extension, tried only when forced: a graph outside the
            four-axis domain is its [Error], and never a Native run. *)
  }
end

val lowered_name :
  Native_interp.Capture.t -> Graph_ir.Tensor_id.t -> string option
(** The exporter name of a source of the lowered graph. *)

val through :
  Map_export.t ->
  (Graph_ir.Tensor_id.t -> string option) ->
  Graph_ir.Tensor_id.t ->
  string option
(** The exporter name of a source of the graph a map leads to, read off the
    source value its cluster holds: [through map upstream_name id]. *)

val canonical :
  hashers:Inference_compare_codecs.Payload_hasher.Registry.t ->
  algorithm:Id.Digest_algorithm.t ->
  ctx:Inference_compare_codecs.Logical_value.Context.t ->
  archive:Pt2_archive.t ->
  native:Native_model.t ->
  source_bytes:(string -> (string, string) result) ->
  (Canonical.t, string) result
(** Builds the Native views without the four-axis lowering, so a model Native
    supports but Native4D refuses still has them. [source_bytes] is the value of
    an original source by its exporter name; it is read only when a pass map is.
*)

val make :
  hashers:Inference_compare_codecs.Payload_hasher.Registry.t ->
  algorithm:Id.Digest_algorithm.t ->
  ctx:Inference_compare_codecs.Logical_value.Context.t ->
  archive:Pt2_archive.t ->
  native:Native_model.t ->
  source_bytes:(string -> (string, string) result) ->
  (t, string) result
(** [source_bytes] as for {!canonical}. *)

val refusal : Native4d.Error.t Err.Error.t -> string
(** The text of a lowering failure: the graph is outside the domain. *)
