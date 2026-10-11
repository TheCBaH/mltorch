(** The executable plan: the one value the coordinator accepts.

    Sealing joins the structurally validated plan, the extracted graphs and the
    sealed per-site policy bindings, checks that they describe the same
    experiment (every extraction present once and in order; every comparison has
    a binding whose site is exactly the one the plan derives, so bindings
    resolved against a different graph, mapping or conversion are stale), and
    pins everything in one plan digest. There is no way to build one from an
    unchecked record, and a plan decoded from storage is accepted only by
    recomputing it and finding the same identity. *)

open Inference_compare_schema

type t

type error =
  | Extraction_mismatch
      (** An extraction is missing, repeated, out of order or from another
          graph. *)
  | Hasher of Inference_compare_codecs.Payload_hasher.error
  | Missing_binding of Id.Comparison.t
  | Stale_binding of Id.Comparison.t
      (** A binding's site differs from the one the current plan derives. *)
  | Stale_plan  (** A decoded plan differs from the recomputed one. *)
  | Unbound_site of Id.Comparison.t
      (** A binding names a comparison the plan has no site for. *)

val pp_error : Format.formatter -> error -> unit

val seal :
  hashers:Inference_compare_codecs.Payload_hasher.Registry.t ->
  algorithm:Id.Digest_algorithm.t ->
  Inference_compare_plan.Structural.t ->
  Inference_compare_extract.Extraction.t list ->
  Policy_bindings.t ->
  (t, error) Err.t

val verify_decoded : t -> Execution_plan.t -> (unit, error) Err.t
(** Accepts a plan read back from storage only if it equals the plan recomputed
    from the current inputs (digest and full canonical content). *)

val plan : t -> Execution_plan.t
val structural : t -> Inference_compare_plan.Structural.t
val extractions : t -> Inference_compare_extract.Extraction.t list
val bindings : t -> Policy_bindings.t
