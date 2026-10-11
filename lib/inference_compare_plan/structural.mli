(** The structurally validated plan: both endpoint indexes, the complete
    partition, the frontier recipes, the deterministic unit order and value
    lifetimes. It names no extracted graph and no sealed policy, so it is not
    executable: a runner needs the extraction (graph bodies and their digests)
    and the policy bindings the later stages add, and no value of this type can
    be turned into an executable plan without both. *)

open Inference_compare_schema

type t

val check :
  limits:Resource_limits.t ->
  options:Correspondence_check.options ->
  left:Graph_desc.t ->
  right:Graph_desc.t ->
  Mapping_artifact.t ->
  (t, Plan_error.t) Err.t
(** Equivalent artifacts that differ only in the order of set-like lists give
    the same plan: the artifact and both graph descriptions are put in canonical
    form first. *)

val mapping : t -> Mapping_artifact.t
(** The canonical form that was validated. *)

val left : t -> Graph_index.t
val right : t -> Graph_index.t
val partition : t -> Partition_check.t
val correspondence : t -> Correspondence_check.t
val order : t -> Unit_order.t

val units : t -> Correspondence_check.Unit_plan.t list
(** In execution order. *)

val coverage : t -> Coverage.t
val final_values : t -> Id.Canonical_value.t list

val mapping_preimage : t -> string
(** The domain-tagged canonical text a mapping digest is computed over. *)
