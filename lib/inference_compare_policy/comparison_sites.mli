(** Existing comparison sites: the unit of metric assignment.

    One site per canonical output binding (a canonical candidate, anchored on
    the boundary tensors its left and right candidate recipes read) and one per
    declared member pair (anchored on its two members). Anchors are deduplicated
    and sorted; a canonical argument that a candidate recipe passes through is
    traced to the unit's input tensors whose bindings read it. Nothing here
    guesses an anchor from a name, and an internal or fused-away value never
    becomes a site. *)

open Inference_compare_schema

val build :
  Inference_compare_plan.Structural.t ->
  (Comparison_site.t list, Policy_error.t) Err.t
(** Sorted by comparison id. *)

val find : Comparison_site.t list -> Id.Comparison.t -> Comparison_site.t option
