(** Reconciles the graph a dialect actually built with the planned extraction.

    The inventories must match exactly and in order: the realized graph's inputs
    are the declared boundary inputs and its results are the declared boundary
    outputs. A dialect that exposes an additional internal tensor, drops an
    output or reorders either list is refused; an observable added for
    diagnostics must be declared in the unit's boundary (with its own
    counterpart and fidelity check) rather than appear here. Beyond the
    inventory the whole description must be identical to the plan's. *)

open Inference_compare_schema

val check : Extraction.t -> Graph_desc.t -> (unit, Extract_error.t) Err.t
