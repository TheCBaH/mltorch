(** Random-walk driver for ATen/native equivalence checking.

    Given a walkable op (a [(module Walk.Op)]), thread one PCG through axis
    selection, config mutation and tensor-value synthesis, comparing native vs
    ATen at every step. A run is fully reproducible from its seed. *)

val run :
  (module Aten_walk_recipes.Walk.Op) ->
  ppf:Format.formatter ->
  pcg:Aten_spec.Pcg.t ->
  steps:int ->
  unit
(** [run (module M) ~ppf ~pcg ~steps] validates the (cascaded) initial config
    (step 0), then takes up to [steps] steps. Each step picks an axis, mutates
    it freely, and re-cascades so dependent parameters stay valid; it prints the
    config and the comparison result. Stops on the first mismatch so the failing
    configuration is the last line printed. *)
