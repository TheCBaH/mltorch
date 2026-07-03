(* Pcg now lives in [walk_core] (the backend-neutral foundation shared with the
   native engine); re-exported here so [Aten_spec.Pcg] keeps resolving and its
   type stays identical to [Walk_core.Pcg.t]. *)
include Walk_core.Pcg
