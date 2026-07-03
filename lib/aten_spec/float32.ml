(* Float32 now lives in [walk_core] (the backend-neutral foundation shared with
   the native engine); re-exported here so [Aten_spec.Float32] keeps resolving
   and its type stays identical to [Walk_core.Float32.t]. *)
include Walk_core.Float32
