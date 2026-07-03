(** ATen specialization of the shared random-walk runner: drives a generated
    ATen walk module ([build] produces an [Aten_spec.Op_spec.t]) and verifies
    each config through the ATen-vs-native harness. The generic loop is
    [Walk_core.Walk.run]; this fixes [verify] to [Aten_spec_run.run]. *)

val run :
  (module Walk_core.Walk.Op with type subject = Aten_spec.Op_spec.t) ->
  ppf:Format.formatter ->
  pcg:Walk_core.Pcg.t ->
  steps:int ->
  unit
