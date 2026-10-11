(** Compile-cache identity for a prepared extracted graph. *)

val make :
  Inference_compare_extract.Extraction.t ->
  Inference_compare_schema.Engine_identity.t ->
  string
(** Equal keys mean the same graph body, constants and id maps prepared by the
    same engine build, route, configuration, artifact and environment. *)
