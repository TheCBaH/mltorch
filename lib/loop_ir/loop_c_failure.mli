(** Decodes the [struct model_error] a C kernel fills into the row [Loop_interp]
    reports for the same failure: the counterpart of the JavaScript record
    decoder, over the same {!Loop_js_failure} table. *)

val decode :
  sites:Loop_failure.t array ->
  kind:int ->
  v:int64 array ->
  (Loop_interp.error, string) result
(** [sites] is [Loop_js_failure.sites] of the failing program: a record that
    names a site (a scan projection, an unbound local) is decoded against it. A
    record that does not fit its table is [Error], a defect of the emitter and
    not a failure kind. *)
