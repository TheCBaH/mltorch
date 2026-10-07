(* A generic language-failure exit: an already-computed kind and payload. It
   commits the failure record and ends the invocation; it never reevaluates the
   source operation. Selection expands it into explicit record stores and a
   status return, so no selected program holds one. *)
type t = {
  failure : Mir_failure.t;
  payload : Mir_value.t list;
  order : Mir_value.t;
}
