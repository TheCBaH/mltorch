(* A region: typed parameters, an ordered body, and the values it yields. It
   may capture dominating immutable values; its own definitions leave only
   through [yields]. A failure ends the invocation instead of yielding. *)
type t = {
  id : Ssa_id.Region.t;
  params : Ssa_value.t list;
  body : t Ssa_stmt.t list;
  yields : Ssa_value.t list;
}
