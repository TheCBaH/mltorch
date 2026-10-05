(* An immutable program revision. [entry] takes the invocation's effect and
   yields it back. The counters continue id allocation for a pass that returns
   a new revision; analyses are keyed by [revision] and are stale for any other. *)
type t = {
  revision : Ssa_id.Revision.t;
  buffers : Ssa_buffer.t list;
      (** Binding order: an input is validated, in this order, before anything
          runs. *)
  entry : Ssa_region.t;
  next_value : Ssa_id.Value.Next.t;
  next_region : Ssa_id.Region.Next.t;
}

let find_buffer p id =
  List.find_opt
    (fun (b : Ssa_buffer.t) -> Ssa_id.Buffer.equal b.id id)
    p.buffers
