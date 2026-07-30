(* One end of a cross-dialect relation: everything a generic pair functor needs
   from a dialect, bundled.

   A bare snapshot module is not enough, and this bundle exists because of two
   concrete gaps:

   - [Graph_map.check_claim_closure] runs claim propagation over the
     DESTINATION graph, so it needs the destination dialect's transfer table,
     not just its snapshot. Using Native's table against a Native4D destination
     would be wrong in exactly the case the design cares about — an [Equivalent]
     conversion whose downstream edges must be weakened.
   - [Map_verify] needs each side's symbolic evaluator, its signature lookup and
     its group tree, none of which a [Snapshot] exposes.

   [type op] is shared by both submodules, and that sharing is load-bearing
   rather than decoration: [Transfer.propagate] takes the dialect's graph, which
   is [op Graph_common.Graph.t], and [Snapshot.graph] returns one. Without the
   equality the two [op]s are unrelated abstract types and no functor body
   using both will typecheck. *)

module type S = sig
  type op

  (* The dialect itself, so a consumer that needs both the operation table and
     the snapshot — [Rewrite], which rebuilds graphs AND builds maps — can take
     one argument instead of two that might disagree about [op]. *)
  module Dialect : Dialect.S with type op = op

  module Snapshot : sig
    type 'v t
    type packed = Pack : 'v t -> packed

    val edge : 'v t -> Tensor_id.t -> 'v Correspondence.id option
    val node : 'v t -> Graph_common.Node_id.t -> 'v Node_map.id option
    val edges : 'v t -> 'v Correspondence.Universe.t
    val nodes : 'v t -> 'v Node_map.Universe.t
    val graph : 'v t -> op Graph_common.Graph.t

    (* [Rewrite] re-validates the graph it produces, so it needs the view a
       snapshot already holds rather than building a second one. *)
    val view : 'v t -> Graph_view.Make(Dialect).t

    val create :
      op Graph_common.Graph.t ->
      (packed, Graph_view.Make(Dialect).error) Core.result
  end

  module Transfer : sig
    val propagate :
      explicit:Correspondence.relation Tensor_id.Map.t ->
      preserved:(Tensor_id.t -> bool) ->
      op Graph_common.Graph.t ->
      Correspondence.relation Tensor_id.Map.t
  end

  (* Takes the SNAPSHOT rather than the graph, so a caller cannot hand it a
     graph from a different version than the map is indexed by. *)
  val symbolic : 'v Snapshot.t -> Stage_program.t
  val sig_of : 'v Snapshot.t -> Tensor_id.t -> Tensor_sig.t option
  val group_root : 'v Snapshot.t -> Graph_common.Group.t
end
