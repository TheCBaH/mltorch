(* A [Graph_map] as plain data: complete clusters over two graph versions with
   the version indices erased, so a consumer outside the transform framework
   (the inference comparison's mapping provider) can read a pipeline's
   correspondence without the snapshot machinery. Implicit identities are
   written out, which the map itself deliberately does not store.

   Nothing here is checked again: the map was validated when it was built, and
   this only reads it. A cluster's claim is the label the pass supplied. *)

module Node_cluster : sig
  type t = { src : Graph_ir.Node_id.t list; dst : Graph_ir.Node_id.t list }
  (** Ascending ids. An empty [dst] is a deletion, an empty [src] a creation. *)
end

module Value_cluster : sig
  type t = {
    src : Graph_ir.Tensor_id.t list;
    dst : Graph_ir.Tensor_id.t list;
    claim : Correspondence.relation;
  }
end

type t = { nodes : Node_cluster.t list; values : Value_cluster.t list }

module Make (Src : Side.S) (Dst : Side.S) : sig
  val export :
    ('s, 'd) Graph_map.t -> src:'s Src.Snapshot.t -> dst:'d Dst.Snapshot.t -> t
end
