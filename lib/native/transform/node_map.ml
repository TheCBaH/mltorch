(* See node_map.mli. *)

open Graph_ir

module Id = struct
  type t = Node_id.t

  let compare = Node_id.compare
  let pp = Node_id.pp

  module Set = Node_id.Set
end

module Label = struct
  type t = unit

  let identity = ()
  let join () () = ()
  let equal () () = true
  let pp fmt () = Fmt.string fmt ""
end

include Cluster_relation.Make (Id) (Label)

let pair src dst =
  {
    Cluster.src = Node_id.Set.singleton src;
    dst = Node_id.Set.singleton dst;
    label = ();
  }

let fused ~from dst =
  {
    Cluster.src = Node_id.Set.of_list from;
    dst = Node_id.Set.singleton dst;
    label = ();
  }

let delete id =
  {
    Cluster.src = Node_id.Set.singleton id;
    dst = Node_id.Set.empty;
    label = ();
  }
