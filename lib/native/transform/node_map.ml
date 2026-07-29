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
  { Cluster.src = Set.singleton src; dst = Set.singleton dst; label = () }

let fused ~from dst =
  { Cluster.src = Set.of_list from; dst = Set.singleton dst; label = () }

let delete id = { Cluster.src = Set.singleton id; dst = Set.empty; label = () }
