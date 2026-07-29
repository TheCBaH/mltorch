(* See boundary_index.mli. *)

open Graph_ir

type t = {
  by_dst : Cluster_var.t Tensor_id.Map.t;
  by_src : Cluster_var.t Tensor_id.Map.t;
}

let create clusters =
  (* The counter advances only on a cluster that GETS a variable, so the numbers
     are contiguous and a reader can count them off against the non-vacuous
     clusters in report order. *)
  let _, by_src, by_dst =
    List.fold_left
      (fun (i, s, d) (c : ('src, 'dst) Correspondence.Cluster.t) ->
        if
          Correspondence.Set.is_empty c.src || Correspondence.Set.is_empty c.dst
        then (i, s, d)
        else
          let v = Cluster_var.of_int i in
          let add set m =
            Correspondence.Set.fold
              (fun e m -> Tensor_id.Map.add (Correspondence.raw e) v m)
              set m
          in
          (i + 1, add c.src s, add c.dst d))
      (0, Tensor_id.Map.empty, Tensor_id.Map.empty)
      clusters
  in
  { by_dst; by_src }

let dst t id = Tensor_id.Map.find_opt id t.by_dst
let src t id = Tensor_id.Map.find_opt id t.by_src
