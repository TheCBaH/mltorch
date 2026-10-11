(* A pass pipeline's correspondence as plain data: complete clusters over both
   graphs, the implicit identities written out, the pass's own claims. *)

open Graph_ir
module Export = Map_export.Make (Native_side) (Native_side)

let ids to_int l =
  String.concat "," (List.map (fun i -> string_of_int (to_int i)) l)

let%expect_test "map export: the clusters of the canonical pipeline" =
  let g = Pipeline_test.mixed () in
  let open Err.Syntax in
  (match
     let* (Rewrite.Origin state) =
       (Rewrite.origin g :> (Rewrite.origin, Pass.error) Err.t)
     in
     let+ (Rewrite.Step (once, map)) =
       Pass.run_all state [ Pipeline.canonical ~fold:false ]
     in
     let e =
       Export.export map ~src:(Rewrite.snapshot state)
         ~dst:(Rewrite.snapshot once)
     in
     (g, Rewrite.graph once, e)
   with
  | Error e -> Format.printf "%a@." Pass.pp_error (Err.Error.kind e)
  | Ok (before, after, (e : Map_export.t)) ->
      List.iter
        (fun (c : Map_export.Node_cluster.t) ->
          Printf.printf "nodes [%s] -> [%s]\n" (ids Node_id.to_int c.src)
            (ids Node_id.to_int c.dst))
        e.nodes;
      List.iter
        (fun (c : Map_export.Value_cluster.t) ->
          Format.printf "values [%s] -> [%s] %a\n"
            (ids Tensor_id.to_int c.src)
            (ids Tensor_id.to_int c.dst)
            Correspondence.pp_relation c.claim)
        e.values;
      (* each graph's nodes are partitioned by the clusters, once each *)
      let covers pick g =
        let all = List.concat_map pick e.nodes |> List.sort Node_id.compare in
        List.equal Node_id.equal all
          (List.sort Node_id.compare
             (List.map (fun (n : Graph_ir.node) -> n.id) g.Graph.nodes))
      in
      Printf.printf "partitions src: %b, dst: %b\n"
        (covers (fun (c : Map_export.Node_cluster.t) -> c.src) before)
        (covers (fun (c : Map_export.Node_cluster.t) -> c.dst) after));
  [%expect
    {|
    nodes [0,1] -> []
    nodes [2] -> [7]
    nodes [3] -> []
    nodes [4] -> []
    nodes [5] -> [5]
    values [0,2] -> [0] identical
    values [1] -> [] identical
    values [4] -> [] identical
    values [5] -> [] identical
    values [3] -> [3] identical
    values [6] -> [6] identical
    partitions src: true, dst: true |}]
