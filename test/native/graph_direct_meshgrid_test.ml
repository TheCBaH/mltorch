(* [Meshgrid], its own family. See graph_direct_fixtures.ml. *)

open Graph_ir
open Graph_direct_fixtures

let%expect_test "Direct graph: meshgrid of two rank-1 inputs" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"mg" ~outputs:Fun.id
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 2) ~name:"b" () in
          meshgrid ~name:"out" [ a; b ])
    in
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (1 + chan c)) in
    let b =
      Tensor.materialize (s1c 2) (fun c -> float_of_int (10 * (1 + chan c)))
    in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    Err.return
      (List.map (fun oid -> Tensor_id.Map.find oid env) g.Graph.outputs)
  in
  (match result with
  | Ok outs ->
      List.iteri (fun i t -> Format.printf "out%d = %a@." i Tensor.pp t) outs
  | Error e -> Format.printf "%a@." pp_error (Err.Error.kind e));
  (* [a]=[1,2,3] varies along W, broadcast along C; [b]=[10,20] varies along
     C, broadcast along W -- exactly `torch.meshgrid([a,b], indexing="ij")`'s
     own contract. *)
  [%expect
    {|
    out0 = tensor f32 [W=3 C=2] {1, 1, 2, 2, 3, 3}
    out1 = tensor f32 [W=3 C=2] {10, 20, 10, 20, 10, 20} |}]
