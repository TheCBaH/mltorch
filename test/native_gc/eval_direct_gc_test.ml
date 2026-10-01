(* Under [Retain.Only] a released payload is garbage while the run is still
   going, not only once it returns: an executor stashes an intermediate in a
   [Weak] slot and, one node after its last reader, forces a major collection
   and looks. Under [All] the env still holds it. See .ai/ (tensor release). *)

open Graph_ir

let c4 = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:4

(* x -> relu (t1) -> sqrt (t2) -> relu (t3). t1's last reader is the sqrt, so
   under [Only] it is gone by the time the final relu runs. *)
let graph () =
  Graph_builder.build ~name:"chain3"
    ~outputs:(fun o -> [ o ])
    (let open Graph_builder in
     let* x = input ~shape:c4 () in
     let* a = relu x in
     let* b = sqrt a in
     relu b)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let collect () =
  Gc.full_major ();
  Gc.full_major ()

(* Whether the first node's output was still reachable when the last node
   started, and after the run returned. *)
let alive_mid_and_after retain =
  let g = graph () in
  let first = List.hd g.Graph.nodes and last = List.nth g.Graph.nodes 2 in
  let slot = Weak.create 1 and mid = ref None in
  let node_executor =
    {
      Node_executor.run =
        (fun _ node ~output:_ ~out_shape:_ ~operands:_ ~direct ->
          if Node_id.equal node.Node.id last.Node.id then (
            collect ();
            mid := Some (Weak.check slot 0));
          let result = direct () in
          (if Node_id.equal node.Node.id first.Node.id then
             match Err.payload result with
             | Ok t -> Weak.set slot 0 (Some t)
             | Error _ -> ());
          result);
    }
  in
  let x = Tensor.materialize c4 (fun _ -> 2.) in
  let env =
    Eval_direct.run ~retain ~node_executor g
      ~inputs:[ (List.hd g.Graph.inputs, x) ]
    |> Err.or_raise ~pp_error:Eval_direct.pp_error
  in
  collect ();
  let after = Weak.check slot 0 in
  ignore (Sys.opaque_identity env);
  (Option.get !mid, after)

let%expect_test
    "Only empty: an intermediate is collectable after its last reader" =
  let mid, after =
    alive_mid_and_after (Release_schedule.Retain.Only Tensor_id.Set.empty)
  in
  Fmt.pr "alive mid-run: %b, after: %b@." mid after;
  [%expect {| alive mid-run: false, after: false |}]

let%expect_test "All: the env keeps every intermediate" =
  let mid, after = alive_mid_and_after Release_schedule.Retain.All in
  Fmt.pr "alive mid-run: %b, after: %b@." mid after;
  [%expect {| alive mid-run: true, after: true |}]
