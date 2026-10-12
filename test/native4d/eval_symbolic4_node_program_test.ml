(* [Eval_symbolic4.node_program]: the one-node program a per-node backend
   consumes. The twin of `test/native/eval_symbolic_node_program_test.ml`, on a
   chain that exercises what a whole-graph program would hide: a node consuming
   an intermediate edge, a repeated operand, and a two-output node. Grounding
   each node's program against its own operands must bitwise-agree with
   [Eval_direct4.run]'s per-edge tensors. *)

open Native4d

let shape = Shape4.of_ints ~n:1 ~h:2 ~w:4 ~c:1

let graph =
  Builder.build ~outputs:Fun.id
    Builder.(
      let* x = input ~shape () in
      let* y = input ~shape () in
      let* a = add x y in
      let* b = mul a a in
      max_dim4 { Ops4_max_dim.Max_dim4.axis = Axis4.W; keepdim = true } b)
  |> Err.or_raise ~pp_error:Builder.pp_error

let inputs =
  List.map2
    (fun id scale ->
      ( id,
        Tensor.materialize (Shape4.to_vec6 shape) (fun c ->
            let h = Dim.to_int (Vec6.get c Axis.H)
            and w = Dim.to_int (Vec6.get c Axis.W) in
            scale *. float_of_int ((h * 4) + w + 1)) ))
    graph.Graph.Graph.inputs [ 0.5; -0.25 ]

let direct =
  Eval_direct4.run graph ~constants:[] ~inputs
  |> Err.or_raise ~pp_error:Eval_direct4.pp_error

let bitwise_equal shape a b =
  let ok = ref true in
  Vec6.iter shape (fun c ->
      if not (Float.equal (Tensor.read a c) (Tensor.read b c)) then ok := false);
  !ok

let%expect_test "each node's program grounds to Direct's tensors" =
  List.iter
    (fun (node : Graph.node) ->
      let p = Eval_symbolic4.node_program graph node in
      let operands = Op.operands node.Graph_common.Node.op in
      Fmt.pr "%s: operands %d, inputs %d, outputs %d@."
        (Op.name node.Graph_common.Node.op)
        (List.length operands)
        (List.length p.Stage_program.inputs)
        (List.length p.Stage_program.outputs);
      let grounded =
        Stage_program.ground p ~bind:(fun id -> Tensor_id.Map.find id direct)
        |> Err.or_raise ~pp_error:Stage_program.pp_error
      in
      List.iter
        (fun oid ->
          let sg = Tensor_id.Map.find oid graph.Graph.Graph.tensors in
          Fmt.pr "  %a bitwise equal: %b@." Tensor_id.pp oid
            (bitwise_equal sg.Tensor_sig.shape
               (Tensor_id.Map.find oid direct)
               (Tensor_id.Map.find oid grounded)))
        node.Graph_common.Node.outputs)
    graph.Graph.Graph.nodes;
  [%expect
    {|
    Add: operands 2, inputs 2, outputs 1
      t2 bitwise equal: true
    Mul: operands 2, inputs 1, outputs 1
      t3 bitwise equal: true
    Max_dim4: operands 1, inputs 1, outputs 2
      t4 bitwise equal: true
      t5 bitwise equal: true |}]

(* The program's boundary names each tensor once, and an intermediate edge is a
   boundary input here even though the whole-graph program would resolve it
   from an earlier stage. *)
let%expect_test "boundary is the node's own deduplicated operands" =
  let whole = Eval_symbolic4.run graph in
  Fmt.pr "whole graph: inputs %d, stages %d@."
    (List.length whole.Stage_program.inputs)
    (List.length whole.Stage_program.stages);
  List.iter
    (fun (node : Graph.node) ->
      let p = Eval_symbolic4.node_program graph node in
      Fmt.pr "%s: inputs %a, kinds %d@."
        (Op.name node.Graph_common.Node.op)
        (Fmt.list ~sep:(Fmt.any ", ") Tensor_id.pp)
        (List.map fst p.Stage_program.inputs)
        (Tensor_id.Map.cardinal p.Stage_program.input_kinds))
    graph.Graph.Graph.nodes;
  [%expect
    {|
    whole graph: inputs 2, stages 3
    Add: inputs t0, t1, kinds 2
    Mul: inputs t2, kinds 0
    Max_dim4: inputs t3, kinds 0 |}]
