(* Split out of what was graph_test.ml see graph_direct_fixtures.ml. *)

open Graph_ir
open Graph_direct_fixtures

let pp_discard ppf (g, out, dead) =
  Format.fprintf ppf "%a@.out = %a@.dead = %a" Graph_ir.pp g Tensor.pp out
    Tensor.pp dead

let%expect_test "Direct graph: Discard sink (dead edge still materialised)" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"discard" ~outputs:(fun (out, _) -> [ out ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          let* out = add ~name:"out" a b in
          let* dead = mul ~name:"dead" a b in
          let* () = discard dead in
          return (out, dead))
    in
    let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c)) in
    let b = Tensor.materialize (s1c 3) (fun _ -> 10.) in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    let* out = tensor_of_name g env "out" in
    let* dead = tensor_of_name g env "dead" in
    Err.return (g, out, dead)
  in
  Format.printf "%a@." (pp_result pp_discard) result;
  [%expect
    {|
    graph
    inputs: [t0 f32 [C=3] ->[n0, n1], t1 f32 [C=3] ->[n0, n1]]
    nodes:
      n0: [t2 f32 [C=3]] = add a=t0 b=t1
      n1: [t3 f32 [C=3] ->[n2]] = mul a=t0 b=t1
      n2: [] = discard x=t3 <-n1
    outputs: [t2 f32 [C=3] <-n0]
    out = tensor f32 [C=3] {10, 11, 12}
    dead = tensor f32 [C=3] {0, 10, 20} |}]

(* Batch norm applies per-channel affine using running stats: with mean=[1,5],
   var=[4,4] (inv=0.5) and weight/bias [2,10]/[1,-1], y = (x-mean)*0.5*w+b. *)
let%expect_test "Direct graph: reshape [H=2 W=3 C=1] -> [C=6] (flatten)" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"reshape" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 3 1) ~name:"x" () in
          reshape ~name:"out" { Reshape.Reshape.shape = s 1 1 1 1 1 6 } x)
    in
    let x =
      Tensor.materialize (s 1 1 1 2 3 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 3)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=6] {0, 1, 2, 3, 4, 5} |}]

let%expect_test "Direct graph: repeat [C=2] by [C:3] wraps" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"repeat" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 2) ~name:"x" () in
          repeat ~name:"out" { Repeat.Repeat.repeats = s1c 3 } x)
    in
    let x = Tensor.materialize (s1c 2) (fun c -> float_of_int (chan c)) in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=6] {0, 1, 0, 1, 0, 1} |}]

let%expect_test
    "Direct graph: repeat_interleave [C=2] by 3 duplicates each element" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"repeat_interleave" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 2) ~name:"x" () in
          repeat_interleave ~name:"out"
            {
              Repeat.RepeatInterleave.axis = Axis.C;
              repeats = Op_config.Pos.of_int 3;
            }
            x)
    in
    let x = Tensor.materialize (s1c 2) (fun c -> float_of_int (chan c)) in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=6] {0, 0, 0, 1, 1, 1} |}]

let%expect_test "Direct graph: unfold [C=8] onto W, overlapping windows" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"unfold" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 8) ~name:"x" () in
          unfold ~name:"out"
            {
              Unfold.Unfold.axis = Axis.W;
              size = Dim.extent 3;
              step = Op_config.Pos.of_int 2;
            }
            x)
    in
    let x = Tensor.materialize (s1c 8) (fun c -> float_of_int (chan c)) in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [W=3 C=3] {0, 1, 2, 2, 3, 4, 4, 5, ...} |}]

(* Conv decomposition. The input is laid out NCHW: in the 6D frame its channel sits
   on H, spatial-H on W, spatial-W on C. Two permutes bracket a native (NHWC)
   conv: NCHW->NHWC moves the channel to C and the spatial axes to H/W; NHWC->NCHW
   is its inverse. *)
