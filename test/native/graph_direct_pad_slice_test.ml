(* Split out of what was graph_test.ml see graph_direct_fixtures.ml. *)

open Graph_ir
open Graph_direct_fixtures

let%expect_test "Direct graph: pad reflect on W, constant fill on H" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"pad" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 3 1) ~name:"x" () in
          pad ~name:"out"
            {
              Pad.Pad.pads = [ (Axis.W, { Pad.Pad.before = 1; after = 1 }) ];
              mode = Pad.Pad.Reflect;
            }
            x)
    in
    Format.printf "%a@." Graph_ir.pp g;
    let x =
      Tensor.materialize (s 1 1 1 2 3 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 10)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    graph
    inputs: [t0 f32 [H=2 W=3 C=1] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=5 C=1]] = pad x=t0 params={pads=[W:1,1] mode=reflect}
    outputs: [t1 f32 [H=2 W=5 C=1] <-n0]
    out = tensor f32 [H=2 W=5 C=1] {1, 0, 1, 2, 1, 11, 10, 11, ...} |}]

let%expect_test "Direct graph: a pad that empties an axis is not buildable" =
  let result =
    let open Err.Syntax in
    let+ g =
      lift_build
        Graph_builder.(
          build ~name:"pad" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 3 1) ~name:"x" () in
          pad ~name:"out"
            {
              Pad.Pad.pads = [ (Axis.H, { Pad.Pad.before = -2; after = 0 }) ];
              mode = Pad.Pad.Constant 0.;
            }
            x)
    in
    Format.asprintf "%a" Graph_ir.pp g
  in
  Format.printf "%a@." (pp_result Format.pp_print_string) result;
  [%expect
    {| pad of axis H by (-2, 0) over extent 2 leaves 0 elements; the engine has no empty extent |}]

(* Slice through the builder. The bounds are canonical by contract, so the two
   graph-level facts worth pinning are that the shape rule sees the input edge
   (and so refuses an empty or out-of-range selection at BUILD time, where an
   importer's own check would already have run) and that the rank survives. *)
let%expect_test "Direct graph: slice narrows one axis and keeps the rank" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"slice" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 5 1) ~name:"x" () in
          slice ~name:"out"
            {
              Split.Slice.axis = Axis.W;
              start = 1;
              stop = 5;
              step = Op_config.Pos.of_int 2;
            }
            x)
    in
    Format.printf "%a@." Graph_ir.pp g;
    let x =
      Tensor.materialize (s 1 1 1 2 5 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 10)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    graph
    inputs: [t0 f32 [H=2 W=5 C=1] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=2 C=1]] =
        slice x=t0 params={axis=W start=1 stop=5 step=2}
    outputs: [t1 f32 [H=2 W=2 C=1] <-n0]
    out = tensor f32 [H=2 W=2 C=1] {1, 3, 11, 13} |}]

(* Select drops the axis it picks along, unlike Slice above, so the graph's
   output rank is one less than the input's -- the fact worth pinning here is
   that a single [Select] node appears, not a [Slice]+[Reshape]
   pair. *)
let%expect_test "Direct graph: select drops one axis" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"select" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 5 1) ~name:"x" () in
          select ~name:"out" { Split.Select.axis = Axis.W; index = 3 } x)
    in
    Format.printf "%a@." Graph_ir.pp g;
    let x =
      Tensor.materialize (s 1 1 1 2 5 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 10)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    graph
    inputs: [t0 f32 [H=2 W=5 C=1] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=1]] = select x=t0 params={axis=W index=3}
    outputs: [t1 f32 [W=2 C=1] <-n0]
    out = tensor f32 [W=2 C=1] {3, 13} |}]

(* Select_scatter's write-back counterpart: two graph inputs ([self], [src]),
   one node, output rank equal to [self]'s (unlike [Select] above, which
   drops a rank). *)
let%expect_test
    "Direct graph: select_scatter writes src at index and keeps self elsewhere"
    =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"select_scatter" ~outputs:(fun r -> [ r ])
          @@
          let* self = input ~shape:(s 1 1 1 2 5 1) ~name:"self" () in
          let* src = input ~shape:(s 1 1 1 1 5 1) ~name:"src" () in
          select_scatter ~name:"out"
            { Split.Select_scatter.axis = Axis.H; index = 1 }
            ~self ~src)
    in
    Format.printf "%a@." Graph_ir.pp g;
    let self =
      Tensor.materialize (s 1 1 1 2 5 1) (fun c ->
          float_of_int
            ((Dim.to_int (Vec6.get c Axis.H) * 10)
            + Dim.to_int (Vec6.get c Axis.W)))
    in
    let src =
      Tensor.materialize (s 1 1 1 1 5 1) (fun c ->
          float_of_int (900 + Dim.to_int (Vec6.get c Axis.W)))
    in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ self; src ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    graph
    inputs: [t0 f32 [H=2 W=5 C=1] ->[n0], t1 f32 [W=5 C=1] ->[n0]]
    nodes:
      n0: [t2 f32 [H=2 W=5 C=1]] =
        select_scatter self=t0 src=t1 params={axis=H index=1}
    outputs: [t2 f32 [H=2 W=5 C=1] <-n0]
    out = tensor f32 [H=2 W=5 C=1] {0, 1, 2, 3, 4, 900, 901, 902, ...} |}]

let%expect_test "Direct graph: select_scatter refuses a src of the wrong shape"
    =
  let result =
    let open Err.Syntax in
    let+ g =
      lift_build
        Graph_builder.(
          build ~name:"select_scatter" ~outputs:(fun r -> [ r ])
          @@
          let* self = input ~shape:(s 1 1 1 2 5 1) ~name:"self" () in
          let* src = input ~shape:(s 1 1 1 1 4 1) ~name:"src" () in
          select_scatter ~name:"out"
            { Split.Select_scatter.axis = Axis.H; index = 1 }
            ~self ~src)
    in
    Format.asprintf "%a" Graph_ir.pp g
  in
  Format.printf "%a@." (pp_result Format.pp_print_string) result;
  [%expect
    {| select_scatter src shape must be [W=5 C=1] (axis=H index=1), got [W=4 C=1] |}]

let%expect_test "Direct graph: an empty slice is not buildable" =
  let result =
    let open Err.Syntax in
    let+ g =
      lift_build
        Graph_builder.(
          build ~name:"slice" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 3 1) ~name:"x" () in
          slice ~name:"out"
            {
              Split.Slice.axis = Axis.W;
              start = 2;
              stop = 2;
              step = Op_config.Pos.of_int 1;
            }
            x)
    in
    Format.asprintf "%a" Graph_ir.pp g
  in
  Format.printf "%a@." (pp_result Format.pp_print_string) result;
  [%expect
    {| slice of axis W [2, 2) step 1 over extent 3 selects 0 elements; the engine has no empty extent |}]
