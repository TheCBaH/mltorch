(* torch.ops.aten.abs.default through the serialized path: one dtype-preserving
   [Abs] node. Numeric agreement with ATen, including signed zero and the int64
   wrap at min_int, is checked in test/native_bridge/abs_test.ml. *)

open Programs

let abs_node =
  jstr
    {|{"target":"torch.ops.aten.abs.default","inputs":[{"name":"self","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") (as_tensor "y")

let prog ~x_sizes =
  program ~x_sizes ~nodes:[ abs_node ] ~graph_outputs:[ as_tensor "y" ] ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

let%expect_test "abs.default lowers to one Abs node" =
  dump "float, rank 2:" (prog ~x_sizes:[ 1; 24 ]);
  dump "float, rank 3:" (prog ~x_sizes:[ 1; 1568; 384 ]);
  [%expect
    {|
    float, rank 2:
    graph
    inputs: [t0 f32 [C=24] ->[n0]]
    nodes:
      n0: [t1 f32 [C=24]] = abs x=t0
    outputs: [t1 f32 [C=24] <-n0]
    float, rank 3:
    graph
    inputs: [t0 f32 [W=1568 C=384] ->[n0]]
    nodes:
      n0: [t1 f32 [W=1568 C=384]] = abs x=t0
    outputs: [t1 f32 [W=1568 C=384] <-n0] |}]
