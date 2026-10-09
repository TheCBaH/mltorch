(* torch.ops.aten.detach.default through the serialized path. It severs autograd
   history and nothing else, so an exported inference graph sees the functional
   identity: one [Clone] node, like [alias.default]. Numeric agreement with ATen
   is checked in test/native_bridge/shape_ops_test.ml. *)

open Programs

let detach_node =
  jstr
    {|{"target":"torch.ops.aten.detach.default","inputs":[{"name":"self","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") (as_tensor "y")

let prog ~x_sizes =
  program ~x_sizes ~nodes:[ detach_node ] ~graph_outputs:[ as_tensor "y" ] ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

let%expect_test "detach.default lowers to an identity Clone, several ranks" =
  dump "rank 3:" (prog ~x_sizes:[ 1; 1568; 384 ]);
  dump "rank 1:" (prog ~x_sizes:[ 6 ]);
  [%expect
    {|
    rank 3:
    graph
    inputs: [t0 f32 [W=1568 C=384] ->[n0]]
    nodes:
      n0: [t1 f32 [W=1568 C=384]] = clone x=t0
    outputs: [t1 f32 [W=1568 C=384] <-n0]
    rank 1:
    graph
    inputs: [t0 f32 [C=6] ->[n0]]
    nodes:
      n0: [t1 f32 [C=6]] = clone x=t0
    outputs: [t1 f32 [C=6] <-n0] |}]
