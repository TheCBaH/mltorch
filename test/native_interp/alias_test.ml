(* torch.ops.aten.alias.default through the serialized path. Numeric
   agreement against ATen is checked in shape_ops_test.ml; this file asserts
   what [Native_interp.lower] builds, mirroring view_test.ml's [dump]
   pattern. *)

open Programs

let alias_node =
  jstr
    {|{"target":"torch.ops.aten.alias.default","inputs":[{"name":"self","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") (as_tensor "y")

let prog ~x_sizes =
  program ~x_sizes ~nodes:[ alias_node ] ~graph_outputs:[ as_tensor "y" ] ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

(* No argument to decode besides [self] and no shape check that can fail --
   [alias.default] always legalizes to the existing [Clone] node (paramless,
   shape-preserving; see op_bridge_shape.ml's arm for why [Clone] rather than
   [Reshape]), so unlike view_test.ml's coverage there is no rejection path to
   pin, only the identity node it builds at a few ranks. *)
let%expect_test "alias.default lowers to an identity Clone, several ranks" =
  dump "rank 2:" (prog ~x_sizes:[ 2; 3 ]);
  dump "rank 1:" (prog ~x_sizes:[ 6 ]);
  dump "rank 4:" (prog ~x_sizes:[ 1; 2; 3; 4 ]);
  [%expect
    {|
    rank 2:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = clone x=t0
    outputs: [t1 f32 [W=2 C=3] <-n0]
    rank 1:
    graph
    inputs: [t0 f32 [C=6] ->[n0]]
    nodes:
      n0: [t1 f32 [C=6]] = clone x=t0
    outputs: [t1 f32 [C=6] <-n0]
    rank 4:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=3 C=4]] = clone x=t0
    outputs: [t1 f32 [H=2 W=3 C=4] <-n0] |}]
