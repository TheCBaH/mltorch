(* torch.ops.aten._assert_tensor_metadata.default through the serialized
   path. Dispatch against a real ATen tensor is checked in
   test/native_bridge/shape_ops_test.ml; this file asserts what
   [Native_interp.lower] builds, mirroring alias_test.ml's [dump] pattern. *)

open Programs

let assert_node =
  jstr
    {|{"target":"torch.ops.aten._assert_tensor_metadata.default","inputs":[{"name":"a","arg":%s,"kind":1}],"outputs":[],"metadata":{}}|}
    (as_tensor "x")

(* ATen's own schema returns no [Tensor] ([_assert_tensor_metadata(...) ->
   ()]), so unlike every other fixture in this directory there is no output
   tensor to name as the program's own [graph_outputs] -- [x] is consumed
   ONLY by the assertion, whose entire effect is routing it to [Discard]. *)
let prog ~x_sizes = program ~x_sizes ~nodes:[ assert_node ] ~graph_outputs:[] ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

(* No argument to decode besides [a] and no shape check that can fail --
   [size]/[stride]/[dtype]/[device]/[layout] all restate facts already true
   of [a] and are never read (see the dispatch arm's own comment in
   native_interp_lower_shape.ml), so unlike view_test.ml's coverage there is
   no rejection path to pin, only the [Discard] sink it routes to at a few
   ranks. *)
let%expect_test
    "_assert_tensor_metadata.default lowers to a Discard sink, several ranks" =
  dump "rank 2:" (prog ~x_sizes:[ 2; 3 ]);
  dump "rank 1:" (prog ~x_sizes:[ 6 ]);
  dump "rank 4:" (prog ~x_sizes:[ 1; 2; 3; 4 ]);
  [%expect
    {|
    rank 2:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [] = discard x=t0
    outputs: []
    rank 1:
    graph
    inputs: [t0 f32 [C=6] ->[n0]]
    nodes:
      n0: [] = discard x=t0
    outputs: []
    rank 4:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [] = discard x=t0
    outputs: [] |}]
