(* torch.ops.aten.expand.default through the serialized path. Numeric
   agreement (against real ATen) is checked in
   test/native_bridge/expand_test.ml; this file asserts what
   [Native_interp.lower] builds, mirroring view_test.ml's [dump] pattern. *)

open Programs

let expand_node ~size ~implicit =
  jstr
    {|{"target":"torch.ops.aten.expand.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"size","arg":{"as_ints":[%s]},"kind":1},{"name":"implicit","arg":{"as_bool":%b},"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x")
    (String.concat "," (List.map string_of_int size))
    implicit (as_tensor "y")

let prog ?(x_sizes = [ 1; 3 ]) ?(implicit = false) size =
  program ~x_sizes
    ~nodes:[ expand_node ~size ~implicit ]
    ~graph_outputs:[ as_tensor "y" ]
    ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

let%expect_test "expand lowers to an Expand node" =
  dump "broadcast W" (prog ~x_sizes:[ 1; 3 ] [ 2; 3 ]);
  [%expect
    {|
    broadcast W
    graph
    inputs: [t0 f32 [C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = expand x=t0 params={size=[W=2 C=3]}
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

(* [implicit] is read-and-discarded -- same graph, either spelling. *)
let%expect_test "expand: implicit carries no computational effect" =
  dump "implicit=false" (prog ~x_sizes:[ 1; 3 ] ~implicit:false [ 2; 3 ]);
  dump "implicit=true" (prog ~x_sizes:[ 1; 3 ] ~implicit:true [ 2; 3 ]);
  [%expect
    {|
    implicit=false
    graph
    inputs: [t0 f32 [C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = expand x=t0 params={size=[W=2 C=3]}
    outputs: [t1 f32 [W=2 C=3] <-n0]
    implicit=true
    graph
    inputs: [t0 f32 [C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = expand x=t0 params={size=[W=2 C=3]}
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

(* [-1] keeps self's own extent; [size] one longer than self's rank ADDS a
   leading axis. *)
let%expect_test
    "expand: -1 keeps self's extent, a longer size adds a leading axis" =
  dump "-1 keeps W" (prog ~x_sizes:[ 1; 3 ] [ 2; -1 ]);
  dump "add leading axis" (prog ~x_sizes:[ 3 ] [ 2; 3 ]);
  [%expect
    {|
    -1 keeps W
    graph
    inputs: [t0 f32 [C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = expand x=t0 params={size=[W=2 C=3]}
    outputs: [t1 f32 [W=2 C=3] <-n0]
    add leading axis
    graph
    inputs: [t0 f32 [C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = expand x=t0 params={size=[W=2 C=3]}
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

(* ATen's own [TORCH_CHECK]: [size] must have at least as many entries as
   self's rank. *)
let%expect_test "expand: rejects a size shorter than self's rank" =
  dump "too short" (prog ~x_sizes:[ 2; 3 ] [ 3 ]);
  [%expect
    {|
    too short
      malformed PT2 graph: expand size [3]: expand size [3] must have at least as many entries as self's shape [2, 3] (rank 2) |}]

(* ATen's other own [TORCH_CHECK]: a [-1] naming a leading position beyond
   self's rank has nothing to copy. *)
let%expect_test "expand: rejects a leading -1" =
  dump "leading -1" (prog ~x_sizes:[ 3 ] [ -1; 3 ]);
  [%expect
    {|
    leading -1
      malformed PT2 graph: expand size [-1, 3]: expand size [-1, 3]: -1 at position 0 is not allowed for a leading dimension self (shape [3]) does not have |}]

(* Not [Aten_shape]'s business -- self's axis is 2, not 1, so it cannot
   broadcast to 5. [Pointwise.Expand.output_shape]'s own axis-wise check
   catches it at graph-build time (a [`Build] row, not [malformed] -- the
   same category every other shape-rule rejection reachable through
   [Graph_builder] falls into), same as [Op_bridge]'s. *)
let%expect_test "expand: rejects a non-broadcastable target" =
  dump "non-broadcastable" (prog ~x_sizes:[ 2; 3 ] [ 5; 3 ]);
  [%expect
    {|
    non-broadcastable
      incompatible broadcast extents on axis W: 2 vs 5 |}]
