(* torch.ops.aten.copy.default through the serialized path. Numeric
   agreement (against real ATen) is checked in
   test/native_bridge/copy_test.ml; this file asserts what
   [Native_interp.lower] builds, mirroring expand_test.ml's [dump] pattern.
   Binds directly to the existing Expand node -- see
   native_interp_lower_shape.ml's own comment for why. *)

open Programs

let copy_node ~non_blocking =
  jstr
    {|{"target":"torch.ops.aten.copy.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"src","arg":%s,"kind":1},{"name":"non_blocking","arg":{"as_bool":%b},"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") (as_tensor "src") non_blocking (as_tensor "y")

let prog ?(x_sizes = [ 3 ]) ?(src_sizes = [ 3 ]) ?(non_blocking = false) () =
  program ~x_sizes
    ~extra_tensor_values:[ ("src", tensor_meta src_sizes) ]
    ~params:[ "src" ]
    ~nodes:[ copy_node ~non_blocking ]
    ~graph_outputs:[ as_tensor "y" ]
    ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

let%expect_test "copy lowers to an Expand node, self's shape as the target" =
  dump "equal shapes" (prog ~x_sizes:[ 3 ] ~src_sizes:[ 3 ] ());
  [%expect
    {|
    equal shapes
    graph
    inputs: [t0 f32 [C=3], t1 f32 [C=3] ->[n0] constant]
    nodes:
      n0: [t2 f32 [C=3]] = expand x=t1 params={size=[C=3]}
    outputs: [t2 f32 [C=3] <-n0] |}]

(* [non_blocking] is read-and-discarded -- same graph, either spelling, the
   same proof [expand_test.ml] makes for [implicit]. *)
let%expect_test "copy: non_blocking carries no computational effect" =
  dump "non_blocking=false" (prog ~non_blocking:false ());
  dump "non_blocking=true" (prog ~non_blocking:true ());
  [%expect
    {|
    non_blocking=false
    graph
    inputs: [t0 f32 [C=3], t1 f32 [C=3] ->[n0] constant]
    nodes:
      n0: [t2 f32 [C=3]] = expand x=t1 params={size=[C=3]}
    outputs: [t2 f32 [C=3] <-n0]
    non_blocking=true
    graph
    inputs: [t0 f32 [C=3], t1 f32 [C=3] ->[n0] constant]
    nodes:
      n0: [t2 f32 [C=3]] = expand x=t1 params={size=[C=3]}
    outputs: [t2 f32 [C=3] <-n0] |}]

(* The genuine broadcast case: self's declared shape has extent 2 on W where
   src has 1 -- no corpus occurrence exercises this, but the translation is
   the fully general broadcast. *)
let%expect_test "copy broadcasts src to self's shape" =
  dump "broadcast W" (prog ~x_sizes:[ 2; 3 ] ~src_sizes:[ 1; 3 ] ());
  [%expect
    {|
    broadcast W
    graph
    inputs: [t0 f32 [W=2 C=3], t1 f32 [C=3] ->[n0] constant]
    nodes:
      n0: [t2 f32 [W=2 C=3]] = expand x=t1 params={size=[W=2 C=3]}
    outputs: [t2 f32 [W=2 C=3] <-n0] |}]

(* Not [Aten_shape]'s business -- self's axis is 5, src's is 2, incompatible
   on W. [Pointwise.Expand.output_shape]'s own axis-wise check catches it at
   graph-build time. *)
let%expect_test "copy rejects a non-broadcastable src" =
  dump "non-broadcastable" (prog ~x_sizes:[ 5; 3 ] ~src_sizes:[ 2; 3 ] ());
  [%expect
    {|
    non-broadcastable
      incompatible broadcast extents on axis W: 2 vs 5 |}]
