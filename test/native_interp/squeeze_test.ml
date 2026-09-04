(* torch.ops.aten.squeeze.dim through the serialized path. Numeric agreement
   (against real ATen) is checked in test/native_bridge/squeeze_test.ml; this
   file asserts what [Native_interp.lower] builds, mirroring copy_test.ml's
   [dump] pattern. Binds directly to the existing Reshape node -- see
   native_interp_lower_shape.ml's own comment for why. *)

open Programs

let squeeze_node ~dim =
  jstr
    {|{"target":"torch.ops.aten.squeeze.dim","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"dim","arg":{"as_int":%d},"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") dim (as_tensor "y")

let prog ~x_sizes ~dim =
  program ~x_sizes
    ~nodes:[ squeeze_node ~dim ]
    ~graph_outputs:[ as_tensor "y" ]
    ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

let%expect_test "squeeze drops a unit axis, reshaping to the shorter shape" =
  dump "unit axis" (prog ~x_sizes:[ 2; 1; 3 ] ~dim:1);
  [%expect
    {|
    unit axis
    graph
    inputs: [t0 f32 [H=2 W=1 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = reshape x=t0 params={shape=[W=2 C=3]}
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

(* The corpus's own configuration: negative [dim] resolving to the same unit
   axis as above, read against the SERIALIZED shape rather than a live
   tensor. *)
let%expect_test "squeeze, negative dim" =
  dump "negative dim" (prog ~x_sizes:[ 2; 1; 3 ] ~dim:(-2));
  [%expect
    {|
    negative dim
    graph
    inputs: [t0 f32 [H=2 W=1 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = reshape x=t0 params={shape=[W=2 C=3]}
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

(* Real ATen leaves [self] UNCHANGED (same rank) when the declared extent is
   not 1 -- a [Reshape] to the SAME shape, not a rejection. *)
let%expect_test "squeeze is a no-op on a non-unit axis" =
  dump "non-unit axis" (prog ~x_sizes:[ 2; 3 ] ~dim:0);
  [%expect
    {|
    non-unit axis
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = reshape x=t0 params={shape=[W=2 C=3]}
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

(* [dim] out of the operand's own rank is refused with the same typed row
   [select.int]/[select_scatter.default] report for the identical fault --
   squeeze never widens the valid range the way [unsqueeze.default]'s
   rank+1 does. *)
let%expect_test "squeeze rejects a dim outside self's rank" =
  dump "out of range" (prog ~x_sizes:[ 2; 3 ] ~dim:2);
  [%expect
    {|
    out of range
      malformed PT2 graph: invalid dimension 2 for rank 2 |}]
