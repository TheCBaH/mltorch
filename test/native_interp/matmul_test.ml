(* torch.ops.aten.matmul.default through the serialized path. Numeric
   agreement (against real ATen) is checked in
   test/native_bridge/dispatch_test.ml; this file asserts what
   [Native_interp.lower] builds, mirroring binary_test.ml's [dump] pattern --
   same split between the two shape families as [Op_bridge]'s own arm
   (`.ai/matmul_softmax_design.md` §4-5). *)

open Programs

let matmul_node () =
  jstr
    {|{"target":"torch.ops.aten.matmul.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"other","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") (as_tensor "other") (as_tensor "y")

let prog ~x_sizes ~other_sizes () =
  program ~x_sizes
    ~extra_tensor_values:[ ("other", tensor_meta other_sizes) ]
    ~params:[ "other" ]
    ~nodes:[ matmul_node () ]
    ~graph_outputs:[ as_tensor "y" ]
    ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

let%expect_test "matmul.default 2D @ 2D binds to the existing Bmm node" =
  dump "rank-2:" (prog ~x_sizes:[ 2; 3 ] ~other_sizes:[ 3; 2 ] ());
  [%expect
    {|
    rank-2:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0], t1 f32 [W=3 C=2] ->[n0] constant]
    nodes:
      n0: [t2 f32 [W=2 C=2]] = bmm input=t0 mat2=t1
    outputs: [t2 f32 [W=2 C=2] <-n0] |}]

(* The batched/multi-head shape family (`D`/`H` > 1) binds to the new
   [Batched_matmul] node, not a decomposition or a relayout permute. *)
let%expect_test
    "matmul.default batched (D>1, H>1) binds to the new Batched_matmul node" =
  dump "rank-4:" (prog ~x_sizes:[ 2; 2; 2; 3 ] ~other_sizes:[ 2; 2; 3; 2 ] ());
  [%expect
    {|
    rank-4:
    graph
    inputs:
      [t0 f32 [D=2 H=2 W=2 C=3] ->[n0], t1 f32 [D=2 H=2 W=3 C=2] ->[n0] constant]
    nodes:
      n0: [t2 f32 [D=2 H=2 W=2 C=2]] = batched_matmul input=t0 mat2=t1
    outputs: [t2 f32 [D=2 H=2 W=2 C=2] <-n0] |}]

(* Unequal ATen rank is accepted too -- [Batched_matmul] reads each operand's
   frame independently, and right-alignment already pads the lower-rank
   operand's missing leading axes to 1, exactly ATen's own implicit-unsqueeze
   rule (`eca_halonext26ts`'s real `[N,H,W,C] @ [W,C]` occurrence,
   `.ai/matmul_softmax_design.md` §5). Rank<2 on either side is the one
   remaining typed rejection (§6), covered by the metadata-only importer's
   own arm the same way as [Op_bridge]'s. *)
let%expect_test "matmul.default accepts unequal operand ranks" =
  dump "rank mismatch:" (prog ~x_sizes:[ 2; 2; 3 ] ~other_sizes:[ 3; 2 ] ());
  [%expect
    {|
    rank mismatch:
    graph
    inputs: [t0 f32 [H=2 W=2 C=3] ->[n0], t1 f32 [W=3 C=2] ->[n0] constant]
    nodes:
      n0: [t2 f32 [H=2 W=2 C=2]] = batched_matmul input=t0 mat2=t1
    outputs: [t2 f32 [H=2 W=2 C=2] <-n0] |}]

(* Rank<2 on either side is still explicitly out of scope
   (`.ai/matmul_softmax_design.md` §6, a different ATen output-rank rule with
   no corpus evidence) -- a typed rejection naming both actual shapes,
   matching [Op_bridge]'s own diagnostic. *)
let%expect_test "matmul.default rejects a rank<2 operand" =
  dump "rank too low:" (prog ~x_sizes:[ 2; 3 ] ~other_sizes:[ 3 ] ());
  [%expect
    {|
    rank too low:
      malformed PT2 graph: matmul.default: both operands must be rank>=2, got self=[2, 3] other=[3] |}]
