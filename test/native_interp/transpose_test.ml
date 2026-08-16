(* torch.ops.aten.transpose.int through the serialized path.

   No model in this repo's zoo serializes transpose.int (op3-impl.md F3), so
   these fixtures are the coverage, not a supplement to it. Numeric agreement
   is checked in test/native_bridge_test.ml; this file asserts what
   [Native_interp.lower] builds, mirroring conv_test.ml's [dump] pattern. *)

open Programs

let transpose_node ~dim0 ~dim1 =
  jstr
    {|{"target":"torch.ops.aten.transpose.int","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"dim0","arg":{"as_int":%d},"kind":1},{"name":"dim1","arg":{"as_int":%d},"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") dim0 dim1 (as_tensor "y")

let prog ~x_sizes ~dim0 ~dim1 =
  program ~x_sizes
    ~nodes:[ transpose_node ~dim0 ~dim1 ]
    ~graph_outputs:[ as_tensor "y" ]
    ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

let%expect_test "transpose.int rank-2 swaps W and C" =
  dump "(0,1):" (prog ~x_sizes:[ 3; 4 ] ~dim0:0 ~dim1:1);
  [%expect
    {|
    (0,1):
    graph
    inputs: [t0 f32 [W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [W=4 C=3]] = permute x=t0 perm=[W<-C, C<-W]
    outputs: [t1 f32 [W=4 C=3] <-n0] |}]

let%expect_test
    "transpose.int negative dims (-1,-2) name the same pair as the positive \
     spelling" =
  dump "(-1,-2):" (prog ~x_sizes:[ 3; 4 ] ~dim0:(-1) ~dim1:(-2));
  dump "(0,1):" (prog ~x_sizes:[ 3; 4 ] ~dim0:0 ~dim1:1);
  [%expect
    {|
    (-1,-2):
    graph
    inputs: [t0 f32 [W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [W=4 C=3]] = permute x=t0 perm=[W<-C, C<-W]
    outputs: [t1 f32 [W=4 C=3] <-n0]
    (0,1):
    graph
    inputs: [t0 f32 [W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [W=4 C=3]] = permute x=t0 perm=[W<-C, C<-W]
    outputs: [t1 f32 [W=4 C=3] <-n0] |}]

let%expect_test "transpose.int rank-3 (1,2) swaps the middle two axes" =
  dump "(1,2):" (prog ~x_sizes:[ 2; 3; 4 ] ~dim0:1 ~dim1:2);
  [%expect
    {|
    (1,2):
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=4 C=3]] = permute x=t0 perm=[W<-C, C<-W]
    outputs: [t1 f32 [H=2 W=4 C=3] <-n0] |}]

let%expect_test "transpose.int rank-4 mixed dims (0,-1)" =
  dump "(0,-1):" (prog ~x_sizes:[ 2; 3; 4; 5 ] ~dim0:0 ~dim1:(-1));
  [%expect
    {|
    (0,-1):
    graph
    inputs: [t0 f32 [D=2 H=3 W=4 C=5] ->[n0]]
    nodes:
      n0: [t1 f32 [D=5 H=3 W=4 C=2]] = permute x=t0 perm=[D<-C, C<-D]
    outputs: [t1 f32 [D=5 H=3 W=4 C=2] <-n0] |}]

(* Equal dims: a real identity transpose, not special-cased away -- and
   duplicate dims after normalization ((1,-3) on a rank-4 tensor both name
   axis 1) is the same well-defined case, not a rejection. *)
let%expect_test
    "transpose.int equal dims, and duplicates after normalization, are the \
     identity" =
  dump "(1,1):" (prog ~x_sizes:[ 3; 4 ] ~dim0:1 ~dim1:1);
  dump "(1,-3):" (prog ~x_sizes:[ 1; 3; 4; 5 ] ~dim0:1 ~dim1:(-3));
  [%expect
    {|
    (1,1):
    graph
    inputs: [t0 f32 [W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [W=3 C=4]] = permute x=t0 perm=[]
    outputs: [t1 f32 [W=3 C=4] <-n0]
    (1,-3):
    graph
    inputs: [t0 f32 [H=3 W=4 C=5] ->[n0]]
    nodes:
      n0: [t1 f32 [H=3 W=4 C=5]] = permute x=t0 perm=[]
    outputs: [t1 f32 [H=3 W=4 C=5] <-n0] |}]

let%expect_test "transpose.int rejects an out-of-range dim" =
  dump "dim1=5:" (prog ~x_sizes:[ 3; 4 ] ~dim0:0 ~dim1:5);
  [%expect
    {|
    dim1=5:
      malformed PT2 graph: invalid dimension 5 for rank 2 |}]

let%expect_test "transpose.int rank-6 (0,1)" =
  dump "(0,1):" (prog ~x_sizes:[ 2; 3; 1; 1; 1; 1 ] ~dim0:0 ~dim1:1);
  [%expect
    {|
    (0,1):
    graph
    inputs: [t0 f32 [N=2 T=3 D=1 H=1 W=1 C=1] ->[n0]]
    nodes:
      n0: [t1 f32 [N=3 T=2 D=1 H=1 W=1 C=1]] = permute x=t0 perm=[N<-T, T<-N]
    outputs: [t1 f32 [N=3 T=2 D=1 H=1 W=1 C=1] <-n0] |}]
