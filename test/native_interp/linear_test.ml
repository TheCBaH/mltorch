(* The exact `linear.default` overload through the serialized path.

   As for conv2d.default (see conv_test.ml), these fixtures are the whole
   coverage: every downloadable model is exported post-decomposition and carries
   `addmm.default` instead, so no `interp_*_cram.t` reaches this arm.

   The two arms build the SAME [Linear] IR node and cannot be merged. In
   `addmm`, [self] is the bias and is required, and [mat2] is [In, Out]; here the
   bias is optional and [weight] is [Out, In] -- the transpose. One permutation
   for one of them applied to the other produces a weight whose output and input
   axes are swapped, which is why the fixtures below are all NON-SQUARE: a square
   weight makes that mutation unfalsifiable. *)

open Programs

let linear ?(bias = `Absent) () =
  let bias_arg =
    match bias with
    | `Absent -> ""
    | `None -> {|,{"name":"bias","arg":{"as_none":true},"kind":1}|}
    | `Tensor -> jstr {|,{"name":"bias","arg":%s,"kind":1}|} (as_tensor "b")
  in
  jstr
    {|{"target":"torch.ops.aten.linear.default","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"weight","arg":%s,"kind":1}%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") (as_tensor "w") bias_arg (as_tensor "y")

(* [w_sizes] is ATen's [out_features, in_features]. *)
let prog ?(x_sizes = [ 2; 3 ]) ?(w_sizes = [ 5; 3 ]) ?bias_size node =
  let bias = match bias_size with None -> [] | Some n -> [ ("b", [ n ]) ] in
  let captured = ("w", w_sizes) :: bias in
  program ~x_sizes
    ~extra_tensor_values:(List.map (fun (n, s) -> (n, tensor_meta s)) captured)
    ~params:(List.map fst captured) ~nodes:[ node ]
    ~graph_outputs:[ as_tensor "y" ]
    ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Graph_ir.pp Format.std_formatter l.Pt2_native_graph.graph

let%expect_test "linear.default lowers a non-square weight" =
  dump "no bias:" (prog (linear ()));
  [%expect
    {|
    no bias:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n1], t1 f32 [W=5 C=3] ->[n0] constant]
    nodes:
      group g1 torch.ops.aten.linear.default:
        n0: [t2 f32 [N=5 T=1 D=1 H=1 W=1 C=3] ->[n1]] =
          permute x=t1 perm=[N<-W, W<-N]
        n1: [t3 f32 [W=2 C=5]] =
          linear x=t0 weight=t2 <-n0 bias=none params={in_features=3}
    outputs: [t3 f32 [W=2 C=5] <-n1] |}]

let%expect_test "bias absent and explicit None reach the same graph" =
  show "absent:" (prog (linear ()));
  show "explicit none:" (prog (linear ~bias:`None ()));
  show "tensor:" (prog ~bias_size:5 (linear ~bias:`Tensor ()));
  [%expect
    {|
    absent:                    lowered, nodes=2
    explicit none:             lowered, nodes=2
    tensor:                    lowered, nodes=2 |}]

(* Only the trailing extent is the feature axis; every leading axis passes
   through. A rank-2 fixture alone cannot show that, because there is only one
   leading axis and nothing to preserve. *)
let%expect_test "leading axes pass through, whatever the input rank" =
  dump "rank 4 input:" (prog ~x_sizes:[ 2; 7; 4; 3 ] (linear ()));
  [%expect
    {|
    rank 4 input:
    graph
    inputs: [t0 f32 [D=2 H=7 W=4 C=3] ->[n1], t1 f32 [W=5 C=3] ->[n0] constant]
    nodes:
      group g1 torch.ops.aten.linear.default:
        n0: [t2 f32 [N=5 T=1 D=1 H=1 W=1 C=3] ->[n1]] =
          permute x=t1 perm=[N<-W, W<-N]
        n1: [t3 f32 [D=2 H=7 W=4 C=5]] =
          linear x=t0 weight=t2 <-n0 bias=none params={in_features=3}
    outputs: [t3 f32 [D=2 H=7 W=4 C=5] <-n1] |}]

(* ---- rejections --------------------------------------------------------- *)

let%expect_test "a non-rank-two weight is refused" =
  show "rank 3 weight:" (prog ~w_sizes:[ 5; 3; 1 ] (linear ()));
  show "rank 1 weight:" (prog ~w_sizes:[ 5 ] (linear ()));
  [%expect
    {|
    rank 3 weight:             malformed PT2 graph: w is rank 3, expected 2
    rank 1 weight:             malformed PT2 graph: w is rank 1, expected 2 |}]

(* WHERE it fails is the property. [Linear.output_shape] already compares the
   activation's C and the weight's C against [in_features] (linear.ml:69-84), so
   the importer does not restate the rule -- a second copy is one that can drift
   from the first. *)
let%expect_test "a feature-count mismatch fails in shape inference" =
  show "x trailing 4, w in 3:" (prog ~x_sizes:[ 2; 4 ] (linear ()));
  [%expect
    {| x trailing 4, w in 3:      input C extent must equal in_features: 4 vs 3 |}]

let%expect_test "missing weight metadata names the linear role" =
  show "no weight meta:"
    (program ~x_sizes:[ 2; 3 ]
       ~nodes:[ linear () ]
       ~graph_outputs:[ as_tensor "y" ]
       ());
  [%expect
    {| no weight meta:            malformed PT2 graph: no linear weight metadata for "w" |}]
