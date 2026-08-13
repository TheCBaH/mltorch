(* Untrusted model data must never leave [Native_interp.lower] as an uncaught
   exception. Every case here DID, and was found by review rather than by the
   suite — which is the argument for driving them from a fixture instead of
   reasoning about reachability.

   [me_visualize_unsupported_cram.t] drives one witness per malformed row
   through the CLI; these are the narrower question of what happens when a
   node's declared shape and the shape Native infers disagree, which needs two
   nodes and is clearer in OCaml than in a shell heredoc. *)

open Programs

let relu ~out =
  jstr
    {|{"target":"torch.ops.aten.relu.default","inputs":[{"name":"self","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") out

(* `output_names` flattens `as_tensors` for EVERY node now, not only the ops
   that return a list — so a node shape that used to be refused outright as
   `Non_tensor_node_output` reaches the binder. An arity check living in the
   unbind arm would leave this hole open for every other op; the check belongs
   where the produced ids are, which is `bind`. *)
let%expect_test
    "a non-list op carrying a Tensor[] output is refused, not raised" =
  show "relu with 2 names:"
    (program ~x_sizes:[ 2; 3 ]
       ~nodes:[ relu ~out:(as_tensors [ "y"; "z" ]) ]
       ~graph_outputs:[ as_tensor "y" ]
       ());
  [%expect
    {| relu with 2 names:         malformed PT2 graph: torch.ops.aten.relu.default declares 2 outputs but produces 1 |}]

(* The count the check compares against has to be the count the BUILDER
   produced, not one derived from metadata. Here `y`'s declared shape says the
   unbind yields five slices while the relu that produces it really gives two,
   so a metadata-derived check passes and the binder blows up. *)
let%expect_test "declared metadata disagreeing with the inferred shape" =
  let unbind_y =
    jstr
      {|{"target":"torch.ops.aten.unbind.int","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"dim","arg":{"as_int":0},"kind":1}],"outputs":[%s],"metadata":{}}|}
      (as_tensor "y")
      (as_tensors (slice_names 5))
  in
  show "y declared [5;3]:"
    (program ~x_sizes:[ 2; 3 ]
       ~nodes:[ relu ~out:(as_tensor "y"); unbind_y ]
       ~graph_outputs:[ as_tensor "u0" ]
       ~extra_tensor_values:[ ("y", tensor_meta [ 5; 3 ]) ]
       ());
  [%expect
    {| y declared [5;3]:          malformed PT2 graph: torch.ops.aten.unbind.int declares 5 outputs but produces 2 |}]

(* `axes_for_rank` builds the used-axis list as the innermost `rank` frame axes,
   which SATURATES at six — so for rank seven the `d >= rank` guard admits
   d = 6 and `List.nth` raises. `shape_of_sizes`' own rank check does not cover
   this: it runs over graph inputs and captured tensors, not over an edge a node
   produced. The guard is shared by mean.dim and permute.default, which had the
   same latent path. *)
let%expect_test "a rank-seven node-produced edge is refused, not raised" =
  let unbind_dim6 =
    jstr
      {|{"target":"torch.ops.aten.unbind.int","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"dim","arg":{"as_int":6},"kind":1}],"outputs":[%s],"metadata":{}}|}
      (as_tensor "y") (as_tensors [ "u0" ])
  in
  show "y declared rank 7:"
    (program ~x_sizes:[ 2; 3 ]
       ~nodes:[ relu ~out:(as_tensor "y"); unbind_dim6 ]
       ~graph_outputs:[ as_tensor "u0" ]
       ~extra_tensor_values:[ ("y", tensor_meta [ 1; 1; 1; 1; 1; 1; 2 ]) ]
       ());
  [%expect
    {| y declared rank 7:         malformed PT2 graph: y has rank greater than six |}]

(* A `view` whose target holds both 0 and -1 makes the product of the known
   sizes vanish, and the inference divides by it. Pre-existing and unrelated to
   `Tensor[]`, but it is the same "a declared zero escapes as an uncaught
   exception" hole the `Zero` dim_fault closes one function above. *)
let%expect_test "a view sizing both 0 and -1 is refused, not raised" =
  let view =
    jstr
      {|{"target":"torch.ops.aten.view.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"size","arg":{"as_ints":[0,-1]},"kind":1}],"outputs":[%s],"metadata":{}}|}
      (as_tensor "x") (as_tensor "y")
  in
  show "view [0; -1]:"
    (program ~x_sizes:[ 2; 3 ] ~nodes:[ view ]
       ~graph_outputs:[ as_tensor "y" ]
       ());
  [%expect
    {| view [0; -1]:              malformed PT2 graph: view has a zero-length dimension |}]
