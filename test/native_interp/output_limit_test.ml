(* The two serialized-output bounds in [Native_interp], driven directly.

   They need their own home because the `visualize` path cannot show them:
   Model Explorer's per-node ceiling (max_outputs_metadata_per_node = 1024) is
   TIGHTER than this one and fires first there. Every other caller of
   [Native_interp.lower] — the .pt2 interpreter, and native_graph's
   print/eval/transform/to4d — has no Model Explorer limit at all, and this
   bound is what protects them.

   The rule is exclusive, matching [Kernel.Limits.create] and
   [Split.Unbind.output_shapes]: 4095 names accepted, 4096 refused. The
   traversal stops AT the limit and never learns the real length, which is why
   the payload says [At_least] rather than [Exact]. *)

open Programs

(* One node's [Argument.Tensors] list, at and around the ceiling. The accepted
   row is the control: a harness that refused everything would "pass" while
   proving nothing. *)
let%expect_test "node output list: the ceiling is exclusive" =
  let limit = Kernel.Limits.Hard.outputs in
  List.iter
    (fun n -> show (jstr "%d names:" n) (unbind_program n))
    [ limit - 1; limit; limit + 1 ];
  [%expect
    {|
    4095 names:                lowered, nodes=1
    4096 names:                PT2 graph over limit: at least 4096 outputs, above the maximum of 4095
    4097 names:                PT2 graph over limit: at least 4096 outputs, above the maximum of 4095 |}]

(* Several list-valued GRAPH outputs, each far under the ceiling, whose total
   exceeds it. A bound applied per list would accept this; the budget is
   threaded across the whole interface instead, so it does not.

   The same 1000 slices are returned repeatedly, which keeps every node at 1000
   outputs while the graph's own signature reaches 4000 then 5000. *)
let%expect_test "graph outputs: the ceiling is on the aggregate" =
  let slices = slice_names 1000 in
  let at copies =
    program ~x_sizes:[ 1000; 2 ]
      ~nodes:[ unbind_node ~dim:0 ~outs:slices ]
      ~graph_outputs:(List.init copies (fun _ -> as_tensors slices))
      ()
  in
  List.iter (fun c -> show (jstr "%d x 1000 names:" c) (at c)) [ 4; 5 ];
  [%expect
    {|
    4 x 1000 names:            lowered, nodes=1
    5 x 1000 names:            PT2 graph over limit: at least 4096 outputs, above the maximum of 4095 |}]

(* A THIRD bound, distinct from the two above: [split.Tensor]'s chunk count is
   DERIVED from [extent]/[split_size], not listed in the node's own [outputs]
   arity the way [unbind.int]'s or [split_with_sizes.default]'s already are.
   [Native_interp_decode.split_tensor_sizes] must reject an over-limit derived
   count BEFORE building the [sizes] list -- and, crucially, before ever
   looking at the node's declared [outputs], which [materialized_output_names]
   bounds separately (the first bound above). [outs] below deliberately names
   only ONE output regardless of [extent], so that ceiling never fires and
   cannot be mistaken for this one: whatever rejects an [extent] in the
   millions with [split_size=1] has to be [split_tensor_sizes]'s own
   preflight. *)
let%expect_test
    "split.Tensor: the derived chunk count is bounded before the list is built"
    =
  let limit = Kernel.Limits.Hard.outputs in
  let mismatched_program ~extent =
    program ~x_sizes:[ extent; 2 ]
      ~nodes:[ split_tensor_node ~split_size:1 ~outs:[ "u0" ] ]
      ~graph_outputs:[ as_tensor "u0" ]
      ()
  in
  List.iter
    (fun n -> show (jstr "extent %d:" n) (mismatched_program ~extent:n))
    [ limit; limit + 1000000 ];
  [%expect
    {|
    extent 4096:               PT2 graph over limit: 4096 outputs, above the maximum of 4095
    extent 1004096:            PT2 graph over limit: 1004096 outputs, above the maximum of 4095 |}]
