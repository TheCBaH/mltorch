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

(* [Unfold.Unfold.dest_of] raises [Invalid_argument] only when [dimension]
   resolves to the frame's own [N] -- [y] already occupies all six axes at
   its outermost position, so the axis this op appends has no room at all.
   native_interp_lower_shape.ml's own arm catches it and reports the same
   [`Rank_over_six] row a genuinely too-large declared rank gets above,
   rather than letting it escape as an exception. *)
let%expect_test "unfold on an already-six-axis tensor is refused, not raised" =
  let unfold_dim0 =
    jstr
      {|{"target":"torch.ops.aten.unfold.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"dimension","arg":{"as_int":0},"kind":1},{"name":"size","arg":{"as_int":1},"kind":1},{"name":"step","arg":{"as_int":1},"kind":1}],"outputs":[%s],"metadata":{}}|}
      (as_tensor "y") (as_tensors [ "u0" ])
  in
  show "y declared rank 6:"
    (program ~x_sizes:[ 2; 3 ]
       ~nodes:[ relu ~out:(as_tensor "y"); unfold_dim0 ]
       ~graph_outputs:[ as_tensor "u0" ]
       ~extra_tensor_values:[ ("y", tensor_meta [ 1; 1; 1; 1; 2; 3 ]) ]
       ());
  [%expect
    {| y declared rank 6:         malformed PT2 graph: y has rank greater than six |}]

(* The op CONFIGURATION -- stride, padding, dilation, groups, kernel -- is
   model data exactly as much as a shape is, and it reached
   [Op_config.Pos.of_int]/[Nonneg.of_int]/[Dim.extent] unguarded. Those raise
   [Invalid_argument], and [lower] catches it nowhere (the module's only handler
   is around [Tensor.materialize]), so a serialized zero stride left this
   function as an exception rather than as an [Err] row -- past the Model
   Explorer boundary, where every malformed row is meant to CLASSIFY.

   Both shipped arms had it, which is why the witnesses below are
   convolution.default and max_pool2d_with_indices.default and not the exact
   conv2d/max_pool2d targets: this is a pre-existing hole, not one the
   functional overloads introduce. *)
let conv ~groups ~padding =
  jstr
    {|{"target":"torch.ops.aten.convolution.default","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"weight","arg":%s,"kind":1},{"name":"bias","arg":{"as_none":true},"kind":1},{"name":"stride","arg":{"as_ints":[1,1]},"kind":1},{"name":"padding","arg":{"as_ints":[%d,%d]},"kind":1},{"name":"dilation","arg":{"as_ints":[1,1]},"kind":1},{"name":"transposed","arg":{"as_bool":false},"kind":1},{"name":"output_padding","arg":{"as_ints":[0,0]},"kind":1},{"name":"groups","arg":{"as_int":%d},"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") (as_tensor "w") padding padding groups (as_tensor "y")

let conv_program ~groups ~padding =
  program ~x_sizes:[ 1; 4; 8; 8 ]
    ~extra_tensor_values:[ ("w", tensor_meta [ 8; 4; 3; 3 ]) ]
    ~nodes:[ conv ~groups ~padding ]
    ~graph_outputs:[ as_tensor "y" ]
    ()

let%expect_test "a zero group count is refused, not raised" =
  show "groups = 0:" (conv_program ~groups:0 ~padding:0);
  [%expect
    {| groups = 0:                malformed PT2 graph: torch.ops.aten.convolution.default: groups must be positive, got 0 |}]

(* [Nonneg] rather than [Pos]: zero padding is ordinary, a negative one is not,
   and the two constructors have to be distinguishable in the diagnostic or the
   row cannot say which rule the model broke. *)
let%expect_test "a negative padding is refused, not raised" =
  show "padding = -1:" (conv_program ~groups:1 ~padding:(-1));
  [%expect
    {| padding = -1:              malformed PT2 graph: torch.ops.aten.convolution.default: padding must not be negative, got -1 |}]

(* The pool arm reaches the same constructors by a different route, so it needs
   its own witness: sharing [conv_params] would not have covered it. *)
let%expect_test "a zero pool stride is refused, not raised" =
  let pool =
    jstr
      {|{"target":"torch.ops.aten.max_pool2d_with_indices.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"kernel_size","arg":{"as_ints":[2,2]},"kind":1},{"name":"stride","arg":{"as_ints":[0,0]},"kind":1}],"outputs":[%s,%s],"metadata":{}}|}
      (as_tensor "x") (as_tensor "y") (as_tensor "i")
  in
  show "pool stride = 0:"
    (program ~x_sizes:[ 1; 4; 8; 8 ] ~nodes:[ pool ]
       ~graph_outputs:[ as_tensor "y" ]
       ());
  [%expect
    {| pool stride = 0:           malformed PT2 graph: torch.ops.aten.max_pool2d_with_indices.default: stride must be positive, got 0 |}]

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
    {| view [0; -1]:              malformed PT2 graph: view size [0, -1]: extent must be >= 1, got 0 |}]

(* The divisibility check [resolve_view_size] adds divides by [known] even
   with NO [-1] present, so a declared [0] alongside an ordinary extent is a
   distinct case from the [-1]-alongside-[0] one above: nothing about it
   involves inference. It is caught earlier still, by [Dim.extent_checked] on
   the entry itself, before any division runs. *)
let view_sized size =
  let view =
    jstr
      {|{"target":"torch.ops.aten.view.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"size","arg":{"as_ints":[%s]},"kind":1}],"outputs":[%s],"metadata":{}}|}
      (as_tensor "x")
      (String.concat "," (List.map string_of_int size))
      (as_tensor "y")
  in
  fun ~x_sizes ->
    program ~x_sizes ~nodes:[ view ] ~graph_outputs:[ as_tensor "y" ] ()

let%expect_test "a view sizing a zero with no -1 present is refused" =
  show "view [0; 6]:" (view_sized [ 0; 6 ] ~x_sizes:[ 2; 3 ]);
  [%expect
    {| view [0; 6]:               malformed PT2 graph: view size [0, 6]: extent must be >= 1, got 0 |}]

(* op3-impl.md F1's source-numel half: [Vec6.numel] wraps a source of 2^32
   elements to 0 as a js_of_ocaml [int], and an unchecked [int] equality
   between source and target numel would then pass. [resolve_view] bounds the
   SOURCE via [Vec6.numel_bounded] before any comparison, so this is refused
   even though the target (itself) is numel-EQUAL and would have been valid at
   any representable size. *)
let%expect_test
    "a source past the numel ceiling is refused, not silently wrapped" =
  show "source 65536x65536 -> itself:"
    (view_sized [ 65536; 65536 ] ~x_sizes:[ 65536; 65536 ]);
  [%expect
    {|
    source 65536x65536 -> itself: malformed PT2 graph: view size [65536, 65536]: axis C: 65536 elements so far times extent 65536 reaches the maximum of 2147483648 |}]

(* The fencepost: 65536*32768 = 2^31 EXACTLY, which the hard contract excludes
   ("stays BELOW 2^31") -- pinning that [numel_bounded]'s ceiling is exclusive
   all the way through this importer, not just at the primitive
   ([test/native/vec6_test.ml]). *)
let%expect_test "a source of exactly 2^31 elements is refused" =
  show "source 65536x32768 -> itself:"
    (view_sized [ 65536; 32768 ] ~x_sizes:[ 65536; 32768 ]);
  [%expect
    {|
    source 65536x32768 -> itself: malformed PT2 graph: view size [65536, 32768]: axis C: 65536 elements so far times extent 32768 reaches the maximum of 2147483648 |}]

(* [_native_batch_norm_legit_no_training]'s schema has a REQUIRED [float eps].
   When the optional-float arm was added to the shared [float_arg] rather than to
   a separate decoder, this arm silently read an explicit null epsilon as 0. --
   a different op computed under the right name. The optionality belongs to the
   ARGUMENT, so it belongs at the call site. *)
let%expect_test "batch_norm's required eps rejects an explicit none" =
  let bn eps =
    let eps_arg =
      match eps with
      | None -> ""
      | Some v -> Programs.jstr {|,{"name":"eps","arg":%s,"kind":1}|} v
    in
    Programs.program ~x_sizes:[ 1; 2; 2; 2 ]
      ~extra_tensor_values:
        [ ("m", Programs.tensor_meta [ 2 ]); ("v", Programs.tensor_meta [ 2 ]) ]
      ~params:[ "m"; "v" ]
      ~nodes:
        [
          Programs.jstr
            {|{"target":"torch.ops.aten._native_batch_norm_legit_no_training.default","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"weight","arg":{"as_none":true},"kind":1},{"name":"bias","arg":{"as_none":true},"kind":1},{"name":"running_mean","arg":%s,"kind":1},{"name":"running_var","arg":%s,"kind":1},{"name":"momentum","arg":{"as_float":0.1},"kind":1}%s],"outputs":[%s],"metadata":{}}|}
            (Programs.as_tensor "x") (Programs.as_tensor "m")
            (Programs.as_tensor "v") eps_arg (Programs.as_tensor "y");
        ]
      ~graph_outputs:[ Programs.as_tensor "y" ]
      ()
  in
  Programs.show "eps 1e-5:" (bn (Some {|{"as_float":1e-05}|}));
  Programs.show "eps none:" (bn (Some {|{"as_none":true}|}));
  (* And OMISSION, which the previous shape of this helper answered with a
     silent zero -- "required" meant "defaults to 0." while [Op_bridge] reported
     the argument missing. *)
  Programs.show "eps omitted:" (bn None);
  [%expect
    {|
    eps 1e-5:                  lowered, nodes=3
    eps none:                  malformed PT2 graph: torch.ops.aten._native_batch_norm_legit_no_training.default.eps is not a float
    eps omitted:               malformed PT2 graph: torch.ops.aten._native_batch_norm_legit_no_training.default: missing argument "eps" |}]
