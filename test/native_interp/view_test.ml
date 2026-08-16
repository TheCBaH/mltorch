(* torch.ops.aten.view.default / _unsafe_view.default through the serialized
   path. Both overloads share one dispatch arm (op3-impl.md commit 4), so every
   case here runs under BOTH targets -- _unsafe_view cannot hide behind
   view.default's coverage. Numeric agreement is checked in
   test/native_bridge_test.ml; this file asserts what [Native_interp.lower]
   builds, mirroring conv_test.ml's [dump] pattern. *)

open Programs

let view_targets =
  [ "torch.ops.aten.view.default"; "torch.ops.aten._unsafe_view.default" ]

let view_node target size =
  jstr
    {|{"target":"%s","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"size","arg":{"as_ints":[%s]},"kind":1}],"outputs":[%s],"metadata":{}}|}
    target (as_tensor "x")
    (String.concat "," (List.map string_of_int size))
    (as_tensor "y")

let prog ?(x_sizes = [ 2; 3 ]) target size =
  program ~x_sizes
    ~nodes:[ view_node target size ]
    ~graph_outputs:[ as_tensor "y" ]
    ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

let dump_both ~x_sizes size =
  List.iter (fun t -> dump t (prog ~x_sizes t size)) view_targets

let%expect_test "view/_unsafe_view lower to a Reshape node, both targets" =
  dump_both ~x_sizes:[ 2; 3 ] [ 3; 2 ];
  [%expect
    {|
    torch.ops.aten.view.default
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=3 C=2]] = reshape x=t0 params={shape=[W=3 C=2]}
    outputs: [t1 f32 [W=3 C=2] <-n0]
    torch.ops.aten._unsafe_view.default
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=3 C=2]] = reshape x=t0 params={shape=[W=3 C=2]}
    outputs: [t1 f32 [W=3 C=2] <-n0] |}]

let%expect_test "view/_unsafe_view: rank-increasing and rank-decreasing targets"
    =
  dump_both ~x_sizes:[ 6 ] [ 2; 3 ];
  dump_both ~x_sizes:[ 2; 3; 4 ] [ 24 ];
  [%expect
    {|
    torch.ops.aten.view.default
    graph
    inputs: [t0 f32 [C=6] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = reshape x=t0 params={shape=[W=2 C=3]}
    outputs: [t1 f32 [W=2 C=3] <-n0]
    torch.ops.aten._unsafe_view.default
    graph
    inputs: [t0 f32 [C=6] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = reshape x=t0 params={shape=[W=2 C=3]}
    outputs: [t1 f32 [W=2 C=3] <-n0]
    torch.ops.aten.view.default
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [C=24]] = reshape x=t0 params={shape=[C=24]}
    outputs: [t1 f32 [C=24] <-n0]
    torch.ops.aten._unsafe_view.default
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [C=24]] = reshape x=t0 params={shape=[C=24]}
    outputs: [t1 f32 [C=24] <-n0] |}]

let%expect_test "view/_unsafe_view: leading, middle and trailing -1" =
  dump_both ~x_sizes:[ 2; 3; 4 ] [ -1; 2; 3 ];
  dump_both ~x_sizes:[ 2; 3; 4 ] [ 3; -1; 2 ];
  dump_both ~x_sizes:[ 2; 3; 4 ] [ 3; 2; -1 ];
  [%expect
    {|
    torch.ops.aten.view.default
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=4 W=2 C=3]] = reshape x=t0 params={shape=[H=4 W=2 C=3]}
    outputs: [t1 f32 [H=4 W=2 C=3] <-n0]
    torch.ops.aten._unsafe_view.default
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=4 W=2 C=3]] = reshape x=t0 params={shape=[H=4 W=2 C=3]}
    outputs: [t1 f32 [H=4 W=2 C=3] <-n0]
    torch.ops.aten.view.default
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=3 W=4 C=2]] = reshape x=t0 params={shape=[H=3 W=4 C=2]}
    outputs: [t1 f32 [H=3 W=4 C=2] <-n0]
    torch.ops.aten._unsafe_view.default
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=3 W=4 C=2]] = reshape x=t0 params={shape=[H=3 W=4 C=2]}
    outputs: [t1 f32 [H=3 W=4 C=2] <-n0]
    torch.ops.aten.view.default
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=3 W=2 C=4]] = reshape x=t0 params={shape=[H=3 W=2 C=4]}
    outputs: [t1 f32 [H=3 W=2 C=4] <-n0]
    torch.ops.aten._unsafe_view.default
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=3 W=2 C=4]] = reshape x=t0 params={shape=[H=3 W=2 C=4]}
    outputs: [t1 f32 [H=3 W=2 C=4] <-n0] |}]

(* Unlike [Op_bridge] (which routes the resolved list through
   [Aten_shape.of_aten] and so reports [`Rank_out_of_range]), [Native_interp]'s
   [resolve_view] hands the resolved list to [shape_of_sizes] -- the same
   right-alignment every other importer arm uses -- whose OWN, pre-existing
   rank check reports [`Bad_dimension { fault = `Rank_over_six }] naming the
   SOURCE tensor. Different row, same target rank, both typed. *)
let%expect_test "view/_unsafe_view: rank 6 accepted, rank 7 rejected" =
  dump_both ~x_sizes:[ 6 ] [ 1; 1; 1; 1; 1; 6 ];
  dump_both ~x_sizes:[ 6 ] [ 1; 1; 1; 1; 1; 1; 6 ];
  [%expect
    {|
    torch.ops.aten.view.default
    graph
    inputs: [t0 f32 [C=6] ->[n0]]
    nodes:
      n0: [t1 f32 [C=6]] = reshape x=t0 params={shape=[C=6]}
    outputs: [t1 f32 [C=6] <-n0]
    torch.ops.aten._unsafe_view.default
    graph
    inputs: [t0 f32 [C=6] ->[n0]]
    nodes:
      n0: [t1 f32 [C=6]] = reshape x=t0 params={shape=[C=6]}
    outputs: [t1 f32 [C=6] <-n0]
    torch.ops.aten.view.default
      malformed PT2 graph: x has rank greater than six
    torch.ops.aten._unsafe_view.default
      malformed PT2 graph: x has rank greater than six |}]

(* op3-impl.md F1's three invalid-target holes, now typed rejections. *)
let%expect_test "view/_unsafe_view: reject F1's three invalid targets" =
  dump_both ~x_sizes:[ 2; 3 ] [ -1; -1 ];
  dump_both ~x_sizes:[ 2; 3 ] [ 4; 2 ];
  dump_both ~x_sizes:[ 2; 3 ] [ 4; -1 ];
  [%expect
    {|
    torch.ops.aten.view.default
      malformed PT2 graph: view size [-1, -1]: view size [-1, -1] has more than one inferred (-1) dimension
    torch.ops.aten._unsafe_view.default
      malformed PT2 graph: view size [-1, -1]: view size [-1, -1] has more than one inferred (-1) dimension
    torch.ops.aten.view.default
      malformed PT2 graph: view size [4, 2]: view size [4, 2] does not match 6 elements
    torch.ops.aten._unsafe_view.default
      malformed PT2 graph: view size [4, 2]: view size [4, 2] does not match 6 elements
    torch.ops.aten.view.default
      malformed PT2 graph: view size [4, -1]: view size [4, -1] does not divide 6 elements
    torch.ops.aten._unsafe_view.default
      malformed PT2 graph: view size [4, -1]: view size [4, -1] does not divide 6 elements |}]

(* A zero extent (no -1), and -1 alongside 0 (commit 9390ae6 -- pinned here to
   confirm it still holds under the shared resolver, and now under both
   targets). *)
let%expect_test "view/_unsafe_view: reject a zero extent, with and without -1" =
  dump_both ~x_sizes:[ 2; 3 ] [ 0; 6 ];
  dump_both ~x_sizes:[ 2; 3 ] [ 0; -1 ];
  [%expect
    {|
    torch.ops.aten.view.default
      malformed PT2 graph: view size [0, 6]: extent must be >= 1, got 0
    torch.ops.aten._unsafe_view.default
      malformed PT2 graph: view size [0, 6]: extent must be >= 1, got 0
    torch.ops.aten.view.default
      malformed PT2 graph: view size [0, -1]: extent must be >= 1, got 0
    torch.ops.aten._unsafe_view.default
      malformed PT2 graph: view size [0, -1]: extent must be >= 1, got 0 |}]

(* An aggregate past the extent ceiling, exercising commit 1's int64 bound,
   under both targets -- signatures only, so the oversized SOURCE is free to
   construct here (unlike the ATen bridge). *)
let%expect_test "view/_unsafe_view: reject a source past the numel ceiling" =
  dump_both ~x_sizes:[ 65536; 65536 ] [ 65536; 65536 ];
  [%expect
    {|
    torch.ops.aten.view.default
      malformed PT2 graph: view size [65536, 65536]: axis C: 65536 elements so far times extent 65536 reaches the maximum of 2147483648
    torch.ops.aten._unsafe_view.default
      malformed PT2 graph: view size [65536, 65536]: axis C: 65536 elements so far times extent 65536 reaches the maximum of 2147483648 |}]
