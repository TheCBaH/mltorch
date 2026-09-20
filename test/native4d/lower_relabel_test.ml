(* [Lower_relabel] beyond split attention: permute, expand, clone, the scalar
   and activation ops, batched_matmul and the reductions, all with D read as N.
   Every case lowers a Native graph and compares each output element-for-element
   with Native's own evaluation, on values that are not exact in f32, so a
   mis-conjugated permutation or a wrong reduction axis is a mismatch. *)

open Axis

let s = Fixtures.s
let perm_of = Fixtures.perm_of
let inexact = Lower_region_test.inexact
let compare_graphs = Lower_region_test.compare_graphs ~fill:inexact
let build name b = Graph_builder.build ~name ~outputs:Fun.id b
let ok g = Err.or_raise ~pp_error:Graph_builder.pp_error g
let sum_d = { Reduce.Sum.dims = [ D ]; keepdim = false }

(* [x] is [H=4 W=3 C=6]; the reshape gives it a real D. *)
let onto_d = s 1 1 4 3 3 2

let permuted_group perm =
  build "permute"
    (let open Graph_builder in
     let* x = input ~shape:(s 1 1 1 4 3 6) () in
     let* r = reshape { Reshape.Reshape.shape = onto_d } x in
     let* p = permute (perm_of perm) r in
     let* q = silu p in
     let* out = sum sum_d q in
     Graph_builder.return [ out ])
  |> ok

let%expect_test "permute of a D-carrying value, D moved and moved back" =
  List.iter
    (fun (name, perm) ->
      compare_graphs name ~x_shapes:[ s 1 1 1 4 3 6 ] (permuted_group perm))
    [
      ("D<->H", [ (D, H); (H, D) ]);
      ("D<->W", [ (D, W); (W, D) ]);
      ("D->H->W->D", [ (D, H); (H, W); (W, D) ]);
    ];
  [%expect
    {|
    D<->H: 1 outputs, 4 nodes, identical=true
    D<->W: 1 outputs, 4 nodes, identical=true
    D->H->W->D: 1 outputs, 4 nodes, identical=true |}]

(* The permute is where the extent enters or leaves D: only one side is
   relabelled. *)
let%expect_test "permute that introduces or removes the D extent" =
  let enter =
    build "enter"
      (let open Graph_builder in
       let* x = input ~shape:(s 1 1 1 4 3 6) () in
       let* p = permute (perm_of [ (D, H); (H, D) ]) x in
       let* q = relu p in
       let* out = sum sum_d q in
       Graph_builder.return [ out ])
    |> ok
  in
  compare_graphs "enter" ~x_shapes:[ s 1 1 1 4 3 6 ] enter;
  let leave =
    build "leave"
      (let open Graph_builder in
       let* x = input ~shape:(s 1 1 1 4 3 6) () in
       let* r = reshape { Reshape.Reshape.shape = s 1 1 4 1 3 6 } x in
       let* q = mul_scalar 1.5 r in
       let* p = permute (perm_of [ (D, H); (H, D) ]) q in
       Graph_builder.return [ p ])
    |> ok
  in
  compare_graphs "leave" ~x_shapes:[ s 1 1 1 4 3 6 ] leave;
  [%expect
    {|
    enter: 1 outputs, 3 nodes, identical=true
    leave: 1 outputs, 3 nodes, identical=true |}]

(* Outlook attention: a batch of D windows, each [H=5 W=9 C=9] x [.. C=7]. *)
let matmul_graph =
  build "batched_matmul"
    (let open Graph_builder in
     let* a = input ~shape:(s 1 1 1 5 9 54) () in
     let* b = input ~shape:(s 1 1 1 5 9 42) () in
     let* a = reshape { Reshape.Reshape.shape = s 1 1 6 5 9 9 } a in
     let* b = reshape { Reshape.Reshape.shape = s 1 1 6 5 9 7 } b in
     let* m = batched_matmul a b in
     let* m = add_scalar 0.25 m in
     let* out = sum sum_d m in
     Graph_builder.return [ out ])
  |> ok

let%expect_test "batched_matmul over a D batch" =
  compare_graphs "outlook"
    ~x_shapes:[ s 1 1 1 5 9 54; s 1 1 1 5 9 42 ]
    matmul_graph;
  [%expect {| outlook: 1 outputs, 5 nodes, identical=true |}]

let%expect_test "expand and clone of a D-carrying value" =
  let g =
    build "expand_clone"
      (let open Graph_builder in
       let* x = input ~shape:(s 1 1 1 3 2 2) () in
       let* w = input ~shape:(s 1 1 1 12 2 2) () in
       let* w = reshape { Reshape.Reshape.shape = s 1 1 4 3 2 2 } w in
       let* e = expand { Pointwise.Expand.size = s 1 1 4 3 2 2 } x in
       let* c = clone e in
       let* m = mul c w in
       let* out = sum sum_d m in
       Graph_builder.return [ out ])
    |> ok
  in
  compare_graphs "expand" ~x_shapes:[ s 1 1 1 3 2 2; s 1 1 1 12 2 2 ] g;
  [%expect {| expand: 1 outputs, 4 nodes, identical=true |}]

(* Reductions: a reduced D, or a kept one on the way to a later reduction. *)
let%expect_test "amax, mean and sum over D-carrying values" =
  let g =
    build "reductions"
      (let open Graph_builder in
       let* x = input ~shape:(s 1 1 1 4 3 6) () in
       let* r = reshape { Reshape.Reshape.shape = onto_d } x in
       let* hi = amax { Reduce.Amax.dims = [ H ]; keepdim = true } r in
       let* mean_h = mean { Reduce.Mean.dims = [ H; C ]; keepdim = true } r in
       let* both = sub hi mean_h in
       let* pooled = mean { Reduce.Mean.dims = [ D ]; keepdim = false } both in
       let* top = amax { Reduce.Amax.dims = [ D; H ]; keepdim = false } r in
       let* total = sum { Reduce.Sum.dims = [ D ]; keepdim = true } r in
       Graph_builder.return [ pooled; top; total ])
    |> ok
  in
  compare_graphs "reductions" ~x_shapes:[ s 1 1 1 4 3 6 ] g;
  [%expect {| reductions: 3 outputs, 7 nodes, identical=true |}]

(* Refusals: a group that meets anything whose meaning does not survive the
   move is left alone, and the ordinary path names the blocker. *)
let%expect_test "refused groups" =
  let refuse name build_group =
    compare_graphs name
      ~x_shapes:[ s 1 1 1 4 3 6 ]
      (build name build_group |> ok)
  in
  (* The extent leaves the group on a graph output. *)
  refuse "output carries D"
    (let open Graph_builder in
     let* x = input ~shape:(s 1 1 1 4 3 6) () in
     let* r = reshape { Reshape.Reshape.shape = onto_d } x in
     let* p = permute (perm_of [ (D, W); (W, D) ]) r in
     Graph_builder.return [ p ]);
  (* An unbind of the innermost axes drops it under a larger extent on T. *)
  refuse "unbind of D under T"
    (let open Graph_builder in
     let* x = input ~shape:(s 1 1 1 4 3 6) () in
     let* r = reshape { Reshape.Reshape.shape = s 1 2 3 2 3 2 } x in
     let* parts = unbind { Split.Unbind.axis = D } r in
     let* outs =
       List.fold_right
         (fun part acc ->
           let* acc = acc in
           let* flat = reshape { Reshape.Reshape.shape = s 1 1 1 1 8 3 } part in
           Graph_builder.return (flat :: acc))
         parts (Graph_builder.return [])
     in
     Graph_builder.return outs);
  [%expect
    {|
    output carries D: node n1: axis D is outside the N/H/W/C dialect
    unbind of D under T: node n1: axis D is outside the N/H/W/C dialect |}]

let%expect_test "the widened groups' maps verify" =
  Lower_region_test.verified_graph "permute" (permuted_group [ (D, H); (H, D) ]);
  Lower_region_test.verified_graph "outlook" matmul_graph;
  [%expect
    {|
    permute: 8 clusters: 1 proved (structural) [sampled 8], 1 unproved (over max_rounds) [sampled 8], 6 vacuous
    outlook: 11 clusters: 2 proved (structural) [sampled 8], 1 unproved (over max_rounds) [sampled 8], 8 vacuous |}]

(* A depthwise convolution over a D batch (mvitv2's pooled heads): Native treats
   N, T and D alike as batch, so reading D as N changes nothing for it. *)
let%expect_test "conv2d over a D batch" =
  let window ~kernel : Conv.Conv2d.axis_window =
    {
      kernel = Dim.extent kernel;
      stride = Op_config.Pos.of_int 2;
      pad_before = Op_config.Nonneg.of_int 1;
      pad_after = Op_config.Nonneg.of_int 1;
      dilation = Op_config.Pos.of_int 1;
    }
  in
  let params : Conv.Conv2d.params =
    {
      h = window ~kernel:3;
      w = window ~kernel:3;
      in_channels = Dim.extent 6;
      groups = Op_config.Pos.of_int 6;
    }
  in
  let g =
    build "conv_d"
      (let open Graph_builder in
       let* x = input ~shape:(s 1 1 1 10 5 12) () in
       let* w = input ~shape:(s 6 1 1 3 3 1) () in
       let* r = reshape { Reshape.Reshape.shape = s 1 1 4 5 5 6 } x in
       let* c = conv2d params ~x:r ~weight:w () in
       let* out = sum sum_d c in
       Graph_builder.return [ out ])
    |> ok
  in
  compare_graphs "depthwise" ~x_shapes:[ s 1 1 1 10 5 12; s 6 1 1 3 3 1 ] g;
  [%expect {| depthwise: 1 outputs, 3 nodes, identical=true |}]

(* ---- T and D fused into N -------------------------------------------------- *)

(* mvitv2's relative-position bias: heads on D, then a five-axis sum of the
   attention, a bias along H and a bias along W, each reaching it through a
   permute that moves the heads onto T. The axes N, T and D are adjacent in the
   frame, so a tensor carrying T and D reads as one with N = T * D. *)
let bias_graph =
  build "rel_pos"
    (let open Graph_builder in
     let* attn = input ~shape:(s 1 1 1 2 9 9) () in
     let* q = input ~shape:(s 1 1 1 6 3 4) () in
     let* rel_h = input ~shape:(s 1 1 1 3 4 3) () in
     let* rel_w = input ~shape:(s 1 1 1 3 4 3) () in
     let* attn = reshape { Reshape.Reshape.shape = s 1 2 3 3 3 3 } attn in
     let* q = reshape { Reshape.Reshape.shape = s 1 1 2 3 3 4 } q in
     let* by_h = batched_matmul q rel_h in
     let* by_h =
       permute (perm_of [ (T, D); (D, H); (H, W); (W, C); (C, T) ]) by_h
     in
     let* q_t = permute (perm_of [ (H, W); (W, H) ]) q in
     let* by_w = batched_matmul q_t rel_w in
     let* by_w = permute (perm_of [ (T, D); (D, W); (W, T) ]) by_w in
     let* biased = add attn by_h in
     let* biased = add biased by_w in
     let* out = reshape { Reshape.Reshape.shape = s 1 1 1 2 9 9 } biased in
     Graph_builder.return [ out ])
  |> ok

let bias_shapes = [ s 1 1 1 2 9 9; s 1 1 1 6 3 4; s 1 1 1 3 4 3; s 1 1 1 3 4 3 ]

let%expect_test "five-axis bias sum, heads through T" =
  Lower_region_test.compare_graphs
    ~fill:(Interleave_test.distinct ())
    "rel_pos" ~x_shapes:bias_shapes bias_graph;
  [%expect {| rel_pos: 1 outputs, 11 nodes, identical=true |}]

let%expect_test "the fused group's map verifies" =
  Lower_region_test.verified_graph "rel_pos" bias_graph;
  [%expect
    {| rel_pos: 24 clusters: 4 proved (structural) [sampled 8], 1 unproved (over max_rounds) [sampled 8], 19 vacuous |}]

(* ---- windowed attention ---------------------------------------------------- *)

let distinct_compare name ~x_shapes g =
  Lower_region_test.compare_graphs
    ~fill:(Interleave_test.distinct ())
    name ~x_shapes g

(* [numel] elements laid out on W, which no group reads as T or D. *)
let flat numel = s 1 1 1 1 numel 1

let flatten outs ~numel =
  List.fold_right
    (fun out acc ->
      let open Graph_builder in
      let* acc = acc in
      let* out = reshape { Reshape.Reshape.shape = flat numel } out in
      Graph_builder.return (out :: acc))
    outs (Graph_builder.return [])

(* Hiera and sam2: the window batch sits on T and D, the heads on H, and the
   attention reads them as batch axes. A rank-3 mask is one per head. *)
let windowed ~batch ~mask =
  build "windowed"
    (let open Graph_builder in
     let* q = input ~shape:(s 1 1 1 1 24 4) () in
     let* k = input ~shape:(s 1 1 1 1 24 4) () in
     let* v = input ~shape:(s 1 1 1 1 24 4) () in
     let* m =
       if mask then
         let* m = input ~shape:(s 1 1 1 2 2 2) () in
         Graph_builder.return (Some m)
       else Graph_builder.return None
     in
     let heads x = reshape { Reshape.Reshape.shape = batch } x in
     let* q = heads q in
     let* k = heads k in
     let* v = heads v in
     let* a =
       sdpa
         { Attention.Sdpa.scale = Default }
         ~query:q ~key:k ~value:v ?mask:m ()
     in
     let* out = reshape { Reshape.Reshape.shape = s 1 1 1 1 24 4 } a in
     Graph_builder.return [ out ])
  |> ok

let%expect_test "windowed attention over T and D, and over D alone" =
  let shapes = [ s 1 1 1 1 24 4; s 1 1 1 1 24 4; s 1 1 1 1 24 4 ] in
  let with_mask = shapes @ [ s 1 1 1 2 2 2 ] in
  distinct_compare "T and D" ~x_shapes:with_mask
    (windowed ~batch:(s 1 2 3 2 2 4) ~mask:true);
  distinct_compare "D" ~x_shapes:with_mask
    (windowed ~batch:(s 1 1 6 2 2 4) ~mask:true);
  distinct_compare "N, T and D" ~x_shapes:shapes
    (windowed ~batch:(s 2 3 2 2 2 2) ~mask:false);
  [%expect
    {|
    T and D: 1 outputs, 5 nodes, identical=true
    D: 1 outputs, 5 nodes, identical=true
    N, T and D: 1 outputs, 5 nodes, identical=true |}]

let%expect_test "the windowed attention's map verifies" =
  Lower_region_test.verified_graph "windowed"
    (windowed ~batch:(s 1 2 3 2 2 4) ~mask:true);
  [%expect
    {| windowed: 13 clusters: 1 proved (structural), 4 proved (structural) [sampled 8], 8 vacuous |}]

(* ---- splitting a fused batch ----------------------------------------------- *)

let numel shape =
  List.fold_left
    (fun acc a -> acc * Dim.to_int (Vec6.get shape a))
    1 [ N; T; D; H; W; C ]

(* [x] unbound along [axis] after a reshape onto [shape], each part scaled so it
   stays in the group, then laid flat so no output carries T or D. *)
let unbound ~shape ~axis =
  let count = Dim.to_int (Vec6.get shape axis) in
  build "unbind"
    (let open Graph_builder in
     let* x = input ~shape:(s 1 1 1 1 (numel shape) 1) () in
     let* r = reshape { Reshape.Reshape.shape } x in
     let* parts = unbind { Split.Unbind.axis } r in
     let* parts =
       List.fold_right
         (fun part acc ->
           let* acc = acc in
           let* part = mul_scalar 1.5 part in
           Graph_builder.return (part :: acc))
         parts (Graph_builder.return [])
     in
     let* outs = flatten parts ~numel:(numel shape / count) in
     Graph_builder.return outs)
  |> ok

let%expect_test "unbind of a fused batch, along each axis" =
  List.iter
    (fun (name, shape, axis) ->
      distinct_compare name
        ~x_shapes:[ s 1 1 1 1 (numel shape) 1 ]
        (unbound ~shape ~axis))
    [
      ("N", s 3 1 2 4 3 2, N);
      ("N under T and D", s 3 2 2 4 3 2, N);
      ("T", s 1 3 2 4 3 2, T);
      ("D", s 1 1 3 4 3 2, D);
      ("H", s 1 2 3 4 3 2, H);
      ("H under N", s 2 3 2 4 3 2, H);
      ("W", s 1 2 3 2 4 2, W);
      ("C", s 1 2 3 2 2 4, C);
    ];
  [%expect
    {|
    N: 3 outputs, 8 nodes, identical=true
    N under T and D: 3 outputs, 8 nodes, identical=true
    T: 3 outputs, 8 nodes, identical=true
    D: 3 outputs, 8 nodes, identical=true
    H: 4 outputs, 14 nodes, identical=true
    H under N: 4 outputs, 14 nodes, identical=true
    W: 4 outputs, 14 nodes, identical=true
    C: 4 outputs, 14 nodes, identical=true |}]

(* The qkv split of sam2, after the heads move onto D. *)
let%expect_test "split_with_sizes of a D batch" =
  List.iter
    (fun (name, axis, sizes) ->
      let shape = s 1 1 6 3 4 6 in
      let g =
        build "split"
          (let open Graph_builder in
           let* x = input ~shape:(s 1 1 1 1 (numel shape) 1) () in
           let* r = reshape { Reshape.Reshape.shape } x in
           let* parts =
             split_with_sizes { Split.Split_with_sizes.axis; sizes } r
           in
           let* parts =
             List.fold_right
               (fun part acc ->
                 let* acc = acc in
                 let* part = silu part in
                 let* flat =
                   reshape
                     {
                       Reshape.Reshape.shape =
                         flat (numel shape / List.length sizes);
                     }
                     part
                 in
                 Graph_builder.return (flat :: acc))
               parts (Graph_builder.return [])
           in
           Graph_builder.return parts)
        |> ok
      in
      distinct_compare name ~x_shapes:[ s 1 1 1 1 (numel shape) 1 ] g)
    [
      ("D", D, [ 3; 3 ]);
      ("C", C, [ 2; 2; 2 ]);
      ("H", H, [ 1; 1; 1 ]);
      ("W", W, [ 2; 2 ]);
    ];
  [%expect
    {|
    D: 2 outputs, 6 nodes, identical=true
    C: 3 outputs, 8 nodes, identical=true
    H: 3 outputs, 8 nodes, identical=true
    W: 2 outputs, 6 nodes, identical=true |}]

(* ---- reductions that re-pack, and layers on the frame ---------------------- *)

let flat_to outs =
  List.fold_right
    (fun (out, numel) acc ->
      let open Graph_builder in
      let* acc = acc in
      let* out = reshape { Reshape.Reshape.shape = flat numel } out in
      Graph_builder.return (out :: acc))
    outs (Graph_builder.return [])

let%expect_test "reductions dropping H, W or C under a fused batch" =
  List.iter
    (fun (name, shape) ->
      let n = numel shape in
      let ext a = Dim.to_int (Vec6.get shape a) in
      let g =
        build "repack"
          (let open Graph_builder in
           let* x = input ~shape:(s 1 1 1 1 n 1) () in
           let* r = reshape { Reshape.Reshape.shape } x in
           let* r = mul_scalar 0.75 r in
           let* hi = amax { Reduce.Amax.dims = [ H ]; keepdim = false } r in
           let* avg = mean { Reduce.Mean.dims = [ H; W ]; keepdim = false } r in
           let* total = sum { Reduce.Sum.dims = [ C ]; keepdim = false } r in
           let* outs =
             flat_to
               [
                 (hi, n / ext H); (avg, n / (ext H * ext W)); (total, n / ext C);
               ]
           in
           Graph_builder.return outs)
        |> ok
      in
      distinct_compare name ~x_shapes:[ s 1 1 1 1 n 1 ] g)
    [
      ("T and D", s 1 2 3 4 3 5);
      ("D", s 1 1 6 4 3 5);
      ("N, T and D", s 2 3 2 4 3 5);
    ];
  [%expect
    {|
    T and D: 3 outputs, 11 nodes, identical=true
    D: 3 outputs, 11 nodes, identical=true
    N, T and D: 3 outputs, 11 nodes, identical=true |}]

let%expect_test "linear and pooling over a fused batch" =
  let shape = s 1 2 3 4 4 6 in
  let n = numel shape in
  let avg : Pool.AvgPool2d.params =
    {
      ceil_mode = false;
      count_include_pad = true;
      kernel = Fixtures.hw (Dim.extent 2);
      stride = Fixtures.hw (Op_config.Pos.of_int 2);
      pad = Fixtures.hw (Op_config.Nonneg.of_int 0);
    }
  in
  let g =
    build "layers"
      (let open Graph_builder in
       let* x = input ~shape:(s 1 1 1 1 n 1) () in
       let* w = input ~shape:(s 5 1 1 1 1 6) () in
       let* r = reshape { Reshape.Reshape.shape } x in
       let* l =
         linear { Linear.Linear.in_features = Dim.extent 6 } ~x:r ~weight:w ()
       in
       let* hi = max_pool2d Fixtures.pool_params l in
       let* mid = avg_pool2d avg l in
       let* outs = flat_to [ (hi, 120); (mid, 120) ] in
       Graph_builder.return outs)
    |> ok
  in
  distinct_compare "layers" ~x_shapes:[ s 1 1 1 1 n 1; s 5 1 1 1 1 6 ] g;
  [%expect {| layers: 2 outputs, 6 nodes, identical=true |}]

(* T = 2, D = 1 against T = 1, D = 2 both fuse to 2 but broadcast to 4: the
   relabelled graph's own shapes disagree with the source's, so validation drops
   the group and the ordinary path names the blocker. *)
let%expect_test "operands whose fused batches only look alike" =
  let g =
    build "mismatch"
      (let open Graph_builder in
       let* a = input ~shape:(s 1 1 1 1 24 1) () in
       let* b = input ~shape:(s 1 1 1 1 24 1) () in
       let* a = reshape { Reshape.Reshape.shape = s 1 2 1 2 3 2 } a in
       let* b = reshape { Reshape.Reshape.shape = s 1 1 2 2 3 2 } b in
       let* sum = add a b in
       let* out = reshape { Reshape.Reshape.shape = flat 48 } sum in
       Graph_builder.return [ out ])
    |> ok
  in
  distinct_compare "mismatch" ~x_shapes:[ s 1 1 1 1 24 1; s 1 1 1 1 24 1 ] g;
  [%expect
    {| mismatch: tensor t2 has extent on T or D: [T=2 D=1 H=2 W=3 C=2] |}]
