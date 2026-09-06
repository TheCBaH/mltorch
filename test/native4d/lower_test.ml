(* Native to Native4D conversion: what converts, what it produces, and what it
   refuses. The rows here are the same contract table [domain_test.ml] pins,
   carried through to an actual destination graph and map — a graph that passes
   the domain check but cannot be built is a lowerer bug, not a domain one.

   Split from a single lower_test.ml under the tracked file-size ceiling
   (scripts/check-file-size.sh): this file covers the factory/norm/precondition
   rows; lower_shape_test.ml covers the reshape/expand/repeat/transpose family.
   lower_fixtures.ml has the shared rendering helpers ([build]/[show]/
   [outcome]). *)

open Native4d
open Lower_fixtures

let%expect_test "lower: Zeros becomes the direct Zeros4 counterpart" =
  let source =
    build "zeros"
      (Graph_builder.zeros
         {
           Factory.Zeros.shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:4;
           fmt = Payload.Fmt Payload.F32;
         })
  in
  show "zeros" source;
  [%expect
    {|
    zeros:
      graph4
    inputs: []
    nodes:
      n0: [t0] = zeros4 shape=[N=1 H=2 W=3 C=4] fmt=f32
    outputs: [t0 [H=2 W=3 C=4]] |}]

let%expect_test "lower: Arange becomes the direct Arange4 counterpart" =
  let source =
    build "arange"
      (Graph_builder.arange
         {
           Factory.Arange.start = 0.5;
           stop = 4.;
           step = 1.;
           fmt = Payload.Fmt Payload.F32;
         })
  in
  show "arange" source;
  [%expect
    {|
    arange:
      graph4
    inputs: []
    nodes:
      n0: [t0] = arange4 start=0.5 stop=4 step=1 fmt=f32
    outputs: [t0 [C=4]] |}]

let%expect_test "lower: batch_norm_no_stats retains all three reductions" =
  let source =
    Graph_builder.build ~name:"batch_norm_no_stats" ~outputs:Fun.id
      (let open Graph_builder in
       let* x = input ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:2) () in
       let c_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
       let* weight = input ~shape:c_shape () in
       let* bias = input ~shape:c_shape () in
       batch_norm_no_stats
         { Norm.BatchNormNoStats.channel = Axis.C; eps = 1e-5 }
         ~x ~weight ~bias ())
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  show "batch_norm_no_stats" source;
  [%expect
    {|
    batch_norm_no_stats:
      graph4
    inputs: [t0 [H=2 W=1 C=2],
    t1 [C=2],
    t2 [C=2]]
    nodes:
      n0: [t3,
        t4,
        t5] =
        batch_norm_no_stats x=t0 weight=t1 bias=t2 params={channel=C; eps=1e-05}
    outputs: [t3 [H=2 W=1 C=2],
    t4 [C=2],
    t5 [C=2]] |}]

(* Only the channel axis converts, the same [Batch_norm_no_stats] shape --
   [Domain.check_node] has already gated it to C -- but single-output, and
   [groups] crosses unchanged since it names no axis. *)
let%expect_test "lower: group_norm becomes the direct Group_norm4 counterpart" =
  let source =
    Graph_builder.build ~name:"group_norm"
      ~outputs:(fun o -> [ o ])
      (let open Graph_builder in
       let* x = input ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:4) () in
       let c_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:4 in
       let* weight = input ~shape:c_shape () in
       let* bias = input ~shape:c_shape () in
       group_norm
         {
           Norm.GroupNorm.channel = Axis.C;
           groups = Op_config.Pos.of_int 2;
           eps = 1e-5;
         }
         ~x ~weight ~bias ())
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  show "group_norm" source;
  [%expect
    {|
    group_norm:
      graph4
    inputs: [t0 [H=2 W=2 C=4],
    t1 [C=4],
    t2 [C=4]]
    nodes:
      n0: [t3] =
        group_norm4
          x=t0
          weight=t1
          bias=t2
          params={channel=C; groups=2; eps=1e-05}
    outputs: [t3 [H=2 W=2 C=4]] |}]

(* Direct counterpart, unconditionally (project step 15): [Lstm.Lstm.t]
   names no axis and its shapes hardcode [T=1,D=1] everywhere, so it crosses
   unchanged and needs no [Domain.check_node] precondition the way [Sdpa]
   does. [seq=3]/[batch=2] are distinct so a layout bug swapping [H]/[W]
   would be visible in the printed shapes below, for BOTH layouts --
   [batch_first] only ever chooses which of [H]/[W] carries which, and both
   are dialect axes either way. *)
let lstm_layout_source ~batch_first =
  Graph_builder.build ~name:"lstm"
    ~outputs:(fun (o, h, c) -> [ o; h; c ])
    (let open Graph_builder in
     let seq_shape =
       if batch_first then Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:2
       else Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:2 ~c:2
     in
     let state_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:2 in
     let* x = input ~shape:seq_shape () in
     let* weight_ih =
       input ~shape:(Vec6.shape ~n:8 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2) ()
     in
     let* weight_hh =
       input ~shape:(Vec6.shape ~n:8 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2) ()
     in
     let* h0 = input ~shape:state_shape () in
     let* c0 = input ~shape:state_shape () in
     lstm
       { Lstm.Lstm.hidden_size = 2; input_size = 2; batch_first }
       ~input:x
       ~layers:
         [
           {
             Lstm.Lstm.Layer.forward =
               { Lstm.Lstm.Direction.weight_ih; weight_hh; bias = None };
             reverse = None;
           };
         ]
       ~h0 ~c0 ())
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let%expect_test
    "lower: lstm keeps layout-dependent sequence placement, both ways" =
  show "lstm time-first" (lstm_layout_source ~batch_first:false);
  show "lstm batch-first" (lstm_layout_source ~batch_first:true);
  [%expect
    {|
    lstm time-first:
      graph4
    inputs: [t0 [H=3 W=2 C=2],
    t1 [N=8 T=1 D=1 H=1 W=1 C=2],
    t2 [N=8 T=1 D=1 H=1 W=1 C=2],
    t3 [W=2 C=2],
    t4 [W=2 C=2]]
    nodes:
      n0: [t5,
        t6,
        t7] =
        lstm
          input=t0
          layers=[{forward={weight_ih=t1; weight_hh=t2; bias=none}; reverse=none}]
          h0=t3
          c0=t4
          params={hidden_size=2; input_size=2; batch_first=false}
    outputs: [t5 [H=3 W=2 C=2],
    t6 [W=2 C=2],
    t7 [W=2 C=2]]
    lstm batch-first:
      graph4
    inputs: [t0 [H=2 W=3 C=2],
    t1 [N=8 T=1 D=1 H=1 W=1 C=2],
    t2 [N=8 T=1 D=1 H=1 W=1 C=2],
    t3 [W=2 C=2],
    t4 [W=2 C=2]]
    nodes:
      n0: [t5,
        t6,
        t7] =
        lstm
          input=t0
          layers=[{forward={weight_ih=t1; weight_hh=t2; bias=none}; reverse=none}]
          h0=t3
          c0=t4
          params={hidden_size=2; input_size=2; batch_first=true}
    outputs: [t5 [H=2 W=3 C=2],
    t6 [W=2 C=2],
    t7 [W=2 C=2]] |}]

let%expect_test
    "lower: carries a six-dimensional captured constant behind a \
     four-dimensional export" =
  let source_to_native4d =
    Fixtures.perm_of
      [ (Axis.N, Axis.T); (Axis.T, Axis.N); (Axis.H, Axis.W); (Axis.W, Axis.H) ]
  in
  let source =
    build "symbolic_const"
      (let open Graph_builder in
       let* x = input ~shape:(Fixtures.nhwc ~n:1 ~h:3 ~w:3 ~c:2) () in
       (* The captured source has T=3, so it is genuinely outside Native4D.
          The Const-SSA permutation moves that axis onto N, leaving T=D=1 for
          the exported convolution weight. *)
       let* w = constant ~shape:(Fixtures.s 1 3 1 2 2 2) () in
       let* wp = permute source_to_native4d w in
       let base = Fixtures.conv_params ~in_channels:2 ~groups:1 in
       let params =
         {
           base with
           h = Fixtures.conv_axis ~kernel:2;
           w = Fixtures.conv_axis ~kernel:2;
         }
       in
       conv2d params ~x ~weight:wp ())
  in
  let weight =
    Tensor_id.Map.find (Tensor_id.of_int 1) source.Graph_ir.Graph.tensors
  in
  let store =
    Constant_store.bind_captured Constant_store.empty ~tensor:weight
      (Const_ssa.Capture.of_string "layer.weight")
    |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  match Rewrite.origin ~constant_store:store source with
  | Error e -> Format.printf "%a@." Rewrite.pp_error (Err.Error.kind e)
  | Ok (Rewrite.Origin state) ->
      (match Pass.run_all state [ Fold_const.pass ] with
      | Error e -> Format.printf "%a@." Pass.pp_error (Err.Error.kind e)
      | Ok (Rewrite.Step (folded, _)) -> (
          match
            Lower.convert
              ~constant_store:(Rewrite.constant_store folded)
              (Rewrite.snapshot folded)
          with
          | Error e -> Format.printf "%a@." Error.pp (Err.Error.kind e)
          | Ok (Lower.Pack lowered) -> (
              Format.printf "%a@." Constant_store.pp
                lowered.Lower.constant_store;
              (match
                 Framework.Verify_from_native.run
                   ~src_constant_store:(Rewrite.constant_store folded)
                   ~dst_constant_store:lowered.Lower.constant_store
                   lowered.Lower.map ~src:(Rewrite.snapshot folded)
                   ~dst:lowered.Lower.dst
               with
              | Error e ->
                  Format.printf "verify: %a@." Map_verify.pp_error
                    (Err.Error.kind e)
              | Ok report
                when Map_verify.Policy.accepts Map_verify.Policy.Require_proved
                       report ->
                  Format.printf "verify: %s@."
                    (Map_verify.Report.summary report)
              | Ok report ->
                  Format.printf "verify rejected: %s@."
                    (Map_verify.Report.summary report));
              let input =
                Tensor.materialize (Fixtures.nhwc ~n:1 ~h:3 ~w:3 ~c:2) (fun c ->
                    float_of_int
                      ((Dim.to_int (Vec6.get c Axis.H) * 10)
                      + Dim.to_int (Vec6.get c Axis.W)))
              in
              (match
                 Eval_direct4.run (Lower.graph lowered)
                   ~constants:(Tensor_id.Map.bindings lowered.Lower.constants)
                   ~inputs:[ (Tensor_id.of_int 0, input) ]
               with
              | Error e ->
                  Format.printf "before materialization: %a@."
                    Eval_direct4.pp_error (Err.Error.kind e)
              | Ok _ ->
                  Format.printf "before materialization: unexpectedly ran@.");
              match
                Lower.evaluate
                  (fun _ ->
                    Err.return
                      (Tensor.materialize (Fixtures.s 1 3 1 2 2 2) (fun c ->
                           float_of_int
                             ((Dim.to_int (Vec6.get c Axis.H) * 10)
                             + Dim.to_int (Vec6.get c Axis.W)))))
                  lowered
                  ~inputs:[ (Tensor_id.of_int 0, input) ]
              with
              | Error e ->
                  Format.printf "%a@." Lower.pp_eval_error (Err.Error.kind e)
              | Ok (env, report) ->
                  Format.printf "applies=%d@." report.applies;
                  Format.printf "%a@." Tensor.pp
                    (Tensor_id.Map.find (Tensor_id.of_int 3) env))));
      [%expect
        {|
    exports:
    t2 -> t2
    plan:
    t1 = captured "layer.weight"
    t2 = permute x=t1 perm=[N<-T, T<-N, H<-W, W<-H]
    verify: 3 clusters: 3 proved (structural)
    before materialization: missing constant tensor t2
    applies=1
    tensor f32 [H=2 W=2 C=3] {282, 282, 282, 326, 326, 326, 722, 722, ...} |}]

(* ---- the direct legalizations --------------------------------------------- *)

let%expect_test "lower: pointwise and pooling pass straight through" =
  show "relu"
    (build "relu"
       (let open Graph_builder in
        let* x = input ~shape:(Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3) () in
        relu x));
  [%expect
    {|
    relu:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n0: [t1] = relu x=t0
    outputs: [t1 [H=2 W=2 C=3]] |}]

let%expect_test "lower: Group 5 activations pass straight through" =
  show "silu"
    (build "silu"
       (let open Graph_builder in
        let* x = input ~shape:(Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3) () in
        silu x));
  show "sigmoid"
    (build "sigmoid"
       (let open Graph_builder in
        let* x = input ~shape:(Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3) () in
        sigmoid x));
  show "gelu"
    (build "gelu"
       (let open Graph_builder in
        let* x = input ~shape:(Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3) () in
        gelu Pointwise.Gelu.Exact x));
  show "gelu tanh"
    (build "gelu tanh"
       (let open Graph_builder in
        let* x = input ~shape:(Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3) () in
        gelu Pointwise.Gelu.Tanh x));
  show "hardsigmoid"
    (build "hardsigmoid"
       (let open Graph_builder in
        let* x = input ~shape:(Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3) () in
        hardsigmoid x));
  show "hardswish"
    (build "hardswish"
       (let open Graph_builder in
        let* x = input ~shape:(Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3) () in
        hardswish x));
  [%expect
    {|
    silu:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n0: [t1] = silu x=t0
    outputs: [t1 [H=2 W=2 C=3]]
    sigmoid:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n0: [t1] = sigmoid x=t0
    outputs: [t1 [H=2 W=2 C=3]]
    gelu:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n0: [t1] = gelu x=t0 approximate=none
    outputs: [t1 [H=2 W=2 C=3]]
    gelu tanh:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n0: [t1] = gelu x=t0 approximate=tanh
    outputs: [t1 [H=2 W=2 C=3]]
    hardsigmoid:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n0: [t1] = hardsigmoid x=t0
    outputs: [t1 [H=2 W=2 C=3]]
    hardswish:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n0: [t1] = hardswish x=t0
    outputs: [t1 [H=2 W=2 C=3]] |}]

let%expect_test "lower: scalar pointwise ops pass straight through" =
  show "add_scalar"
    (build "add_scalar"
       (let open Graph_builder in
        let* x = input ~shape:(Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3) () in
        add_scalar 0.5 x));
  show "mul_scalar"
    (build "mul_scalar"
       (let open Graph_builder in
        let* x = input ~shape:(Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3) () in
        mul_scalar 2. x));
  [%expect
    {|
    add_scalar:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n0: [t1] = add_scalar x=t0 scalar=0.5
    outputs: [t1 [H=2 W=2 C=3]]
    mul_scalar:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n0: [t1] = mul_scalar x=t0 scalar=2
    outputs: [t1 [H=2 W=2 C=3]] |}]

(* §7.1: [Clone] contributes no node. Its output is tied to its input, which is
   a substitution rather than a rewrite — every later reference rewires, and the
   map records the two edges as one value. *)
let%expect_test "lower: clone is removed, not translated" =
  show "clone"
    (build "clone"
       (let open Graph_builder in
        let* x = input ~shape:(Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3) () in
        let* c = clone x in
        relu c));
  [%expect
    {|
    clone:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n1: [t2] = relu x=t0
    outputs: [t2 [H=2 W=2 C=3]] |}]

(* §7.3. The Native weight is already [Out,1,1,1,1,In] — a 1x1 convolution
   weight — so this is params-only, with no data movement and no change to the
   input-channel reduction order. *)
let%expect_test "lower: linear becomes a 1x1 convolution" =
  show "linear" (Fixtures.linear_layer ());
  [%expect
    {|
    linear:
      graph4
    inputs: [t0 [C=8],
    t1 [N=4 T=1 D=1 H=1 W=1 C=8]]
    nodes:
      n0: [t2] =
        conv2d
          x=t0
          weight=t1
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=8}
    outputs: [t2 [C=4]] |}]

(* [Conv1d] carries no [h] field at all (conv_conv1d.ml) -- its own
   [to_conv2d_params] supplies [Conv2d.unit_window] before reusing
   [forward_conv] unchanged, so [Conv2D4] shows the pinned unit window on
   [h] even though nothing in [Conv1d]'s own payload named it. *)
let%expect_test "lower: conv1d becomes conv2d with H pinned to the unit window"
    =
  show "conv1d" (Fixtures.conv1d_basic ~in_channels:2 ~out_channels:3 ());
  [%expect
    {|
    conv1d:
      graph4
    inputs: [t0 [W=4 C=2],
    t1 [N=3 T=1 D=1 H=1 W=1 C=2]]
    nodes:
      n0: [t2] =
        conv2d
          x=t0
          weight=t1
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=2}
    outputs: [t2 [W=4 C=3]] |}]

(* §7.4. [Bmm] legalizes directly to [Batched_matmul], one node, no
   relayout — retiring the previous permute-into-1x1-convolution route (which
   was sound only at batch 1). At batch 2, the SAME single-node lowering
   fires: there is no batch-shaped restriction left to impose (the domain
   test right above proves the admission; this proves what it converts TO). *)
let%expect_test
    "lower: bmm becomes batched_matmul directly, at any batch extent" =
  show "bmm batch=1" (Fixtures.bmm_batch 1 ());
  show "bmm batch=2" (Fixtures.bmm_batch 2 ());
  [%expect
    {|
    bmm batch=1:
      graph4
    inputs: [t0 [W=2 C=3],
    t1 [W=3 C=4]]
    nodes:
      n0: [t2] = batched_matmul input=t0 mat2=t1
    outputs: [t2 [W=2 C=4]]
    bmm batch=2:
      graph4
    inputs: [t0 [H=2 W=2 C=3],
    t1 [H=2 W=3 C=4]]
    nodes:
      n0: [t2] = batched_matmul input=t0 mat2=t1
    outputs: [t2 [H=2 W=2 C=4]] |}]

(* §7.5 as corrected by C1: keepdim=false is MeanKeepDims + Reshape4, and only
   when the packed shape re-enters the dialect. *)
let%expect_test "lower: mean keepdim=false gains a reshape" =
  show "mean" (Fixtures.mean_over_hw ~keepdim:false ~n:1 ());
  [%expect
    {|
    mean:
      graph4
    inputs: [t0 [H=4 W=4 C=3]]
    nodes:
      n0: [t2] = mean_keepdims x=t0 params={dims=[H, W]}
      n1: [t1] = reshape4 x=t2 params={shape=[N=1 H=1 W=1 C=3]}
    outputs: [t1 [C=3]] |}]

(* ---- what it refuses ------------------------------------------------------ *)

let%expect_test "lower: the domain's rejections are the lowerer's" =
  List.iter
    (fun (name, g) -> outcome name (g ()))
    [
      ("non-unit D", Fixtures.non_unit_d);
      ("unused non-4D input", Fixtures.unused_input_non_4d);
      ("mean over D", Fixtures.mean_over_d);
      ("live max-pool indices", Fixtures.maxpool_indices_live);
    ];
  [%expect
    {|
    non-unit D                 tensor t0 has extent on T or D: [D=2 H=4 W=4 C=3]
    unused non-4D input        tensor t1 has extent on T or D: [D=5 H=4 W=4 C=3]
    mean over D                node n0: axis D is outside the N/H/W/C dialect
    live max-pool indices      node n0: max-pool index output t2 is live; the dialect has no argmax-pool operation |}]

(* §7.2/§8: a grouping that is neither 1 nor depthwise used to be rejected here;
   it now normalizes to [GroupedConv2D], the general form, with [groups] itself
   carried as a genuine field rather than implied by the constructor. *)
let%expect_test "lower: general grouped convolution becomes GroupedConv2D" =
  show "conv groups=2"
    (Fixtures.conv2d_grouped ~groups:2 ~in_channels:4 ~out_channels:4 ());
  [%expect
    {|
    conv groups=2:
      graph4
    inputs: [t0 [H=4 W=4 C=4],
    t1 [N=4 T=1 D=1 H=1 W=1 C=2]]
    nodes:
      n0: [t2] =
        grouped_conv2d
          x=t0
          weight=t1
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=4;
                 groups=2}
    outputs: [t2 [H=4 W=4 C=4]] |}]

(* An unread constant is model-bound state, not interface, so it is OMITTED and
   recorded as a deletion — the seam [Rewrite.apply] already cuts along. The
   destination graph should simply not have it. *)
let%expect_test "lower: an unread non-4D constant is omitted" =
  show "unread constant" (Fixtures.unread_constant_non_4d ());
  [%expect
    {|
    unread constant:
      graph4
    inputs: [t0 [H=4 W=4 C=3]]
    nodes:
      n0: [t2] = relu x=t0
    outputs: [t2 [H=4 W=4 C=3]] |}]

(* ---- batch norm: the one Equivalent legalization -------------------------- *)

let chan c = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c
let fill shape v = Tensor.materialize shape (fun _ -> v)

let standalone_bn () =
  build "batch_norm"
    (let open Graph_builder in
     let* x = input ~shape:(Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:2) () in
     let* mean = constant ~shape:(chan 2) () in
     let* var = constant ~shape:(chan 2) () in
     batch_norm
       { Norm.BatchNorm.channel = Axis.C; eps = 0. }
       ~x ~running_mean:mean ~running_var:var ())

let bn_constants g =
  match g.Graph_ir.Graph.inputs with
  | [ _; mean; var ] ->
      Tensor_id.Map.of_seq
        (List.to_seq [ (mean, fill (chan 2) 1.); (var, fill (chan 2) 4.) ])
  | _ -> invalid_arg "unexpected fixture shape"

(* §7.6: precomputing one scale and offset per channel re-associates the
   arithmetic, so f32 rounding points move and the claim is Equivalent, not
   Identical. With mean=1, var=4, eps=0 the scale is 1/2 and the offset -1/2. *)
let described_map ?constants g =
  match Snapshot.create g with
  | Error e ->
      Format.asprintf "snapshot: %a" Graph_view.pp_error (Err.Error.kind e)
  | Ok (Snapshot.Pack src) -> (
      match Lower.convert ?constants src with
      | Error e -> Format.asprintf "%a" Error.pp (Err.Error.kind e)
      | Ok (Lower.Pack r) -> Format.asprintf "%a" Graph_map.pp r.Lower.map)

let%expect_test "lower: standalone batch norm becomes a depthwise 1x1" =
  let g = standalone_bn () in
  show "batch_norm" ~constants:(bn_constants g) g;
  (* THE CLAIM. Equivalent, not Identical — and the map has to say so, because
     an edge left implicit reads as Identical and claim closure would then have
     the verifier assert bit-equality on everything downstream. *)
  Format.printf "@[<v 2>map:@,%s@]@."
    (described_map ~constants:(bn_constants g) g);
  [%expect
    {|
    batch_norm:
      graph4
    inputs: [t0 [H=2 W=2 C=2],
    t1 [C=2],
    t2 [C=2],
    t4 [N=2 T=1 D=1 H=1 W=1 C=1],
    t5 [C=2]]
    nodes:
      n0: [t3] =
        depthwise_conv2d
          x=t0
          weight=t4
          bias=t5
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=2}
    outputs: [t3 [H=2 W=2 C=2]]
    map:
      values:
      {t3} -> {t3} equivalent
      {} -> {t4} identical
      {} -> {t5} identical
    nodes:
      identity
    provenance:
      {t2} -> t4
      {t1, t2} -> t5 |}]

(* Absence is an error only for a legalization that needs the payload AT
   CONVERSION TIME — batch norm alone. Every other constant is copied through
   unread, so demanding one would refuse graphs that convert fine. *)
let%expect_test "lower: batch norm without payloads cannot be precomputed" =
  outcome "no payloads" (standalone_bn ());
  [%expect
    {| no payloads                node n0 needs constant t1's payload at conversion time, and none was supplied |}]

(* Validity is different: checked over the WHOLE supplied map before any
   legalization, because every payload is copied into the result and read much
   later, far from the mistake. *)
let%expect_test "lower: a payload disagreeing with its signature is rejected" =
  let g = standalone_bn () in
  let wrong =
    match g.Graph_ir.Graph.inputs with
    | _ :: mean :: _ -> Tensor_id.Map.singleton mean (fill (chan 5) 1.)
    | _ -> invalid_arg "unexpected fixture shape"
  in
  outcome "wrong shape" ~constants:wrong g;
  [%expect
    {| wrong shape                constant t1's payload does not match its recorded signature |}]

(* ---- determinism ---------------------------------------------------------- *)

(* Ids, node order and diagnostics, all of which a map consumer depends on. Two
   independent conversions of the same graph must be indistinguishable. *)
let%expect_test "lower: conversion is deterministic" =
  let g = Fixtures.bmm_batch 1 () in
  let once = described g ~render:(Format.asprintf "%a" Graph.pp) in
  let twice = described g ~render:(Format.asprintf "%a" Graph.pp) in
  Format.printf "identical: %b@." (String.equal once twice);
  [%expect {| identical: true |}]

(* ---- one precondition, at the lowerer --------------------------------------

   Stated by [Domain.check] and pinned there — see domain_test.ml. This row
   checks the OTHER half: that [Lower.convert] refuses it too, and so never
   reaches the read that would go out of bounds. [Bmm] used to have a second
   precondition here (its mat2 operand's storage format, guarding the OLD
   permute-into-convolution legalization's materialization) — gone along with
   that legalization now that [Bmm] converts directly to [Batched_matmul],
   which reads mat2 the same way Native's own [Bmm.Compute] does. *)

let%expect_test "lower: the preconditions reach the lowerer" =
  List.iter
    (fun (name, g) -> outcome name (g ()))
    [ ("batch_norm, short stats", Fixtures.batch_norm_short_stats) ];
  [%expect
    {|
    batch_norm, short stats    node n0: batch norm parameter t1 has extent 1 on C, but the normalized axis has 2 |}]

(* The complete ordered output list has to survive. Under the ID policy a
   value-preserving edge keeps its SOURCE id, so all three slices appear in no
   cluster and are implicitly [Identical] — which means a lowering that dropped
   or reordered one would produce a graph that still validates, and the golden
   is where it shows.

   The graph output is deliberately slice 1, not slice 0: it makes the printed
   graph distinguish "kept the list" from "kept the first one". *)
let%expect_test "lower: unbind keeps every slice, in order" =
  show "unbind" (Fixtures.unbind_c_batch1 ());
  [%expect
    {|
    unbind:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n0: [t1, t2, t3] = unbind x=t0 params={axis=C}
    outputs: [t1 [W=2 C=2],
    t2 [W=2 C=2],
    t3 [W=2 C=2]] |}]

(* [Unbind]'s single-output sibling. Selecting C shifts every axis outside it
   one place inward -- H onto W and W onto C -- the same repacking the golden
   above shows for every slice at once. [index] crosses unchanged -- Native
   has already resolved it against the axis extent, so there is nothing left
   for this arm to restate. *)
let%expect_test "lower: select4 drops its axis and carries the index" =
  show "select" (Fixtures.select_c_batch1 ());
  [%expect
    {|
    select:
      graph4
    inputs: [t0 [H=2 W=2 C=3]]
    nodes:
      n0: [t1] = select4 x=t0 params={axis=C index=1}
    outputs: [t1 [W=2 C=2]] |}]

(* [Select4]'s write-back counterpart: unlike [select4] above, the output is
   [self]'s OWN shape unchanged (no drop, no repack), so there is nothing
   for this arm to re-validate the way [select4]'s does. [axis]/[index]
   cross the same way. *)
let%expect_test
    "lower: select_scatter4 writes src at one position, self elsewhere" =
  show "select_scatter" (Fixtures.select_scatter_w ());
  [%expect
    {|
    select_scatter:
      graph4
    inputs: [t0 [W=3 C=2],
    t1 [C=2]]
    nodes:
      n0: [t2] = select_scatter4 self=t0 src=t1 params={axis=W index=1}
    outputs: [t2 [W=3 C=2]] |}]

(* [Index_tensor]'s own version of [select_scatter4]'s test above: [self]'s
   named axis is overwritten by [index]'s own length rather than written at
   a compile-time position, but the shape story is identical -- no drop, no
   repack. *)
let%expect_test "lower: index_tensor4 gathers self's named axis by index" =
  show "index_tensor" (Fixtures.index_tensor_w ());
  [%expect
    {|
    index_tensor:
      graph4
    inputs: [t0 [W=3 C=2],
    t1 [C=2]]
    nodes:
      n0: [t2] = index_tensor4 self=t0 index=t1 params={axis=W}
    outputs: [t2 [W=2 C=2]] |}]

(* [Unbind]'s rank-preserving sibling: every window KEEPS the axis, in order,
   and [sizes] itself crosses unchanged -- Native has already bounded and
   summed it, so there is nothing left for this arm to restate. *)
let%expect_test "lower: split_with_sizes keeps every window, in order" =
  show "split_with_sizes" (Fixtures.split_with_sizes_w_batch2 ());
  [%expect
    {|
    split_with_sizes:
      graph4
    inputs: [t0 [N=2 T=1 D=1 H=2 W=4 C=3]]
    nodes:
      n0: [t1, t2] = split_with_sizes4 x=t0 params={axis=W sizes=[1, 3]}
    outputs: [t1 [N=2 T=1 D=1 H=2 W=1 C=3],
    t2 [N=2 T=1 D=1 H=2 W=3 C=3]] |}]

(* Only the axis KEYS convert; the signed amounts and the mode cross unchanged,
   which is what makes the claim [Identical] and is exactly what a golden over
   the printed op checks. A lowering that clamped the crop to zero, or dropped
   the second entry, prints differently here and nowhere else — the numeric
   comparison in verify_test.ml sees a shape change and refuses earlier. *)
let%expect_test "lower: pad4 carries both entries, signs and all" =
  show "pad" (Fixtures.pad_hw_crop ());
  [%expect
    {|
    pad:
      graph4
    inputs: [t0 [H=4 W=4 C=2]]
    nodes:
      n0: [t1] = pad4 x=t0 params={pads=[H:1,-1, W:0,2] mode=constant(0.25)}
    outputs: [t1 [H=4 W=6 C=2]] |}]

(* Only the axis converts; the three bounds cross unchanged, which is what makes
   the claim [Identical] and what the printed op checks. A lowering that dropped
   the step or swapped start and stop still produces a graph that validates -- on
   these bounds it even produces the same OUTPUT SHAPE for a start/stop swap
   above the ceiling -- so the golden is where it shows. *)
(* Only the axes convert; [eps] and both affine EDGES cross unchanged. A
   lowering that dropped an operand, swapped weight and bias, or mapped the
   wrong axis still produces a graph that validates and has the right output
   shape, so the printed op is where it shows. *)
let%expect_test "lower: layer_norm carries both affine operands" =
  show "one axis" (Fixtures.layer_norm_over [ Axis.C ] ());
  show "two axes" (Fixtures.layer_norm_over [ Axis.W; Axis.C ] ());
  [%expect
    {|
    one axis:
      graph4
    inputs: [t0 [H=4 W=4 C=3],
    t1 [C=3],
    t2 [C=3]]
    nodes:
      n0: [t3] = layer_norm x=t0 weight=t1 bias=t2 params={dims=[C]; eps=1e-05}
    outputs: [t3 [H=4 W=4 C=3]]
    two axes:
      graph4
    inputs: [t0 [H=4 W=4 C=3],
    t1 [W=4 C=3],
    t2 [W=4 C=3]]
    nodes:
      n0: [t3] =
        layer_norm x=t0 weight=t1 bias=t2 params={dims=[W, C]; eps=1e-05}
    outputs: [t3 [H=4 W=4 C=3]] |}]

let%expect_test "lower: slice4 carries all three bounds" =
  show "slice" (Fixtures.slice_w ());
  [%expect
    {|
    slice:
      graph4
    inputs: [t0 [H=2 W=5 C=1]]
    nodes:
      n0: [t1] = slice4 x=t0 params={axis=W start=1 stop=5 step=2}
    outputs: [t1 [H=2 W=2 C=1]] |}]

(* Only the axis converts; every operand crosses unchanged and in order, which
   is what makes the claim [Identical] and what the printed op checks. A
   lowering that dropped an operand or reordered the list still produces a
   graph that validates -- concat's shape rule does not care about operand
   ORDER along the non-joined axes -- so the golden is where it shows. *)
let%expect_test "lower: concat4 carries the axis and every operand in order" =
  show "concat" (Fixtures.concat_w ());
  [%expect
    {|
    concat:
      graph4
    inputs: [t0 [H=2 W=1 C=1],
    t1 [H=2 W=2 C=1]]
    nodes:
      n0: [t2] = concat4 xs=[t0, t1] params={axis=W}
    outputs: [t2 [H=2 W=3 C=1]] |}]
