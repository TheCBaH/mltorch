(* Native to Native4D conversion: what converts, what it produces, and what it
   refuses. The rows here are the same contract table [domain_test.ml] pins,
   carried through to an actual destination graph and map — a graph that passes
   the domain check but cannot be built is a lowerer bug, not a domain one. *)

open Native4d

let build name m =
  match Graph_builder.build ~name ~outputs:(fun o -> [ o ]) m with
  | Ok g -> g
  | Error e ->
      invalid_arg
        (Format.asprintf "fixture %s: %a" name Graph_builder.pp_error
           e.Core.Error.kind)

(* The result is packed over the destination version, so it cannot escape the
   scope that unpacks it — every consumer below renders inside. *)
let described ?constants g ~render =
  match Snapshot.create g with
  | Error e ->
      Format.asprintf "snapshot: %a" Graph_view.pp_error e.Core.Error.kind
  | Ok (Snapshot.Pack src) -> (
      match Lower.convert ?constants src with
      | Error e -> Format.asprintf "%a" Error.pp e.Core.Error.kind
      | Ok (Lower.Pack r) -> render (Lower.graph r))

(* Prints the destination graph, so a legalization producing the wrong op — or
   the wrong number of them — is visible rather than merely "ok". *)
let show name ?constants g =
  Format.printf "@[<v 2>%s:@,%s@]@." name
    (described ?constants g ~render:(Format.asprintf "%a" Graph.pp))

let outcome name ?constants g =
  Format.printf "%-26s %s@." name
    (described ?constants g ~render:(fun dst ->
         Format.asprintf "converted, %d nodes"
           (List.length dst.Graph_common.Graph.nodes)))

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

(* §7.4. mat2 permutes to the weight layout and the 1x1 convolution's output is
   already Bmm's own shape, so there is NO trailing reshape. *)
let%expect_test "lower: single-batch bmm becomes permute + 1x1 convolution" =
  show "bmm" (Fixtures.bmm_batch 1 ());
  [%expect
    {|
    bmm:
      graph4
    inputs: [t0 [W=2 C=3],
    t1 [W=3 C=4]]
    nodes:
      n0: [t3] = permute4 x=t1 perm=[N<-C, W<-N, C<-W]
      n1: [t2] =
        conv2d
          x=t0
          weight=t3
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=3}
    outputs: [t2 [W=2 C=4]] |}]

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
      ("bmm batch 2", Fixtures.bmm_batch 2);
      ( "conv groups=2",
        Fixtures.conv2d_grouped ~groups:2 ~in_channels:4 ~out_channels:4 );
      ("live max-pool indices", Fixtures.maxpool_indices_live);
    ];
  [%expect
    {|
    non-unit D                 tensor t0 has extent on T or D: [D=2 H=4 W=4 C=3]
    unused non-4D input        tensor t1 has extent on T or D: [D=5 H=4 W=4 C=3]
    mean over D                node n0: axis D is outside the N/H/W/C dialect
    bmm batch 2                node n0: bmm batch extent is 2; only a single batch legalizes to a 1x1 convolution
    conv groups=2              node n0: convolution has 2 groups, which is neither 1 nor depthwise
    live max-pool indices      node n0: max-pool index output t2 is live; the dialect has no argmax-pool operation |}]

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
      Format.asprintf "snapshot: %a" Graph_view.pp_error e.Core.Error.kind
  | Ok (Snapshot.Pack src) -> (
      match Lower.convert ?constants src with
      | Error e -> Format.asprintf "%a" Error.pp e.Core.Error.kind
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

(* ---- two preconditions, at the lowerer -------------------------------------

   Both are stated by [Domain.check] and pinned there — see domain_test.ml.
   These rows check the OTHER half: that [Lower.convert] refuses them too, and
   so never reaches the materialization that would round an operand or the read
   that would go out of bounds. *)

let%expect_test "lower: the preconditions reach the lowerer" =
  List.iter
    (fun (name, g) -> outcome name (g ()))
    [
      ("bmm, i64 mat2", Fixtures.bmm_lossy_operand);
      ("batch_norm, short stats", Fixtures.batch_norm_short_stats);
    ];
  [%expect
    {|
    bmm, i64 mat2              node n0: bmm operand t1 is stored in a format f32 cannot hold exactly, and the legalization materializes it
    batch_norm, short stats    node n0: batch norm parameter t1 has extent 1 on C, but the normalized axis has 2 |}]
