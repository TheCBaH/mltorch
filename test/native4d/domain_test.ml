(* The Native4D domain contract, as a table. Each row is a Native graph and the
   answer [Domain.check] gives for it; together they ARE the partiality contract
   of .ai/native4d_design.md §1, pinned independently of any Native4D IR.

   Rows change visibly as later stages land — the canonical pipeline (stage 1)
   removes the two ops that are rejected here only because nothing has rewritten
   them yet. That is the point of pinning the table first. *)

let pp_ok fmt () = Fmt.string fmt "in the dialect"

let pp_outcome =
  Core.Pretty.err_result ~ok:pp_ok ~error:(Native4d.Error.pp :> _ Fmt.t)

(* Validation is [Graph_view]'s job, not the domain check's, so a fixture that
   fails it is a broken fixture and should say so loudly rather than be reported
   as out-of-domain. *)
let check name g =
  let view =
    Graph_view.of_graph g
    |> Err.or_raise ~pp_error:(fun ppf e ->
        Fmt.pf ppf "fixture %s is not a valid graph: %a" name
          Graph_view.pp_error e)
  in
  Format.printf "%-28s %a@." name pp_outcome (Native4d.Domain.check view)

let table rows = List.iter (fun (name, g) -> check name (g ())) rows

(* ---- the four-axis invariant ---------------------------------------------- *)

let%expect_test "domain: the four-axis invariant" =
  table
    [
      ("all-four-axis", Fixtures.all_four_axis);
      ("non-unit T", Fixtures.non_unit_t);
      ("non-unit D", Fixtures.non_unit_d);
      ("unread non-4D constant", Fixtures.unread_constant_non_4d);
      ("read non-4D constant", Fixtures.read_constant_non_4d);
      ("unused non-4D input", Fixtures.unused_input_non_4d);
    ];
  [%expect
    {|
    all-four-axis                in the dialect
    non-unit T                   tensor t0 has extent on T or D: [T=2 D=1 H=4 W=4
                                                                  C=3]
    non-unit D                   tensor t0 has extent on T or D: [D=2 H=4 W=4
                                                                  C=3]
    unread non-4D constant       in the dialect
    read non-4D constant         tensor t1 has extent on T or D: [D=5 H=1 W=1
                                                                  C=1]
    unused non-4D input          tensor t1 has extent on T or D: [D=5 H=4 W=4
                                                                  C=3] |}]

(* ---- axis-naming ops ------------------------------------------------------ *)

(* Four ops name axes and the predicate differs for each; [Permute] is the one
   that cannot be "names no axis outside N/H/W/C", since a valid Native perm is a
   full six-axis bijection and so always names T and D. *)
let%expect_test "domain: which axes an op may name" =
  table
    [
      ("mean over H,W", Fixtures.mean_over_hw ~keepdim:true ~n:1);
      ("mean over D", Fixtures.mean_over_d);
      ("permute H<->W", Fixtures.permute_hw);
      ("permute C onto D", Fixtures.permute_c_onto_d);
      ("rms_norm over C", Fixtures.rms_norm_over [ Axis.C ]);
      ("rms_norm over D", Fixtures.rms_norm_over [ Axis.D ]);
      ("layer_norm over C", Fixtures.layer_norm_over [ Axis.C ]);
      ("layer_norm over W,C", Fixtures.layer_norm_over [ Axis.W; Axis.C ]);
      ("layer_norm over D", Fixtures.layer_norm_over [ Axis.D ]);
      ("batch_norm on C", Fixtures.batch_norm_on Axis.C);
      ("batch_norm on H", Fixtures.batch_norm_on Axis.H);
    ];
  [%expect
    {|
    mean over H,W                in the dialect
    mean over D                  node n0: axis D is outside the N/H/W/C dialect
    permute H<->W                in the dialect
    permute C onto D             node n0: axis D is outside the N/H/W/C dialect
    rms_norm over C              in the dialect
    rms_norm over D              node n0: axis D is outside the N/H/W/C dialect
    layer_norm over C            in the dialect
    layer_norm over W,C          in the dialect
    layer_norm over D            node n0: axis D is outside the N/H/W/C dialect
    batch_norm on C              in the dialect
    batch_norm on H              node n0: axis H is outside the N/H/W/C dialect |}]

(* Constant kind is what makes the per-channel scale precomputable; a parameter
   that is still a node output — an unfolded relayout permute, say — is dynamic,
   which is why constant folding runs before conversion. *)
let%expect_test "domain: batch norm needs constant parameters" =
  table
    [
      ("constant params", Fixtures.batch_norm_on Axis.C);
      ("dynamic params", Fixtures.batch_norm_on ~dynamic:true Axis.C);
    ];
  [%expect
    {|
    constant params              in the dialect
    dynamic params               node n2: batch norm parameters are not all constant, so no per-channel scale can be precomputed |}]

(* ---- keepdim=false, the case design §7.5 gets wrong ----------------------- *)

(* [Reduce.Mean.kept_map] packs survivors onto the INNERMOST axes, right-aligned.
   Reducing H,W of [N,H,W,C] leaves survivors [N;T;D;C] packed onto [D;H;W;C], so
   N's extent lands on D — and the reshape §7.5 proposes has a target the dialect
   forbids. With N=1 every survivor is unit and it converts; with N=2 it does
   not, and the shape rule catches it without a special case. *)
let%expect_test "domain: mean keepdim=false depends on the batch extent" =
  table
    [
      ("keepdim=false, N=1", Fixtures.mean_over_hw ~keepdim:false ~n:1);
      ("keepdim=false, N=2", Fixtures.mean_over_hw ~keepdim:false ~n:2);
    ];
  [%expect
    {|
    keepdim=false, N=1           in the dialect
    keepdim=false, N=2           tensor t1 has extent on T or D: [D=2 H=1 W=1
                                                                  C=3] |}]

(* ---- convolution grouping ------------------------------------------------- *)

let%expect_test "domain: convolution grouping" =
  table
    [
      ( "groups=1",
        Fixtures.conv2d_grouped ~groups:1 ~in_channels:4 ~out_channels:8 );
      ( "depthwise",
        Fixtures.conv2d_grouped ~groups:4 ~in_channels:4 ~out_channels:4 );
      ( "groups=2, 2 per group",
        Fixtures.conv2d_grouped ~groups:2 ~in_channels:4 ~out_channels:4 );
      ( "transposed, groups=1",
        Fixtures.convolution_grouped ~transposed:true ~groups:1 ~channels:4 );
      ( "transposed, groups=2",
        Fixtures.convolution_grouped ~transposed:true ~groups:2 ~channels:4 );
    ];
  [%expect
    {|
    groups=1                     in the dialect
    depthwise                    in the dialect
    groups=2, 2 per group        in the dialect
    transposed, groups=1         in the dialect
    transposed, groups=2         node n0: transposed convolution has 2 groups; only 1 legalizes |}]

(* A legalization can be sound only under a precondition that SHAPE INFERENCE
   does not imply. Neither of these is the four-axis frame, and they are not the
   same kind of condition as each other either:

   - [Lossy_bmm_operand] is not about shape at all. It is about the operand's
     storage FORMAT, because the legalization materializes it.
   - [Batch_norm_extent] is about shape — just not a shape anything else checks.
     [Norm.BatchNorm.output_shape] is a function of the input alone, so a
     parameter's extent is never compared against the normalized axis.

   What they share is the useful predicate: neither is implied by the op's own
   output shape, which is all shape inference looks at. Both passed every
   structural check and then either computed the wrong answer or read out of
   bounds, so both are the domain check's business rather than the lowerer's. *)
let%expect_test "domain: preconditions shape inference does not imply" =
  table
    [
      ("bmm, i64 mat2", Fixtures.bmm_lossy_operand);
      ("batch_norm, short stats", Fixtures.batch_norm_short_stats);
    ];
  [%expect
    {|
    bmm, i64 mat2                node n0: bmm operand t1 is stored in a format f32 cannot hold exactly, and the legalization materializes it
    batch_norm, short stats      node n0: batch norm parameter t1 has extent 1 on C, but the normalized axis has 2 |}]

(* ---- bmm ------------------------------------------------------------------ *)

(* Only a single batch legalizes: for batch > 1 mat2 varies with the output's H
   coordinate, while convolution weights are shared across spatial positions. *)
let%expect_test "domain: bmm batch extent" =
  table [ ("batch=1", Fixtures.bmm_batch 1); ("batch=2", Fixtures.bmm_batch 2) ];
  [%expect
    {|
    batch=1                      in the dialect
    batch=2                      node n0: bmm batch extent is 2; only a single batch legalizes to a 1x1 convolution |}]

(* [Batched_matmul] has no admissible shape at all, the same argument as
   [Sdpa]'s own test below: unlike [Bmm], whose single-batch case legalizes
   to a 1x1 convolution, Native4D has no [Ops4] counterpart at any [D] extent
   (including 1), so there is no escape hatch to check for. *)
let%expect_test "domain: batched_matmul is always outside the dialect" =
  table [ ("batched_matmul", Fixtures.batched_matmul) ];
  [%expect
    {|
    batched_matmul               node n0: batched matmul's batch axis is D, which the N/H/W/C dialect has no name for; no legalization is available (the same `D`-axis argument attention_design.md §9 already made for Sdpa) |}]

(* [Sdpa] has no admissible shape at all: unlike [Bmm], whose single-batch
   case legalizes to a 1x1 convolution, Native4D has neither a [Bmm] nor a
   softmax to legalize through, and the op names D as batch regardless of
   configuration (op8-impl.md F8). *)
let%expect_test "domain: sdpa is always outside the dialect" =
  table [ ("sdpa", Fixtures.sdpa) ];
  [%expect
    {|
    sdpa                         node n0: scaled-dot-product attention's batch axis is D, which the N/H/W/C dialect has no name for; no legalization is available (Native has no Bmm or softmax in Native4D) |}]

(* [Softmax] has no [Ops4] counterpart at any axis, unlike [Mean]/[Amax]/
   [Vector_norm]: every axis rejects with the plain "dialect does not have
   it at all" answer, not [check_dims]'s per-axis one. *)
let%expect_test "domain: softmax has no dialect counterpart at any axis" =
  table
    [
      ("softmax over C", Fixtures.softmax_over Axis.C);
      ("softmax over D", Fixtures.softmax_over Axis.D);
    ];
  [%expect
    {|
    softmax over C               node n0: no legalization for
                                   softmax x=t0 params={axis=C}
    softmax over D               node n0: no legalization for
                                   softmax x=t0 params={axis=D} |}]

(* [Expand4] now exists, and admits every [Expand] unconditionally here --
   unlike [Softmax] just above, whose rejection is an axis-domain question,
   whether an expansion stays four-axis is a fact about its OWN target
   shape (a plain [Vec6.shape] at the Native op), checked by [Graph_shape4]'s
   [four] wrap and the lowerer's [Shape4.of_vec6], not by this arm. *)
let%expect_test "domain: expand always admits; its target is checked as a shape"
    =
  table [ ("expand", Fixtures.expand) ];
  [%expect {| expand                       in the dialect |}]

let%expect_test
    "domain: repeat always admits; its multiplier is checked as a shape" =
  table [ ("repeat", Fixtures.repeat) ];
  [%expect {| repeat                       in the dialect |}]

(* [RepeatInterleave] gates its ONE named axis, the same rule [slice]'s own
   test applies. The D case is refused by the axis rule and names D --
   unlike [Repeat] above, since this op's axis IS named data, not folded
   into an opaque per-axis shape. *)
let%expect_test "domain: repeat_interleave gates its named axis" =
  table
    [
      ("repeat_interleave W", Fixtures.repeat_interleave_w);
      ("repeat_interleave D", Fixtures.repeat_interleave_d);
    ];
  [%expect
    {|
    repeat_interleave W          in the dialect
    repeat_interleave D          node n0: axis D is outside the N/H/W/C dialect |}]

(* ---- ops the pipeline is supposed to have removed ------------------------- *)

(* Both reject, for different reasons and with different futures: the discarded
   case becomes an ordinary [Max_pool2d] once stage 1 runs, while a live index
   needs an argmax-pool operation the dialect does not have. *)
let%expect_test "domain: max pool with indices" =
  table
    [
      ("indices discarded", Fixtures.maxpool_indices_discarded);
      ("indices live", Fixtures.maxpool_indices_live);
    ];
  [%expect
    {|
    indices discarded            node n0: no legalization for
                                   max_pool2d_with_indices
                                     x=t0
                                     params={kernel={h=2; w=2};
                                            stride={h=2; w=2};
                                            pad={h=0; w=0};
                                            ceil_mode=false}
    indices live                 node n0: max-pool index output t2 is live; the dialect has no argmax-pool operation |}]

(* ---- the same graphs, after canonicalization ------------------------------ *)

(* THE FLIP. The domain is a property of the CANONICAL graph, not of whatever the
   importer emitted, and this is where that stops being a claim in a comment.
   Discarded max-pool indices reject above and convert here, because
   [Pipeline.canonical] removes the [Discard] sink and then narrows the op — the
   two-pass sequence design §7.8 describes as one.

   The live-index row does NOT flip, and must not: no pass can remove an edge
   something reads, so it is outside the dialect however canonical the graph. *)
let canonical name g =
  let outcome =
    let open Err.Syntax in
    let* (Rewrite.Origin state) =
      (Rewrite.origin g :> (Rewrite.origin, Pass.error) Err.t)
    in
    let+ (Rewrite.Step (final, _)) =
      Pass.run_all state [ Pipeline.canonical ~fold:false ]
    in
    Rewrite.graph final
  in
  match outcome with
  | Error e ->
      Format.printf "%-28s pipeline: %a@." name Pass.pp_error (Err.Error.kind e)
  | Ok g -> check name g

let%expect_test "domain: canonicalization is what makes a graph convertible" =
  List.iter
    (fun (name, g) -> canonical name (g ()))
    [
      ("indices discarded", Fixtures.maxpool_indices_discarded);
      ("indices live", Fixtures.maxpool_indices_live);
    ];
  [%expect
    {|
    indices discarded            in the dialect
    indices live                 node n0: max-pool index output t2 is live; the dialect has no argmax-pool operation |}]

(* ---- legalizations that need no domain constraint ------------------------- *)

let%expect_test "domain: linear is in the dialect" =
  table [ ("linear", Fixtures.linear_layer) ];
  [%expect {| linear                       in the dialect |}]

(* Only the PADDED axes are gated. The first fixture pads H and W and is inside
   the dialect; the second pads D on a graph whose other axes are unremarkable,
   and is refused by the axis rule with D named — the diagnostic that says which
   entry to remove, rather than one about a tensor that grew an extent. *)
let%expect_test "domain: pad gates the padded axes, and only those" =
  table [ ("pad H and W", Fixtures.pad_hw_crop); ("pad D", Fixtures.pad_d) ];
  [%expect
    {|
    pad H and W                  in the dialect
    pad D                        node n0: axis D is outside the N/H/W/C dialect |}]

(* Slice gates its ONE axis, the same rule pad applies to its padded ones. The
   D case is refused by the axis rule and names D. *)
let%expect_test "domain: slice gates the sliced axis" =
  table [ ("slice W", Fixtures.slice_w); ("slice D", Fixtures.slice_d) ];
  [%expect
    {|
    slice W                      in the dialect
    slice D                      node n0: axis D is outside the N/H/W/C dialect |}]

(* Concat gates its ONE axis, the same rule slice applies to its sliced one.
   The D case is refused by the axis rule and names D. *)
let%expect_test "domain: concat gates the joined axis" =
  table [ ("concat W", Fixtures.concat_w); ("concat D", Fixtures.concat_d) ];
  [%expect
    {|
    concat W                     in the dialect
    concat D                     node n0: axis D is outside the N/H/W/C dialect |}]

(* [Stack]'s axis check and its shape consequence are DIFFERENT rejections,
   the same split [Unbind]'s pair below draws -- but DROPPING an axis and
   INSERTING one land the precondition on different sides: [Unbind]'s shape
   consequence tracks N (the outermost axis), [Stack]'s tracks H, per
   [Ops4_split]'s own [Stack4] comment. *)
let%expect_test "domain: stack's axis rule and its shape consequence" =
  table
    [
      ("stack N, H=2", Fixtures.stack_n);
      ("stack H, H=1", Fixtures.stack_h_batch1);
      ("stack H, H=2", Fixtures.stack_h_nonunit);
      ("stack T (rank-5)", Fixtures.stack_rank5_t);
    ];
  [%expect
    {|
    stack N, H=2                 in the dialect
    stack H, H=1                 in the dialect
    stack H, H=2                 tensor t2 has extent on T or D: [D=2 H=2 W=2
                                                                  C=3]
    stack T (rank-5)             node n0: axis T is outside the N/H/W/C dialect |}]

(* [Unbind]'s axis check and its shape consequence are DIFFERENT rejections, and
   the ordering is what makes the first one useful: node predicates run before
   the shape rule, so a rank-five unbind naming T reports the axis rather than
   the tensor that ended up with extent on it.

   The batch-2 C case is the one that shows the axis check is not the whole
   story — C is legal to name, and the graph is still outside the dialect
   because the slices are not four-axis. *)
let%expect_test "domain: unbind's axis rule and its shape consequence" =
  table
    [
      ("unbind C, batch 1", Fixtures.unbind_c_batch1);
      ("unbind N, batch 2", Fixtures.unbind_n);
      ("unbind C, batch 2", Fixtures.unbind_c_batch2);
      ("unbind T (rank-5 ViT)", Fixtures.unbind_rank5_t);
    ];
  [%expect
    {|
    unbind C, batch 1            in the dialect
    unbind N, batch 2            in the dialect
    unbind C, batch 2            tensor t1 has extent on T or D: [T=2 D=1 H=1 W=2
                                                                  C=2]
    unbind T (rank-5 ViT)        node n0: axis T is outside the N/H/W/C dialect |}]

(* [Select]'s own version of the [Unbind] pair above: the axis check and its
   shape consequence are different rejections, for the same reason -- DROPPING
   the axis is what makes the shape consequence possible at all, unlike
   [Split_with_sizes] below. *)
let%expect_test "domain: select's axis rule and its shape consequence" =
  table
    [
      ("select C, batch 1", Fixtures.select_c_batch1);
      ("select N, batch 2", Fixtures.select_n);
      ("select C, batch 2", Fixtures.select_c_batch2);
      ("select T (rank-5)", Fixtures.select_rank5_t);
    ];
  [%expect
    {|
    select C, batch 1            in the dialect
    select N, batch 2            in the dialect
    select C, batch 2            tensor t1 has extent on T or D: [T=2 D=1 H=1 W=2
                                                                  C=2]
    select T (rank-5)            node n0: axis T is outside the N/H/W/C dialect |}]

(* [Split_with_sizes]'s axis check, the same rejection [Unbind]'s gets and for
   the same reason -- but with no shape consequence to pair it against: KEEPING
   the axis means batch 2 converts on its own (unlike [unbind_c_batch2]), so
   the axis rule is the whole story here. *)
(* [Select_scatter] gates its ONE named axis, the same rule [slice]'s own
   test applies -- not [select]'s pair above, since [Select_scatter]'s
   output is [self_shape] unchanged (no drop, no repack), so there is no
   separate shape-consequence rejection to demonstrate. *)
let%expect_test "domain: select_scatter gates its named axis" =
  table
    [
      ("select_scatter W", Fixtures.select_scatter_w);
      ("select_scatter D", Fixtures.select_scatter_d);
    ];
  [%expect
    {|
    select_scatter W             in the dialect
    select_scatter D             node n0: axis D is outside the N/H/W/C dialect |}]

let%expect_test "domain: split_with_sizes's axis rule" =
  table
    [
      ("split_with_sizes W, batch 2", Fixtures.split_with_sizes_w_batch2);
      ("split_with_sizes T (rank-5)", Fixtures.split_with_sizes_rank5_t);
    ];
  [%expect
    {|
    split_with_sizes W, batch 2  in the dialect
    split_with_sizes T (rank-5)  node n0: axis T is outside the N/H/W/C dialect |}]
