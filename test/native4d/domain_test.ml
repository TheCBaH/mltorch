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
    groups=2, 2 per group        node n0: convolution has 2 groups, which is neither 1 nor depthwise
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
                                            pad={h=0; w=0}}
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
