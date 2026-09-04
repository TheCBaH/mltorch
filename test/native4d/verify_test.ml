(* Symbolic verification of a CONVERSION map: the third of the three
   architectural claims §14 asks the milestone to prove — that the verifier can
   compare two different closed operation types at the same coordinates.

   It can because nothing below the two [Stage_program.t]s knows about dialects.
   Both sides ground to terms over the same [Expr], at the same [Vec6.coord],
   because the conversion preserves physical shapes (§4.1) — so no axis map is
   needed in a cluster and the existing machinery applies unchanged.

   Goldens carry VERDICTS, not magnitudes, following test/native/verify_test.ml:
   a report that named floating-point values would be platform-dependent. *)

open Native4d

let build = Fixtures.build

(* Convert, then check the map the conversion produced. Everything happens
   inside the scope that unpacks the destination version — the report is
   version-free and comes back out, the snapshot cannot. *)
let verified ?constants ?(effort = Map_verify.Effort.Thorough) g =
  match Snapshot.create g with
  | Error e ->
      Format.asprintf "snapshot: %a" Graph_view.pp_error (Err.Error.kind e)
  | Ok (Snapshot.Pack src) -> (
      match Lower.convert ?constants src with
      | Error e -> Format.asprintf "%a" Error.pp (Err.Error.kind e)
      | Ok (Lower.Pack r) -> (
          let budget = Map_verify.Effort.budget effort in
          match
            Framework.Verify_from_native.run ~budget
              ~probe:(Map_verify.Effort.probe effort)
              ?src_constants:(Option.map Fun.id constants)
              ~dst_constants:r.Lower.constants r.Lower.map ~src ~dst:r.Lower.dst
          with
          | Error e ->
              Format.asprintf "verify: %a" Map_verify.pp_error
                (Err.Error.kind e)
          | Ok report -> Map_verify.Report.summary report))

let check name ?constants g =
  Format.printf "%-24s %s@." name (verified ?constants g)

(* ---- the Identical legalizations ------------------------------------------ *)

(* Each of these claims the destination computes the same function of its
   corresponding dependencies. "Proved (structural)" means that holds for every
   input, not for one sample. *)
let%expect_test "verify: the direct and reinterpreting legalizations" =
  let nhwc = Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:2 in
  check "relu"
    (build "relu"
       (let open Graph_builder in
        let* x = input ~shape:nhwc () in
        relu x));
  check "add + relu"
    (build "add_relu"
       (let open Graph_builder in
        let* a = input ~shape:nhwc () in
        let* b = input ~shape:nhwc () in
        let* s = add a b in
        relu s));
  check "clone removal"
    (build "clone"
       (let open Graph_builder in
        let* x = input ~shape:nhwc () in
        let* c = clone x in
        relu c));
  check "silu"
    (build "silu"
       (let open Graph_builder in
        let* x = input ~shape:nhwc () in
        silu x));
  check "sigmoid"
    (build "sigmoid"
       (let open Graph_builder in
        let* x = input ~shape:nhwc () in
        sigmoid x));
  check "gelu"
    (build "gelu"
       (let open Graph_builder in
        let* x = input ~shape:nhwc () in
        gelu Pointwise.Gelu.Exact x));
  check "gelu tanh"
    (build "gelu tanh"
       (let open Graph_builder in
        let* x = input ~shape:nhwc () in
        gelu Pointwise.Gelu.Tanh x));
  check "mul_scalar"
    (build "mul_scalar"
       (let open Graph_builder in
        let* x = input ~shape:nhwc () in
        mul_scalar 2. x));
  check "add_scalar"
    (build "add_scalar"
       (let open Graph_builder in
        let* x = input ~shape:nhwc () in
        add_scalar 0.5 x));
  check "hardsigmoid"
    (build "hardsigmoid"
       (let open Graph_builder in
        let* x = input ~shape:nhwc () in
        hardsigmoid x));
  check "hardswish"
    (build "hardswish"
       (let open Graph_builder in
        let* x = input ~shape:nhwc () in
        hardswish x));
  (* Retained fused, so the claim is [Identical] and the verifier proves it
     STRUCTURALLY -- the two sides ground to the same term over the same [Expr],
     including both reductions and both affine operands. A decomposed lowering
     would only ever reach [Equivalent]. *)
  check "layer_norm" (Fixtures.layer_norm_over [ Axis.C ] ());
  check "layer_norm over W,C" (Fixtures.layer_norm_over [ Axis.W; Axis.C ] ());
  check "linear -> 1x1 conv" (Fixtures.linear_layer ());
  check "bmm -> batched_matmul" (Fixtures.bmm_batch 1 ());
  check "mean keepdim=false" (Fixtures.mean_over_hw ~keepdim:false ~n:1 ());
  check "permute alone"
    (build "permute"
       (let open Graph_builder in
        let* x = input ~shape:nhwc () in
        let* p =
          permute
            (Permute.Permute.of_fn (fun a ->
                 match a with Axis.H -> Axis.W | Axis.W -> Axis.H | a -> a))
            x
        in
        relu p));
  [%expect
    {|
    relu                     2 clusters: 2 proved (structural)
    add + relu               4 clusters: 4 proved (structural)
    clone removal            2 clusters: 2 proved (structural)
    silu                     2 clusters: 2 proved (structural)
    sigmoid                  2 clusters: 2 proved (structural)
    gelu                     2 clusters: 2 proved (structural)
    gelu tanh                2 clusters: 2 proved (structural)
    mul_scalar               2 clusters: 2 proved (structural)
    add_scalar               2 clusters: 2 proved (structural)
    hardsigmoid              2 clusters: 2 proved (structural)
    hardswish                2 clusters: 2 proved (structural)
    layer_norm               4 clusters: 2 proved (structural) [sampled 32], 2 unproved (unbound constant)
    layer_norm over W,C      4 clusters: 2 proved (structural) [sampled 32], 2 unproved (unbound constant)
    linear -> 1x1 conv       3 clusters: 1 proved (structural), 2 unproved (unbound constant)
    bmm -> batched_matmul    3 clusters: 3 proved (structural)
    mean keepdim=false       3 clusters: 1 proved (structural), 1 proved (structural) [sampled 32], 1 vacuous
    permute alone            3 clusters: 3 proved (structural) |}]

(* ---- bmm, now a direct counterpart -----------------------------------------

   [Bmm] used to legalize through a materialized, permuted convolution weight
   (a fresh intermediate consumed as a CONVOLUTION WEIGHT, so the destination
   read through a materialized stage where the source read its operand
   directly) -- that shape once tripped a verifier defect in how [normalise]'s
   f32-exactness collapse interacted with [value_tiers]' raw projected terms,
   found and fixed independently of this row. That legalization is now
   retired entirely: [Bmm] converts to [Batched_matmul] directly, reading both
   operands exactly as Native's own [Bmm.Compute] does, with no intermediate
   tensor of any kind. The verdicts below are correspondingly stronger --
   every cluster [proved (structural)], not [tested (coefficients agree)] --
   because there is no longer a materialization step for the two sides to
   disagree about. *)
let%expect_test "verify: bmm, per cluster" =
  (match Snapshot.create (Fixtures.bmm_batch 1 ()) with
  | Error _ -> Format.printf "snapshot failed@."
  | Ok (Snapshot.Pack src) -> (
      match Lower.convert src with
      | Error e -> Format.printf "%a@." Error.pp (Err.Error.kind e)
      | Ok (Lower.Pack r) -> (
          match
            Framework.Verify_from_native.run
              ~budget:(Map_verify.Effort.budget Map_verify.Effort.Thorough)
              r.Lower.map ~src ~dst:r.Lower.dst
          with
          | Error e ->
              Format.printf "%a@." Map_verify.pp_error (Err.Error.kind e)
          | Ok report ->
              Format.printf "%a@." Map_verify.Report.pp_verdicts report)));
  [%expect
    {|
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive]
    {t2} -> {t2} identical: proved (structural) [exhaustive] |}]

(* Renders EVERY graph output, not [List.hd]. For a single-output graph that is
   the same string; for a multi-output one it is the difference between an
   end-to-end check and a check of the first result. The comparison is on the
   rendered text, so an arity difference between the two dialects shows up as a
   disagreement rather than being silently ignored. *)
let render_outputs outputs env =
  String.concat " | "
    (List.map
       (fun id -> Format.asprintf "%a" Tensor.pp (Tensor_id.Map.find id env))
       outputs)

(* A numeric cross-check independent of the structural verifier above: Native
   and Native4D on the same inputs, compared as rendered tensors. *)
let%expect_test "verify: bmm, numerically" =
  let g = Fixtures.bmm_batch 1 () in
  let seq shape =
    let i = ref 0. in
    Tensor.materialize shape (fun _ ->
        i := !i +. 1.;
        !i)
  in
  let inputs =
    List.map
      (fun id ->
        ( id,
          seq (Tensor_id.Map.find id g.Graph_ir.Graph.tensors).Tensor_sig.shape
        ))
      g.Graph_ir.Graph.inputs
  in
  let native_out =
    match Eval_direct.run g ~inputs with
    | Error e ->
        Format.asprintf "native: %a" Eval_direct.pp_error (Err.Error.kind e)
    | Ok env -> render_outputs g.Graph_ir.Graph.outputs env
  in
  let four_out =
    match Snapshot.create g with
    | Error _ -> "snapshot failed"
    | Ok (Snapshot.Pack src) -> (
        match Lower.convert src with
        | Error e -> Format.asprintf "%a" Error.pp (Err.Error.kind e)
        | Ok (Lower.Pack r) -> (
            let dst = Lower.graph r in
            match Eval_direct4.run dst ~inputs with
            | Error e ->
                Format.asprintf "native4d: %a" Eval_direct4.pp_error
                  (Err.Error.kind e)
            | Ok env -> render_outputs dst.Graph_common.Graph.outputs env))
  in
  Format.printf "native:   %s@." native_out;
  Format.printf "native4d: %s@." four_out;
  Format.printf "agree: %b@." (String.equal native_out four_out);
  [%expect
    {|
    native:   tensor f32 [W=2 C=4] {38, 44, 50, 56, 83, 98, 113, 128}
    native4d: tensor f32 [W=2 C=4] {38, 44, 50, 56, 83, 98, 113, 128}
    agree: true |}]

(* The same end-to-end shape over a MULTI-OUTPUT graph, which the bmm case
   cannot cover. Every slice is compared, in order: the whole point of the
   lowering is that the ordered output list survives, and evaluating only the
   first would leave a dropped or reordered slice invisible here exactly as it
   is invisible to the structural checks. *)
(* Native and Native4D over the same inputs, rendered identically. Extracted so
   a second op's numeric proof does not become a second copy of the plumbing. *)
let seq shape =
  let i = ref 0. in
  Tensor.materialize shape (fun _ ->
      i := !i +. 1.;
      !i)

let native_vs_four g =
  let inputs =
    List.map
      (fun id ->
        ( id,
          seq (Tensor_id.Map.find id g.Graph_ir.Graph.tensors).Tensor_sig.shape
        ))
      g.Graph_ir.Graph.inputs
  in
  let native_out =
    match Eval_direct.run g ~inputs with
    | Error e ->
        Format.asprintf "native: %a" Eval_direct.pp_error (Err.Error.kind e)
    | Ok env -> render_outputs g.Graph_ir.Graph.outputs env
  in
  let four_out =
    match Snapshot.create g with
    | Error _ -> "snapshot failed"
    | Ok (Snapshot.Pack src) -> (
        match Lower.convert src with
        | Error e -> Format.asprintf "%a" Error.pp (Err.Error.kind e)
        | Ok (Lower.Pack r) -> (
            let dst = Lower.graph r in
            match Eval_direct4.run dst ~inputs with
            | Error e ->
                Format.asprintf "native4d: %a" Eval_direct4.pp_error
                  (Err.Error.kind e)
            | Ok env -> render_outputs dst.Graph_common.Graph.outputs env))
  in
  Format.printf "native:   %s@." native_out;
  Format.printf "native4d: %s@." four_out;
  Format.printf "agree: %b@." (String.equal native_out four_out)

(* These three operations share their Region program after Native4D's checked
   axis mapping; unlike the earlier generic numeric rows, Native4D Direct must
   therefore materialize that program rather than call [Eval_op4.pixel].  The
   structure comparison lives in [region_mapping_test.ml].  This is the
   end-to-end companion: lower a Native graph, run both Direct drivers on the
   same concrete inputs, and compare every rendered result. *)
let%expect_test "verify: authored Regions map and agree numerically" =
  List.iter
    (fun (name, graph) ->
      Format.printf "%s@." name;
      native_vs_four graph)
    [
      ("rms C", Fixtures.rms_norm_over [ Axis.C ] ());
      ("layer C", Fixtures.layer_norm_tiny ());
      ("softmax W", Fixtures.softmax_over Axis.W ());
    ];
  [%expect
    {|
    rms C
    native:   tensor f32 [H=4 W=4 C=3] {0.46291, 0.925819, 1.38873, 0.789542, 0.986927, 1.18431, 0.870478, 0.994832, ...}
    native4d: tensor f32 [H=4 W=4 C=3] {0.46291, 0.925819, 1.38873, 0.789542, 0.986927, 1.18431, 0.870478, 0.994832, ...}
    agree: true
    layer C
    native:   tensor f32 [W=2 C=3] {-0.224736, 2, 6.67421, -0.224736, 2, 6.67421}
    native4d: tensor f32 [W=2 C=3] {-0.224736, 2, 6.67421, -0.224736, 2, 6.67421}
    agree: true
    softmax W
    native:   tensor f32 [H=4 W=4 C=3] {0.000117266, 0.000117266, 0.000117266, 0.00235536, 0.00235536, 0.00235536, 0.0473086, 0.0473086, ...}
    native4d: tensor f32 [H=4 W=4 C=3] {0.000117266, 0.000117266, 0.000117266, 0.00235536, 0.00235536, 0.00235536, 0.0473086, 0.0473086, ...}
    agree: true |}]

let%expect_test "verify: unbind, numerically, over every slice" =
  native_vs_four (Fixtures.unbind_c_batch1 ());
  [%expect
    {|
    native:   tensor f32 [W=2 C=2] {1, 4, 7, 10} | tensor f32 [W=2 C=2] {2, 5, 8, 11} | tensor f32 [W=2 C=2] {3, 6, 9, 12}
    native4d: tensor f32 [W=2 C=2] {1, 4, 7, 10} | tensor f32 [W=2 C=2] {2, 5, 8, 11} | tensor f32 [W=2 C=2] {3, 6, 9, 12}
    agree: true |}]

(* [Unbind]'s single-output sibling, numerically: the one slice a select at a
   given index reads must agree with the identically-indexed slice
   [unbind_c_batch1] compares above -- both are the same window through the
   same repacking. *)
let%expect_test "verify: select, numerically, through the lowering" =
  native_vs_four (Fixtures.select_c_batch1 ());
  [%expect
    {|
    native:   tensor f32 [W=2 C=2] {2, 5, 8, 11}
    native4d: tensor f32 [W=2 C=2] {2, 5, 8, 11}
    agree: true |}]

let%expect_test "verify: split_with_sizes, numerically, over every window" =
  native_vs_four (Fixtures.split_with_sizes_w_batch2 ());
  [%expect
    {|
    native:   tensor f32 [N=2 T=1 D=1 H=2 W=1 C=3] {1, 2, 3, 13, 14, 15, 25, 26, ...} | tensor f32 [N=2 T=1 D=1 H=2 W=3 C=3] {4, 5, 6, 7, 8, 9, 10, 11, ...}
    native4d: tensor f32 [N=2 T=1 D=1 H=2 W=1 C=3] {1, 2, 3, 13, 14, 15, 25, 26, ...} | tensor f32 [N=2 T=1 D=1 H=2 W=3 C=3] {4, 5, 6, 7, 8, 9, 10, 11, ...}
    agree: true |}]

(* The selection op, end to end. With values 1..10 laid out row-major over
   [H=2 W=5], slicing W by [1,5) step 2 keeps columns 1 and 3 of each row: a
   wrong start, a wrong step or a wrong axis each print different numbers rather
   than only a different shape. *)
let%expect_test "verify: slice, numerically, through the lowering" =
  native_vs_four (Fixtures.slice_w ());
  [%expect
    {|
    native:   tensor f32 [H=2 W=2 C=1] {2, 4, 7, 9}
    native4d: tensor f32 [H=2 W=2 C=1] {2, 4, 7, 9}
    agree: true |}]

(* The joining op, end to end, over operands of DIFFERENT extents along the
   joined axis -- a wrong per-operand offset prints different numbers, not
   only a different shape. *)
let%expect_test "verify: concat, numerically, through the lowering" =
  native_vs_four (Fixtures.concat_w ());
  [%expect
    {|
    native:   tensor f32 [H=2 W=3 C=1] {1, 1, 2, 2, 3, 4}
    native4d: tensor f32 [H=2 W=3 C=1] {1, 1, 2, 2, 3, 4}
    agree: true |}]

(* The boundary-synthesis op, end to end, on the one fixture whose whole output
   prints — [Tensor.pp] truncates after eight elements, and on a fixture large
   enough to need truncating every visible element is fill. Both the synthesized
   row and the copied ones are in the golden, so a lowering that carried the
   axis but dropped the fill value is a difference in the numbers and not only
   in a shape. *)
(* The structural proof above says the two sides compute the same function; this
   says what that function IS, which no Native-vs-Native4D comparison can --
   both run the same [Compute] functor, so agreement here is about the LOWERING
   (the axis conversion, the operand mapping) and not about the formula. The
   hand-computed values for the formula itself live in
   test/native/compute_test.ml. *)
let%expect_test "verify: layer_norm, numerically, through the lowering" =
  native_vs_four (Fixtures.layer_norm_tiny ());
  [%expect
    {|
    native:   tensor f32 [W=2 C=3] {-0.224736, 2, 6.67421, -0.224736, 2, 6.67421}
    native4d: tensor f32 [W=2 C=3] {-0.224736, 2, 6.67421, -0.224736, 2, 6.67421}
    agree: true |}]

let%expect_test "verify: group_norm, numerically, through the lowering" =
  native_vs_four (Fixtures.group_norm_tiny ());
  [%expect
    {|
    native:   tensor f32 [W=2 C=4] {-0.212677, 0.544788, -0.63803, 1.08958, 1.72761, 4.42535, 5.18282, 8.85071}
    native4d: tensor f32 [W=2 C=4] {-0.212677, 0.544788, -0.63803, 1.08958, 1.72761, 4.42535, 5.18282, 8.85071}
    agree: true |}]

let%expect_test "verify: expand, numerically, through the lowering" =
  native_vs_four (Fixtures.expand_tiny ());
  [%expect
    {|
    native:   tensor f32 [W=3 C=2] {1, 2, 1, 2, 1, 2}
    native4d: tensor f32 [W=3 C=2] {1, 2, 1, 2, 1, 2}
    agree: true |}]

let%expect_test "verify: pad, numerically, through the lowering" =
  native_vs_four (Fixtures.pad_tiny ());
  [%expect
    {|
    native:   tensor f32 [H=3 W=2 C=1] {9, 9, 1, 2, 3, 4}
    native4d: tensor f32 [H=3 W=2 C=1] {9, 9, 1, 2, 3, 4}
    agree: true |}]
