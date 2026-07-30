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

let build name m =
  match Graph_builder.build ~name ~outputs:(fun o -> [ o ]) m with
  | Ok g -> g
  | Error e ->
      invalid_arg
        (Format.asprintf "fixture %s: %a" name Graph_builder.pp_error
           e.Core.Error.kind)

(* Convert, then check the map the conversion produced. Everything happens
   inside the scope that unpacks the destination version — the report is
   version-free and comes back out, the snapshot cannot. *)
let verified ?constants ?(effort = Map_verify.Effort.Thorough) g =
  match Snapshot.create g with
  | Error e ->
      Format.asprintf "snapshot: %a" Graph_view.pp_error e.Core.Error.kind
  | Ok (Snapshot.Pack src) -> (
      match Lower.convert ?constants src with
      | Error e -> Format.asprintf "%a" Error.pp e.Core.Error.kind
      | Ok (Lower.Pack r) -> (
          let budget = Map_verify.Effort.budget effort in
          match
            Framework.Verify_from_native.run ~budget
              ~probe:(Map_verify.Effort.probe effort)
              ?src_constants:(Option.map Fun.id constants)
              ~dst_constants:r.Lower.constants r.Lower.map ~src ~dst:r.Lower.dst
          with
          | Error e ->
              Format.asprintf "verify: %a" Map_verify.pp_error e.Core.Error.kind
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
  check "linear -> 1x1 conv" (Fixtures.linear_layer ());
  check "bmm -> permute + conv" (Fixtures.bmm_batch 1 ());
  check "mean keepdim=false" (Fixtures.mean_over_hw ~keepdim:false ~n:1 ());
  (* Isolating what the bmm refutation below turns on: a permutation alone, and
     a convolution whose weight is a graph input rather than a permuted one. *)
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
    linear -> 1x1 conv       3 clusters: 1 proved (structural), 2 unproved (unbound constant)
    bmm -> permute + conv    4 clusters: 2 proved (structural), 1 refuted (counterexample), 1 vacuous
    mean keepdim=false       3 clusters: 1 proved (structural), 1 proved (structural) [sampled 32], 1 vacuous
    permute alone            3 clusters: 3 proved (structural) |}]

(* ---- AN OPEN FINDING: the bmm legalization refutes ------------------------

   The cluster {t2} -> {t2} for the single-batch Bmm comes back REFUTED with an
   exhaustive counterexample, and the numeric cross-check below shows the two
   graphs computing the same values. Both are recorded because the conflict is
   the finding: one of them is wrong and it is not yet established which.

   What has been ruled out, each by a test above rather than by argument:

   - not the permutation. "permute alone" — a Native Permute lowered to
     Permute4 — proves structurally.
   - not fresh intermediates in general. "mean keepdim=false" introduces one
     (the MeanKeepDims output) and proves.
   - not a creation cluster becoming a frontier variable. [Boundary_index]
     skips vacuous clusters, so the created weight edge gets no variable and is
     expanded through.
   - not the arithmetic. The numeric test agrees on a non-degenerate input, and
     a transposed pairing would not (it gives 34 where both give 50).

   What is left is the one shape not covered by any passing case: a fresh
   intermediate consumed as a CONVOLUTION WEIGHT, where the source reads its
   operand directly and the destination reads it through a materialized stage.
   That is a difference in f32 materialization points, which is exactly what an
   [Identical] claim is sensitive to — so the honest reading is that either the
   claim should be weaker than Identical, or the Round collapse is not firing
   through a stage load. Not resolved here; not asserted either way. *)

(* The BMM legalization refuted above, in detail. *)
let%expect_test "verify: bmm, per cluster" =
  (match Snapshot.create (Fixtures.bmm_batch 1 ()) with
  | Error _ -> Format.printf "snapshot failed@."
  | Ok (Snapshot.Pack src) -> (
      match Lower.convert src with
      | Error e -> Format.printf "%a@." Error.pp e.Core.Error.kind
      | Ok (Lower.Pack r) -> (
          match
            Framework.Verify_from_native.run
              ~budget:(Map_verify.Effort.budget Map_verify.Effort.Thorough)
              r.Lower.map ~src ~dst:r.Lower.dst
          with
          | Error e ->
              Format.printf "%a@." Map_verify.pp_error e.Core.Error.kind
          | Ok report ->
              Format.printf "%a@." Map_verify.Report.pp_verdicts report)));
  [%expect
    {|
    {} -> {t3} identical: vacuous
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive]
    {t2} -> {t2} identical: refuted: value at (2): src.t2 vs dst.t2 under {v0(0)=-0x1.1374bc6a7ef9ep+0, v0(1)=-0x1.0624dd2f1a9fcp-5, v0(2)=-0x1.374bc6a7ef9dbp-2, v1(2)=0x1.4ac083126e979p+0, v1(1,2)=0x1.c5a1cac083127p+1, v1(2,2)=-0x1.3f7ced916872bp+1} [exhaustive] |}]

(* Numeric cross-check for the refuted cluster: run the Native graph and its
   conversion on the same inputs. A refutation the numbers agree with would
   mean the verifier is wrong; a refutation they confirm means the lowering is. *)
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
        Format.asprintf "native: %a" Eval_direct.pp_error e.Core.Error.kind
    | Ok env ->
        let t = Tensor_id.Map.find (List.hd g.Graph_ir.Graph.outputs) env in
        Format.asprintf "%a" Tensor.pp t
  in
  let four_out =
    match Snapshot.create g with
    | Error _ -> "snapshot failed"
    | Ok (Snapshot.Pack src) -> (
        match Lower.convert src with
        | Error e -> Format.asprintf "%a" Error.pp e.Core.Error.kind
        | Ok (Lower.Pack r) -> (
            let dst = Lower.graph r in
            match Eval_direct4.run dst ~inputs with
            | Error e ->
                Format.asprintf "native4d: %a" Eval_direct4.pp_error
                  e.Core.Error.kind
            | Ok env ->
                let t =
                  Tensor_id.Map.find
                    (List.hd dst.Graph_common.Graph.outputs)
                    env
                in
                Format.asprintf "%a" Tensor.pp t))
  in
  Format.printf "native:   %s@." native_out;
  Format.printf "native4d: %s@." four_out;
  Format.printf "agree: %b@." (String.equal native_out four_out);
  [%expect
    {|
    native:   tensor f32 [W=2 C=4] {38, 44, 50, 56, 83, 98, 113, 128}
    native4d: tensor f32 [W=2 C=4] {38, 44, 50, 56, 83, 98, 113, 128}
    agree: true |}]
