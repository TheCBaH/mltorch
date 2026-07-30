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
    bmm -> permute + conv    4 clusters: 2 proved (structural), 1 tested (coefficients agree), 1 vacuous
    mean keepdim=false       3 clusters: 1 proved (structural), 1 proved (structural) [sampled 32], 1 vacuous
    permute alone            3 clusters: 3 proved (structural) |}]

(* ---- why the bmm legalization needed a verifier fix -----------------------

   This cluster refuted, with an exhaustive counterexample, while the numeric
   cross-check below showed both graphs computing the same values. The conflict
   was the finding, and it turned out to be the verifier's.

   Ruled out first, each by a test above rather than by argument: the
   permutation ("permute alone" proves), fresh intermediates in general ("mean
   keepdim=false" has one and proves), a creation cluster becoming a frontier
   variable ([Boundary_index] skips vacuous clusters), and the arithmetic (a
   transposed pairing would give 34 where both give 50).

   What remained was the one uncovered shape: a fresh intermediate consumed as a
   CONVOLUTION WEIGHT, so the destination reads through a materialized stage
   where the source reads its operand directly. The terms below show it — an
   extra [f32(...)] around each weight load. [normalise] collapses those, that
   collapse being legal exactly when the cell's stored format round-trips f32 —
   but [value_tiers] was handing the probe the RAW projected terms, where the
   collapse had not been applied. The probe then drew a double f32 cannot hold
   and separated [f32(v)] from [v].

   That is a counterexample no execution can realise, which is the same defect
   the settled-frontier rule already guards against, applied to storage format
   instead of to producers. Fixed by probing the normalised terms; see the
   commit that changed [value_tiers].

   The residual verdict is [tested (coefficients agree)] rather than [proved],
   because structural equality still fails on association: the source sums
   [((0 + a) + b) + c] and the convolution [((((0+(0+(0+a))) + …) + 0)]. Those
   are the same float sequence — [0 + x] is exact — but [normalise] does not
   simplify additive identities, so the prover cannot see it. *)

(* The per-cluster verdicts, now that the probe sees normalised terms. *)
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
    {t2} -> {t2} identical: tested: agrees (1e-05) [exhaustive] |}]

(* The numeric cross-check that started the investigation: a refutation the
   numbers agree with means the verifier is wrong; one they confirm means the
   lowering is. They agreed. *)
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

(* The two ground terms side by side. This is what located the defect: the
   destination's extra [f32(...)] around each weight load, present in the raw
   term and gone from the normalised one — which is why the probe had to be
   given the normalised term rather than the raw one. *)
let%expect_test "verify: bmm, the two terms" =
  let g = Fixtures.bmm_batch 1 () in
  (match Snapshot.create g with
  | Error _ -> Format.printf "snapshot failed@."
  | Ok (Snapshot.Pack src) -> (
      match Lower.convert src with
      | Error e -> Format.printf "%a@." Error.pp e.Core.Error.kind
      | Ok (Lower.Pack r) ->
          let dst = Lower.graph r in
          let src_env =
            Ground_eval.Env.of_program (Eval_symbolic.run g) ~side:`Src
          and dst_env =
            Ground_eval.Env.of_program (Eval_symbolic4.run dst) ~side:`Dst
          in
          let out = List.hd g.Graph_ir.Graph.outputs in
          (* The coordinate the refutation named: C=2, everything else 0. *)
          let coord = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:2 in
          let show label env =
            match Ground_eval.at env out coord with
            | Error e ->
                Format.printf "%s: %a@." label Ground_eval.pp_error
                  e.Core.Error.kind
            | Ok t ->
                (* Expand with NO boundary, so both sides run all the way down to
                   their graph inputs and any difference is structural rather
                   than a frontier artefact. *)
                let t =
                  Ground_eval.expand
                    ~boundary:(fun _ -> None)
                    ~budget:100_000 env t
                in
                Format.printf "@[<v 2>%s:@,%a@]@." label Ground_expr.pp t
          in
          show "src" src_env;
          show "dst" dst_env;
          (* NORMALISED, which is what the verifier actually compares: the
             Round-over-Cell collapse fires here if it fires at all. *)
          let term env =
            match Ground_eval.at env out coord with
            | Error _ -> Ground_expr.Const 0.
            | Ok t ->
                (Ground_expr.normalise
                   ~stored_f32:(Ground_eval.Env.stored_f32 env)
                   (Ground_eval.expand
                      ~boundary:(fun _ -> None)
                      ~budget:100_000 env t))
                  .Ground_expr.expr
          in
          let ls = term src_env and rs = term dst_env in
          Format.printf "@[<v 2>dst normalised:@,%a@]@." Ground_expr.pp rs;
          (* PROJECTED, as the verifier compares them: t0 and t1 are corresponding
             inputs, so each becomes one shared variable and the two sides stop
             naming their cells by side. Comparing unprojected terms says nothing
             — src.t0 and dst.t0 are different generators. *)
          let boundary (o : Ground_expr.Origin.t) =
            match o with
            | Ground_expr.Origin.Src id | Ground_expr.Origin.Dst id ->
                Some (Cluster_var.of_int (Tensor_id.to_int id))
            | Ground_expr.Origin.Boundary _ -> None
          in
          let lp = Ground_expr.project ~boundary ls
          and rp = Ground_expr.project ~boundary rs in
          Format.printf "structurally equal: %b@." (Ground_expr.equal lp rp);
          Format.printf "coefficients agree: %b@."
            (Coeff_form.agree ~tolerance:1e-5 lp rp)));
  [%expect
    {|
    src:
      f32((((0x0p+0 + (src.t0(0) * src.t1(2))) + (src.t0(1) * src.t1(1,2))) + (src.t0
      (2) * src.t1(2,2))))
    dst:
      f32(((((0x0p+0 + (0x0p+0 + (0x0p+0 + (dst.t0(0) * f32(dst.t1(2)))))) + (0x0p+0 + (0x0p+0 + (dst.t0
      (1) * f32(dst.t1(1,2)))))) + (0x0p+0 + (0x0p+0 + (dst.t0(2) * f32(dst.t1
      (2,2)))))) + 0x0p+0))
    dst normalised:
      f32(((((0x0p+0 + (0x0p+0 + (0x0p+0 + (dst.t0(0) * dst.t1(2))))) + (0x0p+0 + (0x0p+0 + (dst.t0
      (1) * dst.t1(1,2))))) + (0x0p+0 + (0x0p+0 + (dst.t0(2) * dst.t1(2,2))))) + 0x0p+0))
    structurally equal: false
    coefficients agree: true |}]
