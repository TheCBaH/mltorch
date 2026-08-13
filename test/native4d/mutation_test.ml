(* One deliberately WRONG conversion per legalization family, and what the
   verifier says about it. Stage 6's mutation matrix.

   A mutation test is only evidence if it has been seen to fail, and the way to
   make that permanent is to keep the wrong conversion in the tree rather than
   to edit the lowerer and revert. So each case builds a Native source and a
   Native4D destination BY HAND, with matching ids, and gives them the map a
   real conversion of that shape would have produced. The verdict is then the
   answer to "would the verifier have caught this".

   For most cases that map is the EMPTY one: no explicit cluster, so every id
   present in both graphs is implicitly [Identical]. Two cases need more, and
   both need it because the real lowerer would produce more:

   - batch-norm precomputation is the dialect's only [Equivalent] legalization,
     so it passes [~claims] naming the two parameter edges [Unverifiable] and
     the output [Equivalent]. An empty map there would assert [Identical] of a
     re-association the lowering does not promise.
   - the bmm weight permutation builds its map outright — [Correspondence.create]
     for the fresh permuted operand and a non-empty [~nodes] — because that
     legalization emits two destination nodes from one source node. The empty
     map cannot express either.

   Mutations are SHAPE-PRESERVING wherever the point is to reach the verifier,
   and that constraint is not cosmetic: a shape change never reaches it at all,
   because [Graph_view] rejects the graph or [Graph_map.create] rejects the
   cluster first. The reduction case is the deliberate exception — it mutates a
   shape precisely to pin WHICH layer rejects it, and its expected output is a
   [Graph_map] error rather than a verdict. Several other mistakes the plan
   listed as mutations to write turn out to be unreachable the same way; the
   note at the bottom records which. *)

open Native4d

let nat name m =
  Graph_builder.build ~name ~outputs:(fun o -> [ o ]) m
  |> Err.or_raise ~pp_error:(fun ppf e ->
      Fmt.pf ppf "%s: %a" name Graph_builder.pp_error e)

let four m =
  Builder.build ~outputs:(fun o -> [ o ]) m
  |> Err.or_raise ~pp_error:Builder.pp_error

(* Verify one hand-built pair against a map with no node clusters and only the
   [claims] the caller names explicitly — so every other id present in both
   graphs is implicitly [Identical]. Ids match because both builders allocate
   from zero in the same order, which the first case below checks by printing
   them. *)
let mutated ?(constants = Tensor_id.Map.empty) ?dst_constants ?(claims = [])
    name src_g dst_g =
  let dst_constants = Option.value dst_constants ~default:constants in
  ignore dst_constants;
  match (Snapshot.create src_g, Framework.Snapshot4.create dst_g) with
  | Error e, _ ->
      Format.printf "%-26s src: %a@." name Graph_view.pp_error
        (Err.Error.kind e)
  | _, Error e ->
      Format.printf "%-26s dst: %a@." name Framework.View4.pp_error
        (Err.Error.kind e)
  | Ok (Snapshot.Pack src), Ok (Framework.Snapshot4.Pack dst) -> (
      match
        (* Raw ids in, clusters built HERE: the snapshots are existential, so a
           caller-supplied closure over them would not typecheck. *)
        let values =
          List.filter_map
            (fun (id, rel) ->
              match (Snapshot.edge src id, Framework.Snapshot4.edge dst id) with
              | Some s, Some d -> Some (Correspondence.pair s d rel)
              | _ -> None)
            claims
        in
        Framework.Map_from_native.create ~src ~dst ~values ~nodes:[]
          ~provenance:Provenance.empty
      with
      | Error e ->
          Format.printf "%-26s map: %a@." name Graph_map.pp_error
            (Err.Error.kind e)
      | Ok map -> (
          match
            Framework.Verify_from_native.run
              ~budget:(Map_verify.Effort.budget Map_verify.Effort.Thorough)
              ~probe:(Map_verify.Effort.probe Map_verify.Effort.Thorough)
              ~src_constants:constants ~dst_constants map ~src ~dst
          with
          | Error e ->
              Format.printf "%-26s verify: %a@." name Map_verify.pp_error
                (Err.Error.kind e)
          | Ok report ->
              Format.printf "%-26s %s@." name (Map_verify.Report.summary report)
          ))

let sq = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:2
let sq4 = Shape4.of_ints ~n:1 ~h:2 ~w:2 ~c:2

(* ---- the correct conversions, as the baseline ----------------------------- *)

(* Without these the refutations below prove nothing: a suite where everything
   refutes is a suite that refutes everything. *)
let%expect_test "mutation: the honest conversions prove" =
  mutated "sub -> sub"
    (nat "sub"
       Graph_builder.(
         let* a = input ~shape:sq () in
         let* b = input ~shape:sq () in
         sub a b))
    (four
       Builder.(
         let* a = input ~shape:sq4 () in
         let* b = input ~shape:sq4 () in
         sub a b));
  mutated "permute -> permute4"
    (nat "permute"
       Graph_builder.(
         let* x = input ~shape:sq () in
         permute
           (Permute.Permute.of_fn (function
             | Axis.H -> Axis.W
             | Axis.W -> Axis.H
             | a -> a))
           x))
    (four
       Builder.(
         let* x = input ~shape:sq4 () in
         permute4
           (Ops4.Permute4.of_fn (function
             | Axis4.H -> Axis4.W
             | Axis4.W -> Axis4.H
             | a -> a))
           x));
  [%expect
    {|
    sub -> sub                 3 clusters: 3 proved (structural)
    permute -> permute4        2 clusters: 2 proved (structural) |}]

(* ---- one mutation per family ---------------------------------------------- *)

(* DIRECT DELEGATION. The failure a copy-pasted op registry actually produces:
   an arm wired to the wrong Native [Compute]. *)
let%expect_test "mutation: a direct arm dispatching the wrong Compute" =
  mutated "sub -> add"
    (nat "sub"
       Graph_builder.(
         let* a = input ~shape:sq () in
         let* b = input ~shape:sq () in
         sub a b))
    (four
       Builder.(
         let* a = input ~shape:sq4 () in
         let* b = input ~shape:sq4 () in
         add a b));
  [%expect
    {| sub -> add                 3 clusters: 2 proved (structural), 1 refuted (counterexample) |}]

(* THE ID POLICY. The crossed-operand case verify.md records: same operator,
   same raw ids, operands exchanged. This is what justifies preserving source
   ids at all — if the verifier could not separate these, an id-preserving
   conversion would be unfalsifiable. *)
let%expect_test "mutation: operands crossed under preserved ids" =
  mutated "sub a b -> sub b a"
    (nat "sub"
       Graph_builder.(
         let* a = input ~shape:sq () in
         let* b = input ~shape:sq () in
         sub a b))
    (four
       Builder.(
         let* a = input ~shape:sq4 () in
         let* b = input ~shape:sq4 () in
         sub b a));
  [%expect
    {| sub a b -> sub b a         3 clusters: 2 proved (structural), 1 refuted (counterexample) |}]

(* SHAPE-ONLY. A permutation lowered to the identity. Square in H and W so the
   shape survives and only the values move — a non-square tensor would be
   rejected before the verifier ever saw it. *)
let%expect_test "mutation: a permutation lowered to the identity" =
  mutated "permute -> identity"
    (nat "permute"
       Graph_builder.(
         let* x = input ~shape:sq () in
         permute
           (Permute.Permute.of_fn (function
             | Axis.H -> Axis.W
             | Axis.W -> Axis.H
             | a -> a))
           x))
    (four
       Builder.(
         let* x = input ~shape:sq4 () in
         permute4 Ops4.Permute4.identity x));
  [%expect
    {| permute -> identity        2 clusters: 1 proved (structural), 1 refuted (counterexample) |}]

(* REDUCTION — and this one does NOT reach the verifier, which is the point of
   including it. A reduction's output shape is determined by which axes it
   removes, so translating the axes wrongly changes the shape, and the map
   rejects the cluster before any term is built. Caught earlier and harder than
   a refutation.

   The obvious way to make it shape-preserving is to reduce over an axis of
   extent 1, but that makes the two reductions genuinely EQUAL rather than
   merely same-shaped — an earlier draft of this test did exactly that and
   reported "coefficients agree", which was the correct answer to a question
   nobody meant to ask. *)
let%expect_test "mutation: a reduction over the wrong axes is caught by shape" =
  mutated "mean [H;W] -> mean [H]"
    (nat "mean"
       Graph_builder.(
         let* x = input ~shape:sq () in
         mean { Reduce.Mean.dims = Axis.[ H; W ]; keepdim = true } x))
    (four
       Builder.(
         let* x = input ~shape:sq4 () in
         mean_keepdims [ Axis4.H ] x));
  [%expect
    {| mean [H;W] -> mean [H]     map: value map: t1 and t1 correspond but differ in shape |}]

(* SCALAR PARAMETERS. A parameter carried across wrong is invisible to every
   structural check — the graphs are the same shape, the same ops, the same
   wiring. Only the values differ. *)
let%expect_test "mutation: a scalar parameter carried across wrong" =
  mutated "add_scalar 1 -> 2"
    (nat "add_scalar"
       Graph_builder.(
         let* x = input ~shape:sq () in
         add_scalar 1.0 x))
    (four
       Builder.(
         let* x = input ~shape:sq4 () in
         add_scalar 2.0 x));
  [%expect
    {| add_scalar 1 -> 2          2 clusters: 1 proved (structural), 1 refuted (counterexample) |}]

(* CONVOLUTION PARAMETERS. Asymmetric padding applied to the wrong side. The
   output shape depends on pad_before + pad_after, so swapping them preserves it
   and moves only the window — which is why this is the conv mutation that
   reaches the verifier at all. *)
let%expect_test "mutation: convolution padding applied to the wrong side" =
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let x_shape4 = Shape4.of_ints ~n:1 ~h:3 ~w:3 ~c:1 in
  let w_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let w_shape4 = Shape4.of_ints ~n:1 ~h:2 ~w:2 ~c:1 in
  let window ~before ~after : Conv.Conv2d.axis_window =
    {
      kernel = Dim.extent 2;
      stride = Op_config.Pos.of_int 1;
      pad_before = Op_config.Nonneg.of_int before;
      pad_after = Op_config.Nonneg.of_int after;
      dilation = Op_config.Pos.of_int 1;
    }
  in
  mutated "conv pad 1/0 -> 0/1"
    (nat "conv"
       Graph_builder.(
         let* x = input ~shape:x_shape () in
         let* w = input ~shape:w_shape () in
         conv2d
           {
             Conv.Conv2d.h = window ~before:1 ~after:0;
             w = window ~before:1 ~after:0;
             in_channels = Dim.extent 1;
             groups = Op_config.Pos.of_int 1;
           }
           ~x ~weight:w ()))
    (four
       Builder.(
         let* x = input ~shape:x_shape4 () in
         let* w = input ~shape:w_shape4 () in
         conv2d
           {
             Ops4.Conv_params.h = window ~before:0 ~after:1;
             w = window ~before:0 ~after:1;
             in_channels = Dim.extent 1;
           }
           ~x ~weight:w ()));
  [%expect
    {| conv pad 1/0 -> 0/1        3 clusters: 2 proved (structural), 1 refuted (counterexample) |}]

(* ---- the Equivalent family ------------------------------------------------

   Batch-norm precomputation is the only legalization that is [Equivalent]
   rather than [Identical], and the plan's mutation for it was a scale computed
   without eps, asserted as an Agrees -> Disagrees transition. That assertion
   is only meaningful if the CORRECT conversion reaches [Agrees]: a probe
   formally refutes only [Identical], so for an [Equivalent] claim a
   disagreement is evidence rather than a counterexample, and "Disagrees or
   Unproved" would be a mutation test that cannot go red.

   So the correct conversion's tier is recorded first. If it is not [Agrees],
   the transition is not assertable on a fixture this size and the plan says to
   say so rather than keep an assertion that cannot fail. *)
let%expect_test "mutation: what tier the honest batch-norm conversion reaches" =
  let chan c = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c in
  let g =
    nat "bn"
      Graph_builder.(
        let* x = input ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2) () in
        let* running_mean = constant ~shape:(chan 2) () in
        let* running_var = constant ~shape:(chan 2) () in
        batch_norm
          { Norm.BatchNorm.channel = Axis.C; eps = 1e-5 }
          ~x ~running_mean ~running_var ())
  in
  let fill v = Tensor.materialize (chan 2) (fun _ -> v) in
  let constants =
    match g.Graph_ir.Graph.inputs with
    | [ _; m; v ] ->
        Tensor_id.Map.of_seq (List.to_seq [ (m, fill 1.); (v, fill 4.) ])
    | _ -> invalid_arg "unexpected fixture shape"
  in
  (match Snapshot.create g with
  | Error _ -> Format.printf "snapshot failed@."
  | Ok (Snapshot.Pack src) -> (
      match Lower.convert ~constants src with
      | Error e -> Format.printf "%a@." Error.pp (Err.Error.kind e)
      | Ok (Lower.Pack r) -> (
          match
            Framework.Verify_from_native.run
              ~budget:(Map_verify.Effort.budget Map_verify.Effort.Thorough)
              ~probe:(Map_verify.Effort.probe Map_verify.Effort.Thorough)
              ~src_constants:constants ~dst_constants:r.Lower.constants
              r.Lower.map ~src ~dst:r.Lower.dst
          with
          | Error e ->
              Format.printf "%a@." Map_verify.pp_error (Err.Error.kind e)
          | Ok report ->
              Format.printf "%a@." Map_verify.Report.pp_verdicts report)));
  [%expect
    {|
    {t3} -> {t3} equivalent: tested: agrees (1e-05) [exhaustive]
    {} -> {t4} identical: vacuous
    {} -> {t5} identical: vacuous
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural, for these constants) [exhaustive]
    {t2} -> {t2} identical: proved (structural, for these constants) [exhaustive] |}]

(* THE EQUIVALENT MUTATION. A scale precomputed WITHOUT eps, against the same
   depthwise convolution the lowerer builds. The claim is [Equivalent], so a
   probe cannot refute it — that is reserved for [Identical] — and the evidence
   available is the coefficient tier. What the test pins is therefore the
   TRANSITION: agrees for the honest scale, disagrees for the mutated one.

   eps is 0.5, not the 1e-5 a real graph would carry, and deliberately: with a
   realistic eps the two scales differ by about 5e-6 relative, which is INSIDE
   the coefficient tolerance, so the mutation would report agrees and the test
   would assert nothing. A mutation has to be larger than the bar it is measured
   against. *)
let%expect_test "mutation: batch-norm scale computed without eps" =
  let chan c = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c in
  let chan4 c = Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c in
  let eps = 0.5 and mean = 1.0 and var = 4.0 in
  let src_g =
    nat "bn"
      Graph_builder.(
        let* x = input ~shape:(chan 1) () in
        let* running_mean = constant ~shape:(chan 1) () in
        let* running_var = constant ~shape:(chan 1) () in
        batch_norm
          { Norm.BatchNorm.channel = Axis.C; eps }
          ~x ~running_mean ~running_var ())
  in
  (* Depthwise 1x1 over the precomputed scale and offset: the shape the
     legalization produces, with ids landing on the source's. *)
  let dst_g =
    four
      Builder.(
        let* x = input ~shape:(chan4 1) () in
        let* scale = constant ~shape:(Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:1) () in
        let* offset = constant ~shape:(chan4 1) () in
        depthwise_conv2d
          {
            Ops4.Conv_params.h = Fixtures4.axis_window ~kernel:1;
            w = Fixtures4.axis_window ~kernel:1;
            in_channels = Dim.extent 1;
          }
          ~x ~weight:scale ~bias:offset ())
  in
  let t_ n = Tensor_id.of_int n in
  let fill shape v = Tensor.materialize shape (fun _ -> v) in
  let src_constants =
    Tensor_id.Map.of_seq
      (List.to_seq [ (t_ 1, fill (chan 1) mean); (t_ 2, fill (chan 1) var) ])
  in
  let dst_constants ~with_eps =
    let s = 1. /. Float.sqrt (var +. if with_eps then eps else 0.) in
    Tensor_id.Map.of_seq
      (List.to_seq
         [ (t_ 1, fill (chan 1) s); (t_ 2, fill (chan 1) (-.(mean *. s))) ])
  in
  (* The output claimed [Equivalent], as the lowerer claims it. The two constant
     slots are claimed [Unverifiable] because they genuinely do not correspond:
     the source holds mean and variance where the destination holds the
     precomputed scale and offset. The real lowerer gives those fresh ids and a
     creation cluster; here they collide, and saying they are identical would be
     a second, unintended mutation drowning the one under test. *)
  let claims =
    [
      (t_ 1, Correspondence.Unverifiable);
      (t_ 2, Correspondence.Unverifiable);
      (t_ 3, Correspondence.Equivalent);
    ]
  in
  List.iter
    (fun (name, with_eps) ->
      mutated ~constants:src_constants ~dst_constants:(dst_constants ~with_eps)
        ~claims name src_g dst_g)
    [ ("scale with eps", true); ("scale without eps", false) ];
  [%expect
    {|
    scale with eps             4 clusters: 1 proved (structural), 1 tested (coefficients agree), 2 unproved (unsupported relation)
    scale without eps          4 clusters: 1 proved (structural), 1 tested (disagrees), 2 unproved (unsupported relation) |}]

(* ---- the bmm weight permutation ------------------------------------------

   Design §11 stage 5 names this one specifically, and it earns the attention:
   BMM is where the verifier's own Round-collapse defect surfaced, so a
   regression on it guards two things at once.

   It cannot go through [mutated]'s empty map. The legalization emits TWO
   destination nodes and one fresh edge, so the id spaces do not line up the way
   a one-for-one rewrite's do — the destination is built here with the ids the
   lowerer would assign (the permuted weight above the source watermark, the
   convolution keeping the source's output id) and the map states the creation
   and the node fusion explicitly.

   Shape-preserving because mat2 is SQUARE in contract and columns: the correct
   permutation is N<-C, H<-H, W<-N, C<-W and the mutated one N<-W, H<-H, W<-N,
   C<-C, both giving [N=2,H=1,W=1,C=2]. Only the values transpose. *)
let%expect_test "mutation: the bmm weight permutation transposed" =
  let t_ n = Tensor_id.of_int n in
  let n_ n = Graph_common.Node_id.of_int n in
  let src_g =
    nat "bmm"
      Graph_builder.(
        let* a = input ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:2) () in
        let* b = input ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:2) () in
        bmm a b)
  in
  let sig_ id shape =
    Tensor_sig.create ~id ~name:"" ~shape ~fmt:(Payload.Fmt Payload.F32) ()
  in
  let dst_g ~correct =
    let perm =
      Ops4.Permute4.of_fn (function
        | Axis4.N -> if correct then Axis4.C else Axis4.W
        | Axis4.W -> Axis4.N
        | Axis4.C -> if correct then Axis4.W else Axis4.C
        | Axis4.H -> Axis4.H)
    in
    {
      Graph_common.Graph.nodes =
        [
          {
            Graph_common.Node.id = n_ 0;
            op = Op.Permute4 { Ops4.Permute4.perm; x = t_ 1 };
            outputs = [ t_ 3 ];
          };
          {
            Graph_common.Node.id = n_ 1;
            op =
              Op.Conv2d
                {
                  Ops4.Conv_payload.params =
                    {
                      Ops4.Conv_params.h = Fixtures4.axis_window ~kernel:1;
                      w = Fixtures4.axis_window ~kernel:1;
                      in_channels = Dim.extent 2;
                    };
                  x = t_ 0;
                  weight = t_ 3;
                  bias = None;
                };
            outputs = [ t_ 2 ];
          };
        ];
      root =
        {
          Graph_common.Group.id = Graph_common.Group_id.of_int 0;
          label = None;
          items = [ Graph_common.Group.Node (n_ 0); Node (n_ 1) ];
        };
      tensors =
        Tensor_id.Map.of_seq
          (List.to_seq
             [
               (t_ 0, sig_ (t_ 0) (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:2));
               (t_ 1, sig_ (t_ 1) (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:2));
               (t_ 2, sig_ (t_ 2) (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:2));
               (t_ 3, sig_ (t_ 3) (Vec6.shape ~n:2 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2));
             ]);
      inputs = [ t_ 0; t_ 1 ];
      input_kinds = Tensor_id.Map.empty;
      outputs = [ t_ 2 ];
    }
  in
  let check name ~correct =
    match
      (Snapshot.create src_g, Framework.Snapshot4.create (dst_g ~correct))
    with
    | Error _, _ | _, Error _ -> Format.printf "%-22s snapshot failed@." name
    | Ok (Snapshot.Pack src), Ok (Framework.Snapshot4.Pack dst) -> (
        let values =
          Option.to_list
            (Option.map Correspondence.create
               (Framework.Snapshot4.edge dst (t_ 3)))
        in
        let nodes =
          match
            ( Snapshot.node src (n_ 0),
              Framework.Snapshot4.node dst (n_ 0),
              Framework.Snapshot4.node dst (n_ 1) )
          with
          | Some s, Some d0, Some d1 ->
              [
                {
                  Node_map.Cluster.src = Node_map.Set.singleton s;
                  dst = Node_map.Set.of_list [ d0; d1 ];
                  label = ();
                };
              ]
          | _ -> []
        in
        match
          Framework.Map_from_native.create ~src ~dst ~values ~nodes
            ~provenance:Provenance.empty
        with
        | Error e ->
            Format.printf "%-22s map: %a@." name Graph_map.pp_error
              (Err.Error.kind e)
        | Ok map -> (
            match
              Framework.Verify_from_native.run
                ~budget:(Map_verify.Effort.budget Map_verify.Effort.Thorough)
                ~probe:(Map_verify.Effort.probe Map_verify.Effort.Thorough)
                map ~src ~dst
            with
            | Error e ->
                Format.printf "%-22s verify: %a@." name Map_verify.pp_error
                  (Err.Error.kind e)
            | Ok report ->
                Format.printf "%-22s %s@." name
                  (Map_verify.Report.summary report)))
  in
  check "correct permutation" ~correct:true;
  check "transposed" ~correct:false;
  [%expect
    {|
    correct permutation    4 clusters: 2 proved (structural), 1 tested (coefficients agree), 1 vacuous
    transposed             4 clusters: 2 proved (structural), 1 refuted (counterexample), 1 vacuous |}]

(* ---- the multi-output mutation -------------------------------------------- *)

(* [Unbind] is the one op whose outputs can be mutated WITHOUT changing any
   shape, any id set, or the graph's own signature — every slice has the same
   shape, so swapping two of them leaves a destination graph that validates and
   a map that is structurally perfect.

   Nothing before the verifier can see it. [Graph_view] checks arity and
   signatures, which are unchanged. [Graph_map.create]'s output check is
   positional over the GRAPH's outputs, and those are in the same order. Only
   [Map_verify] catches it, because [Eval_symbolic4] emits one stage per output
   ORDINAL and the swapped stages ground to different terms.

   Both sides are hand-built rather than lowered, which is what lets the
   destination differ from what the lowerer would produce. *)
let unbind_pair ~swap =
  let src =
    Graph_builder.build ~name:"unbind" ~outputs:Fun.id
      Graph_builder.(
        let* x = input ~shape:sq () in
        unbind { Split.Unbind.axis = Axis.C } x)
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  let dst =
    Builder.build ~outputs:Fun.id
      Builder.(
        let* x = input ~shape:sq4 () in
        unbind Axis4.C x)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  (* Swap the NODE's first two output ids and leave [Graph.outputs] alone. That
     is the mutation nothing structural can see: the graph's signature is
     unchanged, so the positional output check is satisfied, and every slice has
     the same shape, so validation is too. What changed is which ORDINAL each id
     is produced by — the only thing the per-output stages disagree about.
     (Swapping the graph outputs instead is caught immediately by
     [Graph_map.create], which is a different and much weaker claim.) *)
  let dst =
    if not swap then dst
    else
      {
        dst with
        Graph_common.Graph.nodes =
          List.map
            (fun (n : Graph.node) ->
              match n.Graph_common.Node.outputs with
              | a :: b :: rest ->
                  { n with Graph_common.Node.outputs = b :: a :: rest }
              | outs -> { n with Graph_common.Node.outputs = outs })
            dst.Graph_common.Graph.nodes;
      }
  in
  (src, dst)

let%expect_test "mutation: swapping two unbind slices is refuted" =
  let src, dst = unbind_pair ~swap:false in
  mutated "slices in order" src dst;
  let src, dst = unbind_pair ~swap:true in
  mutated "slices swapped" src dst;
  [%expect
    {|
    slices in order            3 clusters: 3 proved (structural)
    slices swapped             3 clusters: 1 proved (structural), 2 refuted (counterexample) |}]
