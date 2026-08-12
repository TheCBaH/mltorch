(* Constant folding: the pass that lets every other pass express constant
   arithmetic as ordinary graph nodes. See .ai/native_transform_design.md §12b.

   The motivating case is the first test — a conv weight sitting behind a
   permute, rearranged on every inference until this hoists it to load time. *)

open Graph_ir

let t_ n = Tensor_id.of_int n

(* Payloads print truncated to eight elements, so the weight ramp deliberately
   varies over H and W (the axes the fixture's permute swaps) and not over N or
   C, which the first eight elements never reach. *)
let hw_ramp shape =
  Tensor.materialize shape (fun c ->
      float_of_int
        ((Dim.to_int (Vec6.get c Axis.H) * 10) + Dim.to_int (Vec6.get c Axis.W)))

let channel_ramp shape base =
  Tensor.materialize shape (fun c ->
      base +. float_of_int (Dim.to_int (Vec6.get c Axis.C)))

let pp_payloads fmt constants =
  Fmt.pf fmt "@[<v>%a@]"
    (Fmt.list ~sep:Fmt.cut (fun fmt (id, payload) ->
         Fmt.pf fmt "@[<h>%a = %a@]" Tensor_id.pp id Tensor.pp payload))
    (Tensor_id.Map.bindings constants)

(* Runs the pipeline and hands back the destination state, so a test can go on
   to evaluate it. Everything a test wants to see is printed here. *)
let run ?(show_before = true) ?(constants = []) g passes k =
  match Rewrite.origin ~constants g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error (Err.Error.kind e)
  | Ok (Rewrite.Origin state) -> (
      if show_before then (
        Format.printf "@[<v 2>before:@,%a@]@." Graph_ir.pp (Rewrite.graph state);
        Format.printf "@[<v 2>payloads:@,%a@]@." pp_payloads
          (Rewrite.constants state));
      match Pass.run_all state passes with
      | Error e -> Format.printf "%a@." Pass.pp_error (Err.Error.kind e)
      | Ok (Rewrite.Step (final, map)) ->
          Format.printf "@[<v 2>after:@,%a@]@." Graph_ir.pp
            (Rewrite.graph final);
          Format.printf "@[<v 2>payloads:@,%a@]@." pp_payloads
            (Rewrite.constants final);
          Format.printf "@[<v 2>map:@,%a@]@." Graph_map.pp map;
          k (Rewrite.graph final) (Rewrite.constants final))

let ignore_result _ _ = ()

(* The graph's externally supplied inputs — constants come from the state. *)
let user_inputs g =
  List.filter (fun id -> Graph_ir.input_kind g id = Input.Input) g.Graph.inputs

let evaluated g ~constants ~inputs =
  match
    Eval_direct.run g
      ~constants:(Tensor_id.Map.bindings constants)
      ~inputs:(List.combine (user_inputs g) inputs)
  with
  | Error e -> Format.asprintf "%a" Eval_direct.pp_error (Err.Error.kind e)
  | Ok env -> (
      match g.Graph.outputs with
      | [ out ] -> Format.asprintf "%a" Tensor.pp (Tensor_id.Map.find out env)
      | _ -> "expected exactly one output")

(* ---- the motivating case -------------------------------------------------- *)

let%expect_test "fold_const: a permuted constant weight becomes data" =
  let w_shape = Graph_fixtures.s 3 1 1 2 2 2 in
  run
    (Graph_fixtures.const_permute ())
    ~constants:[ (t_ 1, hw_ramp w_shape) ]
    [ Fold_const.pass ] ignore_result;
  [%expect
    {|
    before:
      graph
      inputs:
        [t0 f32 [H=3 W=3 C=2] ->[n1],
         t1 f32 [N=3 T=1 D=1 H=2 W=2 C=2] ->[n0] constant]
      nodes:
        n0: [t2 f32 [N=3 T=1 D=1 H=2 W=2 C=2] ->[n1]] =
          permute x=t1 perm=[H<-W, W<-H]
        n1: [t3 f32 [H=2 W=2 C=3]] =
          conv2d
            x=t0
            weight=t2 <-n0
            bias=none
            params={h={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   in_channels=2;
                   groups=1}
      outputs: [t3 f32 [H=2 W=2 C=3] <-n1]
    payloads:
      t1 = tensor f32 [N=3 T=1 D=1 H=2 W=2 C=2] {0, 0, 1, 1, 10, 10, 11, 11, ...}
    after:
      graph
      inputs:
        [t0 f32 [H=3 W=3 C=2] ->[n1],
         t2 f32 [N=3 T=1 D=1 H=2 W=2 C=2] ->[n1] constant]
      nodes:
        n1: [t3 f32 [H=2 W=2 C=3]] =
          conv2d
            x=t0
            weight=t2
            bias=none
            params={h={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   in_channels=2;
                   groups=1}
      outputs: [t3 f32 [H=2 W=2 C=3] <-n1]
    payloads:
      t2 = tensor f32 [N=3 T=1 D=1 H=2 W=2 C=2] {0, 0, 10, 10, 1, 1, 11, 11, ...}
    map:
      values:
        {t1} -> {} identical
      nodes:
        {n0} -> {}
      provenance:
        {t1} -> t2 |}]

let%expect_test "fold_const: the folded graph computes the same output" =
  (* The property that matters: hoisting the permute to load time must not
     change what the graph computes. The input has to vary over H and W — a
     conv sums over its kernel window, so with an input flat in those axes the
     result is insensitive to the weight's H/W layout and a wrongly folded
     weight would still agree. *)
  let w_shape = Graph_fixtures.s 3 1 1 2 2 2 in
  let g = Graph_fixtures.const_permute () in
  let weights = [ (t_ 1, hw_ramp w_shape) ] in
  let x = hw_ramp (Graph_fixtures.nhwc ~h:3 ~w:3 ~c:2) in
  let before =
    evaluated g
      ~constants:
        (List.fold_left
           (fun m (id, p) -> Tensor_id.Map.add id p m)
           Tensor_id.Map.empty weights)
      ~inputs:[ x ]
  in
  Format.printf "before: %s@." before;
  run ~show_before:false g ~constants:weights [ Fold_const.pass ]
    (fun g' constants' ->
      let after = evaluated g' ~constants:constants' ~inputs:[ x ] in
      Format.printf "after:  %s@." after;
      Format.printf "same:   %b@." (String.equal before after));
  [%expect
    {|
    before: tensor f32 [H=2 W=2 C=3] {282, 282, 282, 326, 326, 326, 722, 722, ...}
    after:
      graph
      inputs:
        [t0 f32 [H=3 W=3 C=2] ->[n1],
         t2 f32 [N=3 T=1 D=1 H=2 W=2 C=2] ->[n1] constant]
      nodes:
        n1: [t3 f32 [H=2 W=2 C=3]] =
          conv2d
            x=t0
            weight=t2
            bias=none
            params={h={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   in_channels=2;
                   groups=1}
      outputs: [t3 f32 [H=2 W=2 C=3] <-n1]
    payloads:
      t2 = tensor f32 [N=3 T=1 D=1 H=2 W=2 C=2] {0, 0, 10, 10, 1, 1, 11, 11, ...}
    map:
      values:
        {t1} -> {} identical
      nodes:
        {n0} -> {}
      provenance:
        {t1} -> t2
    after:  tensor f32 [H=2 W=2 C=3] {282, 282, 282, 326, 326, 326, 722, 722, ...}
    same:   true |}]

(* ---- a whole sub-DAG ------------------------------------------------------ *)

let%expect_test "fold_const: a constant sub-DAG collapses under a fixpoint" =
  (* One layer per sweep: n1 only becomes foldable once the sweep before it
     bound t4. The add survives — t0 is a real input, so it is not constant —
     and t4's payload is dropped once nothing reads it. *)
  let shape = Graph_fixtures.s1c 3 in
  run
    (Graph_fixtures.const_arith ())
    ~constants:
      [
        (t_ 1, channel_ramp shape 1.);
        (t_ 2, channel_ramp shape 10.);
        (t_ 3, channel_ramp shape 100.);
      ]
    [ Pass.fixpoint Fold_const.pass ]
    ignore_result;
  [%expect
    {|
    before:
      graph
      inputs:
        [t0 f32 [C=3] ->[n2], t1 f32 [C=3] ->[n0] constant,
         t2 f32 [C=3] ->[n0] constant, t3 f32 [C=3] ->[n1] constant]
      nodes:
        n0: [t4 f32 [C=3] ->[n1]] = mul a=t1 b=t2
        n1: [t5 f32 [C=3] ->[n2]] = mul a=t4 <-n0 b=t3
        n2: [t6 f32 [C=3]] = add a=t0 b=t5 <-n1
      outputs: [t6 f32 [C=3] <-n2]
    payloads:
      t1 = tensor f32 [C=3] {1, 2, 3}
      t2 = tensor f32 [C=3] {10, 11, 12}
      t3 = tensor f32 [C=3] {100, 101, 102}
    after:
      graph
      inputs: [t0 f32 [C=3] ->[n2], t5 f32 [C=3] ->[n2] constant]
      nodes:
        n2: [t6 f32 [C=3]] = add a=t0 b=t5
      outputs: [t6 f32 [C=3] <-n2]
    payloads:
      t5 = tensor f32 [C=3] {1000, 2222, 3672}
    map:
      values:
        {t1} -> {} identical
        {t2} -> {} identical
        {t3} -> {} identical
        {t4} -> {} identical
      nodes:
        {n0} -> {}
        {n1} -> {}
      provenance:
        {t1, t2, t3} -> t5 |}]

(* ---- what it refuses ------------------------------------------------------ *)

let%expect_test "fold_const: a constant with no payload is left alone" =
  (* Payloads live outside the IR, so "declared constant, not yet loaded" is a
     legitimate state rather than an error — the node simply stays. *)
  run ~show_before:false
    (Graph_fixtures.const_permute ())
    [ Fold_const.pass ] ignore_result;
  [%expect
    {|
    after:
      graph
      inputs:
        [t0 f32 [H=3 W=3 C=2] ->[n1],
         t1 f32 [N=3 T=1 D=1 H=2 W=2 C=2] ->[n0] constant]
      nodes:
        n0: [t2 f32 [N=3 T=1 D=1 H=2 W=2 C=2] ->[n1]] =
          permute x=t1 perm=[H<-W, W<-H]
        n1: [t3 f32 [H=2 W=2 C=3]] =
          conv2d
            x=t0
            weight=t2 <-n0
            bias=none
            params={h={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   in_channels=2;
                   groups=1}
      outputs: [t3 f32 [H=2 W=2 C=3] <-n1]
    payloads:

    map:
      values:
        identity
      nodes:
        identity
      provenance:
        none |}]

(* The scalar-parameter and clamping ops reach [Fold_const] through the generic
   operand path, with no per-op arm anywhere in the pass — so a constant chain
   through all five collapses to a single payload under a fixpoint. *)
let%expect_test "fold_const: the scalar and clamping pointwise ops fold" =
  run ~show_before:false
    (Graph_fixtures.const_pointwise ())
    ~constants:
      [ (t_ 0, channel_ramp (Graph_fixtures.nhwc ~h:1 ~w:1 ~c:4) (-4.)) ]
    [ Pass.fixpoint Fold_const.pass ]
    ignore_result;
  [%expect
    {|
    after:
      graph
      inputs: [t5 f32 [C=4] constant]
      nodes:

      outputs: [t5 f32 [C=4]]
    payloads:
      t5 = tensor f32 [C=4] {0, 0, 0.166667, 0.333333}
    map:
      values:
        {t0} -> {} identical
        {t1} -> {} identical
        {t2} -> {} identical
        {t3} -> {} identical
        {t4} -> {} identical
      nodes:
        {n0} -> {}
        {n1} -> {}
        {n2} -> {}
        {n3} -> {}
        {n4} -> {}
      provenance:
        {t0} -> t5 |}]

let%expect_test "fold_const: a multi-output node is out of scope" =
  (* Foldable in principle, but binding two outputs needs one recipe that says
     so, which [Recipe.fold_to_constant] does not express. *)
  run ~show_before:false
    (Graph_fixtures.const_pool ())
    ~constants:[ (t_ 0, hw_ramp (Graph_fixtures.nhwc ~h:4 ~w:4 ~c:2)) ]
    [ Pass.fixpoint Fold_const.pass ]
    ignore_result;
  [%expect
    {|
    after:
      graph
      inputs: [t0 f32 [H=4 W=4 C=2] ->[n0] constant]
      nodes:
        n0: [t1 f32 [H=2 W=2 C=2] ->[n2], t2 f32 [H=2 W=2 C=2] ->[n1]] =
          max_pool2d_with_indices
            x=t0
            params={kernel={h=2; w=2}; stride={h=2; w=2}; pad={h=0; w=0}}
        n1: [] = discard x=t2 <-n0
        n2: [t3 f32 [H=2 W=2 C=2]] = relu x=t1 <-n0
      outputs: [t3 f32 [H=2 W=2 C=2] <-n2]
    payloads:
      t0 = tensor f32 [H=4 W=4 C=2] {0, 0, 1, 1, 2, 2, 3, 3, ...}
    map:
      values:
        identity
      nodes:
        identity
      provenance:
        none |}]

(* ---- composed with the permute passes ------------------------------------- *)

let%expect_test "chain_permute then fold_const hoist a two-permute weight" =
  (* The architecture claim in one test: a pass rearranging a constant emits
     graph nodes, and folding turns the result into data. Neither pass knows
     about the other. The two perms differ so the fused one is a real
     rearrangement — with a cancelling pair the run would be trimmed away and
     there would be nothing left to fold. *)
  let w_shape = Graph_fixtures.s 3 1 1 2 2 2 in
  let g =
    Graph_builder.build ~name:"twice_permuted_weight"
      ~outputs:(fun o -> [ o ])
      Graph_builder.(
        let* x = input ~shape:(Graph_fixtures.nhwc ~h:3 ~w:3 ~c:2) () in
        let* w = constant ~shape:w_shape () in
        let* a = permute Graph_fixtures.swap_hw w in
        let* b = permute Graph_fixtures.swap_wc a in
        conv2d (Graph_fixtures.conv_params ~in_channels:2) ~x ~weight:b ())
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  run g
    ~constants:[ (t_ 1, hw_ramp w_shape) ]
    [ Pass.fixpoint Chain_permute.pass; Pass.fixpoint Fold_const.pass ]
    ignore_result;
  [%expect
    {|
    before:
      graph
      inputs:
        [t0 f32 [H=3 W=3 C=2] ->[n2],
         t1 f32 [N=3 T=1 D=1 H=2 W=2 C=2] ->[n0] constant]
      nodes:
        n0: [t2 f32 [N=3 T=1 D=1 H=2 W=2 C=2] ->[n1]] =
          permute x=t1 perm=[H<-W, W<-H]
        n1: [t3 f32 [N=3 T=1 D=1 H=2 W=2 C=2] ->[n2]] =
          permute x=t2 <-n0 perm=[W<-C, C<-W]
        n2: [t4 f32 [H=2 W=2 C=3]] =
          conv2d
            x=t0
            weight=t3 <-n1
            bias=none
            params={h={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   in_channels=2;
                   groups=1}
      outputs: [t4 f32 [H=2 W=2 C=3] <-n2]
    payloads:
      t1 = tensor f32 [N=3 T=1 D=1 H=2 W=2 C=2] {0, 0, 1, 1, 10, 10, 11, 11, ...}
    after:
      graph
      inputs:
        [t0 f32 [H=3 W=3 C=2] ->[n2],
         t3 f32 [N=3 T=1 D=1 H=2 W=2 C=2] ->[n2] constant]
      nodes:
        n2: [t4 f32 [H=2 W=2 C=3]] =
          conv2d
            x=t0
            weight=t3
            bias=none
            params={h={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   in_channels=2;
                   groups=1}
      outputs: [t4 f32 [H=2 W=2 C=3] <-n2]
    payloads:
      t3 = tensor f32 [N=3 T=1 D=1 H=2 W=2 C=2] {0, 10, 0, 10, 1, 11, 1, 11, ...}
    map:
      values:
        {t1} -> {} identical
        {t2} -> {} identical
      nodes:
        {n0, n1} -> {}
      provenance:
        {t1} -> t3 |}]
