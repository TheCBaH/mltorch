(* Folding an inference batch norm into the conv feeding it — the first rewrite
   that changes values rather than rearranging them, so the first to claim
   [Equivalent] rather than [Identical].
   See .ai/native_transform_design.md §12c.

   The load-bearing test is the numeric one: all eight combinations of the
   independently optional operands (conv bias × BN weight × BN bias), each
   evaluated before and after. The mixed cases are the ones most likely to be
   wrong, and a shape error would not catch any of them. *)

open Graph_ir

let s1c = Graph_fixtures.s1c
let nhwc = Graph_fixtures.nhwc

(* ---- values -------------------------------------------------------------- *)

let vec values =
  Tensor.materialize (s1c 3) (fun c -> values.(Dim.to_int (Vec6.get c Axis.C)))

let hw_ramp shape =
  Tensor.materialize shape (fun c ->
      float_of_int
        ((Dim.to_int (Vec6.get c Axis.N) * 7)
        + (Dim.to_int (Vec6.get c Axis.H) * 3)
        + Dim.to_int (Vec6.get c Axis.W)
        + Dim.to_int (Vec6.get c Axis.C))
      /. 4.)

let weight_shape = Graph_fixtures.weight_shape ~out_channels:3 ~in_channels:2
let x_shape = nhwc ~h:4 ~w:4 ~c:2

(* Roles in the order the case builder allocates them, so zipping against the
   graph's constant inputs lines them up. A divergence shows up immediately as a
   numeric mismatch rather than as a silent mislabelling. *)
let payload_for = function
  | `Weight -> hw_ramp weight_shape
  | `Conv_bias -> vec [| 1.; 2.; 3. |]
  | `Bn_weight -> vec [| 2.; 0.5; 1.5 |]
  | `Bn_bias -> vec [| -1.; 0.5; 2. |]
  | `Mean -> vec [| 0.5; 1.; 1.5 |]
  | `Var -> vec [| 4.; 1.; 0.25 |]

let roles ~conv_bias ~bn_weight ~bn_bias =
  [ `Weight ]
  @ (if conv_bias then [ `Conv_bias ] else [])
  @ (if bn_weight then [ `Bn_weight ] else [])
  @ (if bn_bias then [ `Bn_bias ] else [])
  @ [ `Mean; `Var ]

(* ---- graphs -------------------------------------------------------------- *)

let optional flag shape =
  let open Graph_builder in
  if flag then
    let+ id = constant ~shape () in
    Some id
  else return None

let case ~conv_bias ~bn_weight ~bn_bias =
  match
    Graph_builder.build ~name:"conv_bn"
      ~outputs:(fun o -> [ o ])
      Graph_builder.(
        let* x = input ~shape:x_shape () in
        let* w = constant ~shape:weight_shape () in
        let* b = optional conv_bias (s1c 3) in
        let* gamma = optional bn_weight (s1c 3) in
        let* beta = optional bn_bias (s1c 3) in
        let* mean = constant ~shape:(s1c 3) () in
        let* var = constant ~shape:(s1c 3) () in
        let* y =
          conv2d
            (Graph_fixtures.conv_params ~in_channels:2)
            ~x ~weight:w ?bias:b ()
        in
        let* n =
          batch_norm Graph_fixtures.bn_params ~x:y ?weight:gamma ?bias:beta
            ~running_mean:mean ~running_var:var ()
        in
        relu n)
  with
  | Ok g -> g
  | Error e ->
      invalid_arg
        (Format.asprintf "%a" Graph_builder.pp_error e.Core.Error.kind)

let constants_of g ~conv_bias ~bn_weight ~bn_bias =
  let ids =
    List.filter
      (fun id -> Graph_ir.input_kind g id = Input.Constant)
      g.Graph.inputs
  in
  List.combine ids (List.map payload_for (roles ~conv_bias ~bn_weight ~bn_bias))

(* ---- evaluation ----------------------------------------------------------- *)

let user_inputs g =
  List.filter (fun id -> Graph_ir.input_kind g id = Input.Input) g.Graph.inputs

let output g ~constants ~inputs =
  match
    Eval_direct.run g ~constants ~inputs:(List.combine (user_inputs g) inputs)
  with
  | Error e ->
      invalid_arg (Format.asprintf "%a" Eval_direct.pp_error e.Core.Error.kind)
  | Ok env -> (
      match g.Graph.outputs with
      | [ out ] -> Tensor_id.Map.find out env
      | _ -> invalid_arg "expected exactly one output")

let fold_over (Tensor.Tensor t) f init =
  Vec6.fold_coords t.Tensor.shape ~init ~f:(fun acc coord ->
      let i = (Vec6.offset t.Tensor.shape coord :> int) in
      let c = Dim.to_int (Vec6.get coord Axis.C) in
      f acc (Payload.get_float t.Tensor.payload ~c ~i))

let max_abs t = fold_over t (fun acc v -> Float.max acc (Float.abs v)) 0.

let max_abs_diff (Tensor.Tensor a) (Tensor.Tensor b) =
  Vec6.fold_coords a.Tensor.shape ~init:0. ~f:(fun acc coord ->
      let i = (Vec6.offset a.Tensor.shape coord :> int) in
      let c = Dim.to_int (Vec6.get coord Axis.C) in
      Float.max acc
        (Float.abs
           (Payload.get_float a.Tensor.payload ~c ~i
           -. Payload.get_float b.Tensor.payload ~c ~i)))

(* The claim is [Equivalent], not [Identical]: the two graphs are equal in exact
   arithmetic but round differently, so the check is a relative tolerance. A
   printed magnitude would make the golden depend on the platform's floating
   point; a verdict does not. *)
let agrees before after =
  max_abs_diff before after <= 1e-5 *. Float.max 1. (max_abs before)

(* ---- pipeline ------------------------------------------------------------- *)

let run ?(show = true) ?(passes = [ Fold_batch_norm.pass ]) g ~constants k =
  match Rewrite.origin ~constants g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error e.Core.Error.kind
  | Ok (Rewrite.Origin state) -> (
      match Pass.run_all state passes with
      | Error e -> Format.printf "%a@." Pass.pp_error e.Core.Error.kind
      | Ok (Rewrite.Step (final, map)) ->
          if show then (
            Format.printf "@[<v 2>after:@,%a@]@." Graph_ir.pp
              (Rewrite.graph final);
            Format.printf "@[<v 2>map:@,%a@]@." Graph_map.pp map);
          k (Rewrite.graph final)
            (Tensor_id.Map.bindings (Rewrite.constants final)))

let ignore_result _ _ = ()

(* ---- the rewrite ---------------------------------------------------------- *)

let%expect_test "fold_batch_norm: the normalisation moves into the conv" =
  (* Every parameter becomes a graph node; the conv keeps its shape but takes a
     computed weight and bias. The relu's output is claimed [Equivalent] too,
     propagated from the fresh conv output — leaving it implicitly [Identical]
     would have a verifier assert bit-equality on the model's own output. *)
  let g = case ~conv_bias:true ~bn_weight:true ~bn_bias:true in
  Format.printf "@[<v 2>before:@,%a@]@." Graph_ir.pp g;
  run g
    ~constants:(constants_of g ~conv_bias:true ~bn_weight:true ~bn_bias:true)
    ignore_result;
  [%expect
    {|
    before:
      graph
      inputs:
        [t0 f32 [H=4 W=4 C=2], t1 f32 [N=3 T=1 D=1 H=2 W=2 C=2] constant,
         t2 f32 [C=3] constant, t3 f32 [C=3] constant, t4 f32 [C=3] constant,
         t5 f32 [C=3] constant, t6 f32 [C=3] constant]
      nodes:
        n0: [t7 f32 [H=3 W=3 C=3]] =
          conv2d
            x=t0
            weight=t1
            bias=t2
            params={h={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   in_channels=2;
                   groups=1}
        n1: [t8 f32 [H=3 W=3 C=3]] =
          batch_norm
            x=t7
            weight=t3
            bias=t4
            running_mean=t5
            running_var=t6
            params={channel=C; eps=1e-05}
        n2: [t9 f32 [H=3 W=3 C=3]] = relu x=t8
      outputs: [t9 f32 [H=3 W=3 C=3]]
    after:
      graph
      inputs:
        [t0 f32 [H=4 W=4 C=2], t1 f32 [N=3 T=1 D=1 H=2 W=2 C=2] constant,
         t2 f32 [C=3] constant, t3 f32 [C=3] constant, t4 f32 [C=3] constant,
         t5 f32 [C=3] constant, t6 f32 [C=3] constant, t10 f32 [C=1] constant]
      nodes:
        n3: [t11 f32 [C=3]] = add a=t6 b=t10
        n8: [t16 f32 [C=3]] = sub a=t2 b=t5
        n4: [t12 f32 [C=3]] = sqrt x=t11
        n5: [t13 f32 [C=3]] = div a=t3 b=t12
        n6: [t14 f32 [N=3 T=1 D=1 H=1 W=1 C=1]] = permute x=t13 perm=[N<-C, C<-N]
        n9: [t17 f32 [C=3]] = mul a=t16 b=t13
        n7: [t15 f32 [N=3 T=1 D=1 H=2 W=2 C=2]] = mul a=t1 b=t14
        n10: [t18 f32 [C=3]] = add a=t17 b=t4
        n11: [t19 f32 [H=3 W=3 C=3]] =
          conv2d
            x=t0
            weight=t15
            bias=t18
            params={h={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   in_channels=2;
                   groups=1}
        n2: [t9 f32 [H=3 W=3 C=3]] = relu x=t19
      outputs: [t9 f32 [H=3 W=3 C=3]]
    map:
      values:
        {t7} -> {} identical
        {t8} -> {t19} equivalent
        {t9} -> {t9} equivalent
        {} -> {t10} identical
        {} -> {t11} identical
        {} -> {t12} identical
        {} -> {t13} identical
        {} -> {t14} identical
        {} -> {t15} identical
        {} -> {t16} identical
        {} -> {t17} identical
        {} -> {t18} identical
      nodes:
        {n0, n1} -> {n3, n4, n5, n6, n7, n8, n9, n10, n11}
      provenance:
        none |}]

let%expect_test "fold_batch_norm then fold_const leave two constants" =
  (* The point of emitting the parameter arithmetic as nodes: folding collapses
     all of it, and the result is a plain conv with a computed weight and bias. *)
  let g = case ~conv_bias:true ~bn_weight:true ~bn_bias:true in
  run g
    ~passes:[ Fold_batch_norm.pass; Pass.fixpoint Fold_const.pass ]
    ~constants:(constants_of g ~conv_bias:true ~bn_weight:true ~bn_bias:true)
    ignore_result;
  [%expect
    {|
    after:
      graph
      inputs:
        [t0 f32 [H=4 W=4 C=2], t15 f32 [N=3 T=1 D=1 H=2 W=2 C=2] constant,
         t18 f32 [C=3] constant]
      nodes:
        n11: [t19 f32 [H=3 W=3 C=3]] =
          conv2d
            x=t0
            weight=t15
            bias=t18
            params={h={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   in_channels=2;
                   groups=1}
        n2: [t9 f32 [H=3 W=3 C=3]] = relu x=t19
      outputs: [t9 f32 [H=3 W=3 C=3]]
    map:
      values:
        {t1} -> {} identical
        {t2} -> {} identical
        {t3} -> {} identical
        {t4} -> {} identical
        {t5} -> {} identical
        {t6} -> {} identical
        {t7} -> {} identical
        {t8} -> {t19} equivalent
        {t9} -> {t9} equivalent
        {} -> {t15} identical
        {} -> {t18} identical
      nodes:
        {n0, n1} -> {n11}
      provenance:
        {t1, t3, t6} -> t15
        {t2, t3, t4, t5, t6} -> t18 |}]

(* ---- numeric equivalence, all eight combinations -------------------------- *)

let%expect_test "fold_batch_norm: every operand combination agrees numerically"
    =
  let x = hw_ramp x_shape in
  List.iter
    (fun (conv_bias, bn_weight, bn_bias) ->
      let g = case ~conv_bias ~bn_weight ~bn_bias in
      let constants = constants_of g ~conv_bias ~bn_weight ~bn_bias in
      let before = output g ~constants ~inputs:[ x ] in
      run ~show:false g ~constants
        ~passes:[ Fold_batch_norm.pass; Pass.fixpoint Fold_const.pass ]
        (fun g' constants' ->
          let after = output g' ~constants:constants' ~inputs:[ x ] in
          Format.printf
            "conv_bias=%-5b bn_weight=%-5b bn_bias=%-5b  agrees=%b@." conv_bias
            bn_weight bn_bias (agrees before after)))
    [
      (false, false, false);
      (false, false, true);
      (false, true, false);
      (false, true, true);
      (true, false, false);
      (true, false, true);
      (true, true, false);
      (true, true, true);
    ];
  [%expect
    {|
    conv_bias=false bn_weight=false bn_bias=false  agrees=true
    conv_bias=false bn_weight=false bn_bias=true   agrees=true
    conv_bias=false bn_weight=true  bn_bias=false  agrees=true
    conv_bias=false bn_weight=true  bn_bias=true   agrees=true
    conv_bias=true  bn_weight=false bn_bias=false  agrees=true
    conv_bias=true  bn_weight=false bn_bias=true   agrees=true
    conv_bias=true  bn_weight=true  bn_bias=false  agrees=true
    conv_bias=true  bn_weight=true  bn_bias=true   agrees=true |}]

(* ---- what it refuses ------------------------------------------------------ *)

let matches g =
  match Graph_view.of_graph g with
  | Error e -> Format.printf "view: %a@." Graph_view.pp_error e.Core.Error.kind
  | Ok view ->
      Format.printf "matches: %d@."
        (List.length (Pattern.scan Fold_batch_norm.pattern view))

let%expect_test "fold_batch_norm: channel counts must agree" =
  (* The scale is permuted onto the weight's N axis, so a statistics vector of a
     different length would broadcast against the wrong extent — caught here
     rather than by a shape error inside the rewrite. *)
  let g =
    match
      Graph_builder.build ~name:"mismatch"
        ~outputs:(fun o -> [ o ])
        Graph_builder.(
          let* x = input ~shape:x_shape () in
          let* w = constant ~shape:weight_shape () in
          (* The conv's Cout is 3; give the statistics 2 channels so the
             weight's N and the parameters' C disagree. *)
          let* mean = constant ~shape:(s1c 2) () in
          let* var = constant ~shape:(s1c 2) () in
          let* y =
            conv2d (Graph_fixtures.conv_params ~in_channels:2) ~x ~weight:w ()
          in
          batch_norm Graph_fixtures.bn_params ~x:y ~running_mean:mean
            ~running_var:var ())
    with
    | Ok g -> g
    | Error e ->
        invalid_arg
          (Format.asprintf "%a" Graph_builder.pp_error e.Core.Error.kind)
  in
  matches g;
  [%expect {| matches: 0 |}]

let%expect_test "fold_batch_norm: a non-constant parameter is refused" =
  (* Folding a runtime parameter would move a per-channel scale of the output
     onto the whole weight tensor — more work per inference, not less. *)
  let g =
    match
      Graph_builder.build ~name:"runtime_gamma"
        ~outputs:(fun o -> [ o ])
        Graph_builder.(
          let* x = input ~shape:x_shape () in
          let* w = constant ~shape:weight_shape () in
          let* gamma = input ~shape:(s1c 3) () in
          let* mean = constant ~shape:(s1c 3) () in
          let* var = constant ~shape:(s1c 3) () in
          let* y =
            conv2d (Graph_fixtures.conv_params ~in_channels:2) ~x ~weight:w ()
          in
          batch_norm Graph_fixtures.bn_params ~x:y ~weight:gamma
            ~running_mean:mean ~running_var:var ())
    with
    | Ok g -> g
    | Error e ->
        invalid_arg
          (Format.asprintf "%a" Graph_builder.pp_error e.Core.Error.kind)
  in
  matches g;
  [%expect {| matches: 0 |}]

let%expect_test "fold_batch_norm: a shared conv output is refused" =
  (* Removing the conv would leave the second consumer without the
     unnormalised value. *)
  let g =
    match
      Graph_builder.build ~name:"shared_conv"
        ~outputs:(fun o -> [ o ])
        Graph_builder.(
          let* x = input ~shape:x_shape () in
          let* w = constant ~shape:weight_shape () in
          let* mean = constant ~shape:(s1c 3) () in
          let* var = constant ~shape:(s1c 3) () in
          let* y =
            conv2d (Graph_fixtures.conv_params ~in_channels:2) ~x ~weight:w ()
          in
          let* n =
            batch_norm Graph_fixtures.bn_params ~x:y ~running_mean:mean
              ~running_var:var ()
          in
          add n y)
    with
    | Ok g -> g
    | Error e ->
        invalid_arg
          (Format.asprintf "%a" Graph_builder.pp_error e.Core.Error.kind)
  in
  matches g;
  [%expect {| matches: 0 |}]
