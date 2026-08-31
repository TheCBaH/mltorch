(* Native4D evaluation. Two properties, checked against different oracles.

   1. DIRECT values, against numbers worked out by hand. Not against Native:
      both sides instantiate the same [Compute (S)] functor, so agreement would
      show the parameter translation is self-consistent and say nothing about
      whether it is right.

   2. DIRECT versus grounded SYMBOLIC, bitwise. That one IS a self-comparison —
      the two modes share [Eval_op4] — and is worth having for what it does
      cover: staging, scheduling, the shape machinery, and the [Symbolic]
      instance's binding of constants. It is the stage-3 acceptance criterion.

   What each catches, measured by breaking [Eval_op4] and watching, rather than
   assumed:

   - A WRONG GROUP COUNT is not caught here at all. Setting
     [Depthwise_conv2d]'s groups to 1 raises out of Native's own convolution
     validation — "weight C extent 1 must equal in_channels/groups 2" — because
     a depthwise weight has one input channel per group. The compute these arms
     delegate to guards its own parameters, which is an argument for delegating.

   - A WRONG PERMUTATION MAP is caught only by property 1. Making [perm6] the
     identity leaves the per-op table entirely green, since both modes share the
     translation, and moves the permute test's output from
     [out [H=3 W=2]: [0 3 1 4 2 5]] to [out [H=2 W=3]: [0 1 2 3 4 5]].

   So the two properties are not redundant, and neither subsumes the other. *)

open Native4d

let s4 ~n ~h ~w ~c = Shape4.of_ints ~n ~h ~w ~c

let build ~outputs m =
  Builder.build ~outputs m |> Err.or_raise ~pp_error:Builder.pp_error

(* Row-major fill, so every element is distinguishable in the output. *)
let seq shape =
  let i = ref (-1.) in
  Tensor.materialize (Shape4.to_vec6 shape) (fun _ ->
      i := !i +. 1.;
      !i)

let values t =
  let (Tensor.Tensor tt) = t in
  let shape = tt.Tensor.shape in
  let acc = ref [] in
  Vec6.iter shape (fun c -> acc := Tensor.read_at t (Vec6.get c) :: !acc);
  List.rev !acc

let pp_values fmt vs =
  Fmt.pf fmt "[%a]"
    (Fmt.list ~sep:(Fmt.any " ") (fun fmt v -> Fmt.pf fmt "%g" v))
    vs

(* ---- direct, against hand values ------------------------------------------ *)

let run_direct ?(constants = []) g ~inputs =
  Eval_direct4.run g ~constants ~inputs
  |> Err.or_raise ~pp_error:Eval_direct4.pp_error

let single g ~inputs ?(constants = []) () =
  let env = run_direct g ~constants ~inputs in
  let out = List.hd g.Graph.Graph.outputs in
  Tensor_id.Map.find out env

let%expect_test "direct4: pointwise" =
  let shape = s4 ~n:1 ~h:1 ~w:1 ~c:4 in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape () in
       let* y = input ~shape () in
       let* s = add x y in
       relu s)
  in
  (* -1.5 makes the sum straddle zero, so relu clamps exactly half. An offset
     that clamped all four or none would be indistinguishable from a relu that
     was not applied at all. *)
  let a = Tensor.materialize (Shape4.to_vec6 shape) (fun _ -> -1.5) in
  let b = seq shape in
  let ids = g.Graph.Graph.inputs in
  Format.printf "relu(-1.5 + [0 1 2 3]), expect [0 0 0.5 1.5]: %a@." pp_values
    (values (single g ~inputs:[ (List.nth ids 0, a); (List.nth ids 1, b) ] ()));
  [%expect {| relu(-1.5 + [0 1 2 3]), expect [0 0 0.5 1.5]: [0 0 0.5 1.5] |}]

(* H fans from 1 to 3 and C fans from 1 to 2, W stays fixed at its own
   extent: a wrong per-axis broadcast coordinate would repeat the wrong
   values across H or C instead of W's own two. *)
let%expect_test "direct4: expand4 fans H and C, keeps W" =
  let x_shape = s4 ~n:1 ~h:1 ~w:2 ~c:1 in
  let target = s4 ~n:1 ~h:3 ~w:2 ~c:2 in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:x_shape () in
       expand4 target x)
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 x_shape) (fun c ->
        if Dim.to_int (Vec6.get c Axis.W) = 0 then 3. else 7.)
  in
  let id = List.hd g.Graph.Graph.inputs in
  Format.printf "%a@." pp_values (values (single g ~inputs:[ (id, x) ] ()));
  [%expect {| [3 3 7 7 3 3 7 7 3 3 7 7] |}]

let%expect_test "direct4: unbind preserves int64 cells" =
  let shape = s4 ~n:1 ~h:1 ~w:2 ~c:2 in
  let i64 = Payload.Fmt Payload.I64 in
  let g =
    Builder.build ~dtype:i64 ~outputs:Fun.id
      (let open Builder in
       let* x = input ~shape () in
       unbind Axis4.C x)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  let data =
    Bigarray.(
      Array1.of_array int64 c_layout
        [| Int64.min_int; -1L; 9_007_199_254_740_993L; Int64.max_int |])
  in
  let x =
    Tensor.Tensor
      {
        Tensor.shape = Shape4.to_vec6 shape;
        payload = { Payload.fmt = Payload.I64; quant = Payload.No_quant; data };
      }
  in
  let env = run_direct g ~inputs:[ (List.hd g.Graph.Graph.inputs, x) ] in
  List.iter
    (fun id -> Format.printf "%a@." Tensor.pp (Tensor_id.Map.find id env))
    g.Graph.Graph.outputs;
  [%expect
    {|
    tensor i64 [C=2] {-9223372036854775808, 9007199254740993}
    tensor i64 [C=2] {-1, 9223372036854775807} |}]

let%expect_test "direct4: adaptive_avg_pool2d keeps ATen bins" =
  let shape = s4 ~n:1 ~h:5 ~w:5 ~c:1 in
  let params =
    {
      Pool.AdaptiveAvgPool2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 3; w = Op_config.Pos.of_int 3 };
    }
  in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape () in
       adaptive_avg_pool2d params x)
  in
  let x = seq shape in
  let id = List.hd g.Graph.Graph.inputs in
  Format.printf "%a@." pp_values (values (single g ~inputs:[ (id, x) ] ()));
  [%expect {| [3 4.5 6 10.5 12 13.5 18 19.5 21] |}]

let chan c = Dim.to_int (Vec6.get c Axis.C)

let activation_single f ~values:vs =
  let shape = s4 ~n:1 ~h:1 ~w:1 ~c:(List.length vs) in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape () in
       f x)
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 shape) (fun c -> List.nth vs (chan c))
  in
  let ids = g.Graph.Graph.inputs in
  values (single g ~inputs:[ (List.nth ids 0, x) ] ())

(* Hand values, not Native as oracle: [Compute (Direct)] is the same functor
   whichever dialect calls it, so agreement between Native and Native4D here
   would only show the parameter translation is self-consistent, not that the
   formula is right. -6, -0.5, 0.5, 6 are worked out by hand in op5-impl. *)
let%expect_test "direct4: silu" =
  let vs = activation_single Builder.silu ~values:[ -6.; -0.5; 0.5; 6. ] in
  Format.printf "%a@." pp_values vs;
  [%expect {| [-0.0148357 -0.18877 0.31123 5.98516] |}]

let%expect_test "direct4: hardsigmoid" =
  let vs =
    activation_single Builder.hardsigmoid ~values:[ -6.; -0.5; 0.5; 6. ]
  in
  Format.printf "%a@." pp_values vs;
  [%expect {| [0 0.416667 0.583333 1] |}]

let%expect_test "direct4: hardswish" =
  let vs = activation_single Builder.hardswish ~values:[ -6.; -0.5; 0.5; 6. ] in
  Format.printf "%a@." pp_values vs;
  [%expect {| [-0 -0.208333 0.291667 6] |}]

let%expect_test "direct4: leaky_relu" =
  let vs =
    activation_single
      (Builder.leaky_relu { Pointwise.Leaky_relu.negative_slope = 0.2 })
      ~values:[ -2.; -0.5; 0.; 3. ]
  in
  Format.printf "%a@." pp_values vs;
  [%expect {| [-0.4 -0.1 0 3] |}]

let%expect_test "direct4: zeros4 preserves requested F64 dtype" =
  let g =
    build
      ~outputs:(fun y -> [ y ])
      (Builder.zeros4
         {
           Ops4.Zeros4.shape = s4 ~n:1 ~h:1 ~w:2 ~c:2;
           fmt = Payload.Fmt Payload.F64;
         })
  in
  Format.printf "%a@." Tensor.pp (single g ~inputs:[] ());
  [%expect {| tensor f64 [W=2 C=2] {0, 0, 0, 0} |}]

let%expect_test "direct4: arange4 preserves Float start and endpoint" =
  let g =
    build
      ~outputs:(fun y -> [ y ])
      (Builder.arange4
         {
           Ops4.Arange4.start = 0.5;
           stop = 4.;
           step = 1.;
           fmt = Payload.Fmt Payload.F32;
         })
  in
  Format.printf "%a@." Tensor.pp (single g ~inputs:[] ());
  [%expect {| tensor f32 [C=4] {0.5, 1.5, 2.5, 3.5} |}]

let%expect_test "direct4: batch_norm_no_stats keeps activation and statistics" =
  let x_shape = s4 ~n:1 ~h:2 ~w:1 ~c:2 in
  let c_shape = s4 ~n:1 ~h:1 ~w:1 ~c:2 in
  let g =
    build ~outputs:Fun.id
      (let open Builder in
       let* x = input ~shape:x_shape () in
       let* weight = input ~shape:c_shape () in
       let* bias = input ~shape:c_shape () in
       batch_norm_no_stats
         { Ops4.Batch_norm_no_stats.channel = Axis4.C; eps = 0. }
         ~x ~weight ~bias ())
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 x_shape) (fun c ->
        [| [| 1.; 3. |]; [| 5.; 7. |] |].(chan c).(Dim.to_int
                                                     (Vec6.get c Axis.H)))
  in
  let vec2 a b =
    Tensor.materialize (Shape4.to_vec6 c_shape) (fun c -> [| a; b |].(chan c))
  in
  let env =
    run_direct g
      ~inputs:
        (List.combine g.Graph.Graph.inputs [ x; vec2 2. 10.; vec2 1. (-1.) ])
  in
  List.iter
    (fun id ->
      Format.printf "%a@." pp_values (values (Tensor_id.Map.find id env)))
    g.Graph.Graph.outputs;
  [%expect {|
    [-1 -11 3 9]
    [2 6]
    [1 1] |}]

let%expect_test "symbolic4: batch_norm_no_stats reaches the kernel adapter" =
  let x_shape = s4 ~n:1 ~h:2 ~w:1 ~c:2 in
  let c_shape = s4 ~n:1 ~h:1 ~w:1 ~c:2 in
  let g =
    build ~outputs:Fun.id
      (let open Builder in
       let* x = input ~shape:x_shape () in
       let* weight = input ~shape:c_shape () in
       let* bias = input ~shape:c_shape () in
       batch_norm_no_stats
         { Ops4.Batch_norm_no_stats.channel = Axis4.C; eps = 1e-5 }
         ~x ~weight ~bias ())
  in
  (match Kernel_adapt.of_stage_program (Eval_symbolic4.run g) with
  | Ok _ -> print_endline "kernel accepted"
  | Error e -> Format.printf "%a@." Kernel_adapt.pp_error (Err.Error.kind e));
  [%expect {| kernel accepted |}]

(* Depthwise is the arm where a wrong group count is invisible without a hand
   value: with in_channels = 2 and a 1x1 weight, output channel i is
   x[i] * w[i], whereas groups=1 would sum both channels into each output. *)
let%expect_test "direct4: depthwise convolution is per channel" =
  let x_shape = s4 ~n:1 ~h:1 ~w:1 ~c:2 in
  let w_shape = s4 ~n:2 ~h:1 ~w:1 ~c:1 in
  let params : Ops4.Conv_params.t =
    {
      h = Fixtures4.axis_window ~kernel:1;
      w = Fixtures4.axis_window ~kernel:1;
      in_channels = Dim.extent 2;
    }
  in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:x_shape () in
       let* w = constant ~shape:w_shape () in
       depthwise_conv2d params ~x ~weight:w ())
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 x_shape) (fun c ->
        if Dim.to_int (Vec6.get c Axis.C) = 0 then 3. else 5.)
  in
  let w =
    Tensor.materialize (Shape4.to_vec6 w_shape) (fun c ->
        if Dim.to_int (Vec6.get c Axis.N) = 0 then 10. else 100.)
  in
  let ids = g.Graph.Graph.inputs in
  Format.printf "expect [30 500]: %a@." pp_values
    (values
       (single g
          ~inputs:[ (List.nth ids 0, x) ]
          ~constants:[ (List.nth ids 1, w) ]
          ()));
  [%expect {| expect [30 500]: [30 500] |}]

(* The general form, between [Conv2d] (all channels shared) and
   [Depthwise_conv2d] (one channel per group) above: 2 groups of 2 input
   channels each, so a hand value catches a wrong per-group channel slice —
   output channel 0 must read input channels [0;1], never [2;3]. *)
let%expect_test "direct4: grouped convolution reads its own group's channels" =
  let x_shape = s4 ~n:1 ~h:1 ~w:1 ~c:4 in
  let w_shape = s4 ~n:2 ~h:1 ~w:1 ~c:2 in
  let params : Ops4.Grouped_conv_params.t =
    {
      h = Fixtures4.axis_window ~kernel:1;
      w = Fixtures4.axis_window ~kernel:1;
      in_channels = Dim.extent 4;
      groups = Op_config.Pos.of_int 2;
    }
  in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:x_shape () in
       let* w = constant ~shape:w_shape () in
       grouped_conv2d params ~x ~weight:w ())
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 x_shape) (fun c ->
        match Dim.to_int (Vec6.get c Axis.C) with
        | 0 -> 3.
        | 1 -> 5.
        | 2 -> 7.
        | _ -> 11.)
  in
  let w =
    Tensor.materialize (Shape4.to_vec6 w_shape) (fun c ->
        match
          (Dim.to_int (Vec6.get c Axis.N), Dim.to_int (Vec6.get c Axis.C))
        with
        | 0, 0 -> 10.
        | 0, _ -> 1.
        | _, 0 -> 1.
        | _ -> 10.)
  in
  let ids = g.Graph.Graph.inputs in
  Format.printf "expect [35 117]: %a@." pp_values
    (values
       (single g
          ~inputs:[ (List.nth ids 0, x) ]
          ~constants:[ (List.nth ids 1, w) ]
          ()));
  [%expect {| expect [35 117]: [35 117] |}]

(* Two groups of two channels each, mean/var computed only WITHIN a group:
   group0 = {3;5} (mean 4, std 1) and group1 = {7;13} (mean 10, std 3) give
   the SAME normalized pattern [-1;1] despite very different absolute
   scales -- a wrong window (one global mean/var across all four channels)
   would not. *)
let%expect_test "direct4: group_norm4 reduces within its own channel group" =
  let x_shape = s4 ~n:1 ~h:1 ~w:1 ~c:4 in
  let c_shape = s4 ~n:1 ~h:1 ~w:1 ~c:4 in
  let params : Ops4.Group_norm4.params =
    { channel = Axis4.C; groups = Op_config.Pos.of_int 2; eps = 1e-8 }
  in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:x_shape () in
       let* w = constant ~shape:c_shape () in
       let* b = constant ~shape:c_shape () in
       group_norm4 params ~x ~weight:w ~bias:b ())
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 x_shape) (fun c ->
        match Dim.to_int (Vec6.get c Axis.C) with
        | 0 -> 3.
        | 1 -> 5.
        | 2 -> 7.
        | _ -> 13.)
  in
  let ones = Tensor.materialize (Shape4.to_vec6 c_shape) (fun _ -> 1.) in
  let zeros = Tensor.materialize (Shape4.to_vec6 c_shape) (fun _ -> 0.) in
  let ids = g.Graph.Graph.inputs in
  Format.printf "expect [-1 1 -1 1]: %a@." pp_values
    (values
       (single g
          ~inputs:[ (List.nth ids 0, x) ]
          ~constants:[ (List.nth ids 1, ones); (List.nth ids 2, zeros) ]
          ()));
  [%expect {| expect [-1 1 -1 1]: [-1 1 -1 1] |}]

let%expect_test "direct4: mean over H and W keeps the axes" =
  let shape = s4 ~n:1 ~h:2 ~w:2 ~c:1 in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape () in
       mean_keepdims [ Axis4.H; Axis4.W ] x)
  in
  let out = single g ~inputs:[ (List.hd g.Graph.Graph.inputs, seq shape) ] () in
  let (Tensor.Tensor tt) = out in
  Format.printf "shape %a, mean of 0..3 = %a@." Vec6.pp_shape tt.Tensor.shape
    pp_values (values out);
  [%expect {| shape [C=1], mean of 0..3 = [1.5] |}]

let%expect_test "direct4: permute4 swaps H and W" =
  let shape = s4 ~n:1 ~h:2 ~w:3 ~c:1 in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape () in
       permute4 (Ops4.Permute4.of_fn (function H -> W | W -> H | a -> a)) x)
  in
  let out = single g ~inputs:[ (List.hd g.Graph.Graph.inputs, seq shape) ] () in
  let (Tensor.Tensor tt) = out in
  Format.printf "in  [H=2 W=3]: %a@." pp_values (values (seq shape));
  Format.printf "out %a: %a@." Vec6.pp_shape tt.Tensor.shape pp_values
    (values out);
  [%expect
    {|
    in  [H=2 W=3]: [0 1 2 3 4 5]
    out [H=3 W=2 C=1]: [0 3 1 4 2 5] |}]

(* Hand values, both modes. What this checks that the per-op table below cannot:
   the dialect's [Axis4.t] keys reach [Pad.Pad.Compute] as the axes they name.
   Swap H and W in [Graph_shape4.pad_params] and both modes here go red, while
   the direct-versus-symbolic table stays entirely green — both of its sides
   read the same adapter. *)
let%expect_test "direct4: pad4, constant and reflect" =
  let shape = s4 ~n:1 ~h:2 ~w:3 ~c:1 in
  let pad mode pads =
    let g =
      build
        ~outputs:(fun o -> [ o ])
        (let open Builder in
         let* x = input ~shape () in
         pad4 { Ops4.Pad4.pads; mode } x)
    in
    let out =
      single g ~inputs:[ (List.hd g.Graph.Graph.inputs, seq shape) ] ()
    in
    let (Tensor.Tensor tt) = out in
    Format.printf "out %a: %a@." Vec6.pp_shape tt.Tensor.shape pp_values
      (values out)
  in
  Format.printf "in  [H=2 W=3]: %a@." pp_values (values (seq shape));
  (* One column of fill on each side of W, none on H: rows [9 0 1 2 9] and
     [9 3 4 5 9]. *)
  pad (Pad.Pad.Constant 9.) [ (Axis4.W, { Pad.Pad.before = 1; after = 1 }) ];
  (* Mirror about the boundary element, which is NOT repeated: row [0 1 2]
     extended left by one is [1 0 1 2], and by two on the right is [.. 1 0]. *)
  pad Pad.Pad.Reflect [ (Axis4.W, { Pad.Pad.before = 1; after = 2 }) ];
  (* H is the OUTER axis here, so a mirror on it moves whole rows: [0 1 2] and
     [3 4 5] become [3 4 5], [0 1 2], [3 4 5]. A pad that silently used W would
     print the same extents as a W-mirror only if H and W matched, which is why
     this fixture is 2x3. *)
  pad Pad.Pad.Reflect [ (Axis4.H, { Pad.Pad.before = 1; after = 0 }) ];
  (* Cropping: a negative entry removes the last row, and nothing is
     synthesized, so no fill value can appear. *)
  pad (Pad.Pad.Constant 9.) [ (Axis4.H, { Pad.Pad.before = 0; after = -1 }) ];
  [%expect
    {|
    in  [H=2 W=3]: [0 1 2 3 4 5]
    out [H=2 W=5 C=1]: [9 0 1 2 9 9 3 4 5 9]
    out [H=2 W=6 C=1]: [1 0 1 2 1 0 4 3 4 5 4 3]
    out [H=3 W=3 C=1]: [3 4 5 0 1 2 3 4 5]
    out [W=3 C=1]: [0 1 2] |}]

(* ---- direct versus grounded symbolic --------------------------------------- *)

(* The stage-3 acceptance criterion, over one graph per op. Comparison is
   BITWISE — [Core.Float_bits.exact] — for the reason map_verify.mli gives about
   [Float.equal] conflating -0./+0. and every NaN, which are exactly the
   distinctions a max-pool turns on. *)
let same_bits a b =
  let va = values a and vb = values b in
  List.length va = List.length vb
  && List.for_all2 Core.Float_bits.equal_exact va vb

let agree name g ~inputs ~constants =
  let direct = run_direct g ~constants ~inputs in
  let program = Eval_symbolic4.run g in
  let bind id =
    match List.assoc_opt id inputs with
    | Some t -> t
    | None -> (
        match List.assoc_opt id constants with
        | Some t -> t
        | None -> invalid_arg "unbound")
  in
  let grounded = Stage_program.ground program ~bind in
  (* EVERY output, not [List.hd]. For the eighteen single-output ops that is the
     same thing; for [Unbind] it is the difference between comparing the whole
     op and comparing its first slice, and a per-ordinal bug is exactly what a
     multi-output op can have. *)
  let verdict out =
    let d = Tensor_id.Map.find out direct in
    match Tensor_id.Map.find_opt out grounded with
    | None -> "symbolic produced no stage"
    | Some s -> if same_bits d s then "direct = symbolic" else "MISMATCH"
  in
  match g.Graph.Graph.outputs with
  | [ out ] -> Format.printf "%-22s %s@." name (verdict out)
  | outputs ->
      List.iteri
        (fun i out -> Format.printf "%-22s out%d %s@." name i (verdict out))
        outputs

(* Same coverage discipline as op_json_test.ml: an op with no fixture is an op
   whose two evaluation paths have never been compared. *)
let%expect_test "direct4 = symbolic4: every op has a fixture" =
  Format.printf "fixtures: %d, registry: %d@."
    (List.length (Fixtures4.per_op ()))
    (List.length Op.op_registry);
  [%expect {| fixtures: 53, registry: 53 |}]

let%expect_test "direct4 = symbolic4, bitwise, per op" =
  List.iter
    (fun (name, g, inputs, constants) -> agree name g ~inputs ~constants)
    (Fixtures4.per_op ());
  [%expect
    {|
    add                    direct = symbolic
    sub                    direct = symbolic
    mul                    direct = symbolic
    div                    direct = symbolic
    add_scalar             direct = symbolic
    div_scalar             direct = symbolic
    expand4                direct = symbolic
    mul_scalar             direct = symbolic
    pow                    direct = symbolic
    rsub_scalar            direct = symbolic
    clamp                  direct = symbolic
    hardtanh               direct = symbolic
    leaky_relu             direct = symbolic
    zeros4                 direct = symbolic
    arange4                direct = symbolic
    batch_norm_no_stats    out0 direct = symbolic
    batch_norm_no_stats    out1 direct = symbolic
    batch_norm_no_stats    out2 direct = symbolic
    relu                   direct = symbolic
    repeat4                direct = symbolic
    repeat_interleave4     direct = symbolic
    gelu                   direct = symbolic
    sigmoid                direct = symbolic
    silu                   direct = symbolic
    hardsigmoid            direct = symbolic
    hardswish              direct = symbolic
    sqrt                   direct = symbolic
    to_copy                direct = symbolic
    max_pool2d             direct = symbolic
    adaptive_avg_pool2d    direct = symbolic
    adaptive_max_pool2d    direct = symbolic
    avg_pool2d             direct = symbolic
    mean_keepdims          direct = symbolic
    max_keepdims           direct = symbolic
    sum_keepdims           direct = symbolic
    pad4                   direct = symbolic
    slice4                 direct = symbolic
    select4                direct = symbolic
    select_scatter4        direct = symbolic
    concat4                direct = symbolic
    stack4                 direct = symbolic
    permute4               direct = symbolic
    reshape4               direct = symbolic
    rms_norm               direct = symbolic
    layer_norm             direct = symbolic
    group_norm4            direct = symbolic
    conv2d                 direct = symbolic
    depthwise_conv2d       direct = symbolic
    grouped_conv2d         direct = symbolic
    transposed_conv2d      direct = symbolic
    unbind                 out0 direct = symbolic
    unbind                 out1 direct = symbolic
    split_with_sizes4      out0 direct = symbolic
    split_with_sizes4      out1 direct = symbolic
    upsample_bilinear2d    direct = symbolic
    upsample_nearest2d     direct = symbolic
    vector_norm_keepdims   direct = symbolic |}]

(* Hand values, not Native-as-oracle: both sides would instantiate the same
   [Split.Unbind.Compute] functor, so agreement would prove the adapter and the
   staging rather than the arithmetic.

   Unbinding C on [N=1 H=1 W=2 C=3] drops the innermost axis, so H shifts onto W
   and W onto C: each slice is [W=1 C=2], holding one column. Every ordinal is
   printed, since that is what a single-output test cannot see. *)
let%expect_test "direct4: unbind takes one slice per coordinate" =
  let shape = s4 ~n:1 ~h:1 ~w:2 ~c:3 in
  let g =
    build ~outputs:Fun.id
      (let open Builder in
       let* x = input ~shape () in
       unbind Axis4.C x)
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 shape) (fun c ->
        float_of_int
          ((Dim.to_int (Vec6.get c Axis.W) * 10)
          + Dim.to_int (Vec6.get c Axis.C)))
  in
  let env = run_direct g ~inputs:[ (List.hd g.Graph.Graph.inputs, x) ] in
  List.iteri
    (fun i out ->
      Format.printf "out%d = %a@." i Tensor.pp (Tensor_id.Map.find out env))
    g.Graph.Graph.outputs;
  [%expect
    {|
    out0 = tensor f32 [C=2] {0, 10}
    out1 = tensor f32 [C=2] {1, 11}
    out2 = tensor f32 [C=2] {2, 12} |}]

(* Hand values, not Native-as-oracle: both sides would instantiate the same
   [Split.Select.Compute] functor, so agreement would prove the adapter and
   the staging rather than the arithmetic.

   The same shape and value formula [unbind]'s own oracle test above uses, so
   selecting index 1 along C reads the same one-wide window through the same
   repacking as that test's out1 -- checkable against it by eye. *)
let%expect_test "direct4: select4 reads one slice at the chosen index" =
  let shape = s4 ~n:1 ~h:1 ~w:2 ~c:3 in
  let g =
    build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape () in
       select4 { Ops4.Select4.axis = Axis4.C; index = 1 } x)
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 shape) (fun c ->
        float_of_int
          ((Dim.to_int (Vec6.get c Axis.W) * 10)
          + Dim.to_int (Vec6.get c Axis.C)))
  in
  let env = run_direct g ~inputs:[ (List.hd g.Graph.Graph.inputs, x) ] in
  Format.printf "%a@." Tensor.pp
    (Tensor_id.Map.find (List.hd g.Graph.Graph.outputs) env);
  [%expect {| tensor f32 [C=2] {1, 11} |}]

(* Unequal sizes [1;3] on a KEPT axis, unlike [unbind] above: a wrong offset
   for the second piece would read starting from W=0 instead of W=1, so this
   would print {0, 1, 2} for out1 instead of the correct {1, 2, 3}. *)
let%expect_test "direct4: split_with_sizes keeps the axis, sliced by window" =
  let shape = s4 ~n:1 ~h:1 ~w:4 ~c:1 in
  let g =
    build ~outputs:Fun.id
      (let open Builder in
       let* x = input ~shape () in
       split_with_sizes4 Axis4.W [ 1; 3 ] x)
  in
  let x =
    Tensor.materialize (Shape4.to_vec6 shape) (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.W)))
  in
  let env = run_direct g ~inputs:[ (List.hd g.Graph.Graph.inputs, x) ] in
  List.iteri
    (fun i out ->
      Format.printf "out%d = %a@." i Tensor.pp (Tensor_id.Map.find out env))
    g.Graph.Graph.outputs;
  [%expect
    {|
    out0 = tensor f32 [C=1] {0}
    out1 = tensor f32 [W=3 C=1] {1, 2, 3} |}]
