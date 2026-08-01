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
  Builder.build ~outputs m |> Core.or_raise Builder.pp_error

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
  Eval_direct4.run g ~constants ~inputs |> Core.or_raise Eval_direct4.pp_error

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

(* ---- direct versus grounded symbolic --------------------------------------- *)

(* The stage-3 acceptance criterion, over one graph per op. Comparison is
   BITWISE — [Int64.bits_of_float] — for the reason map_verify.mli gives about
   [Float.equal] conflating -0./+0. and every NaN, which are exactly the
   distinctions a max-pool turns on. *)
let same_bits a b =
  let va = values a and vb = values b in
  List.length va = List.length vb
  && List.for_all2
       (fun x y -> Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y))
       va vb

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
  let out = List.hd g.Graph.Graph.outputs in
  let d = Tensor_id.Map.find out direct in
  match Tensor_id.Map.find_opt out grounded with
  | None -> Format.printf "%-22s symbolic produced no stage@." name
  | Some s ->
      Format.printf "%-22s %s@." name
        (if same_bits d s then "direct = symbolic" else "MISMATCH")

(* Same coverage discipline as op_json_test.ml: an op with no fixture is an op
   whose two evaluation paths have never been compared. *)
let%expect_test "direct4 = symbolic4: every op has a fixture" =
  Format.printf "fixtures: %d, registry: %d@."
    (List.length (Fixtures4.per_op ()))
    (List.length Op.op_registry);
  [%expect {| fixtures: 19, registry: 19 |}]

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
    clamp                  direct = symbolic
    hardtanh               direct = symbolic
    relu                   direct = symbolic
    sqrt                   direct = symbolic
    max_pool2d             direct = symbolic
    avg_pool2d             direct = symbolic
    mean_keepdims          direct = symbolic
    permute4               direct = symbolic
    reshape4               direct = symbolic
    rms_norm               direct = symbolic
    conv2d                 direct = symbolic
    depthwise_conv2d       direct = symbolic
    transposed_conv2d      direct = symbolic |}]
