let chan c = Dim.to_int (Vec6.get c Axis.C)
let row c = Dim.to_int (Vec6.get c Axis.H)
let col c = Dim.to_int (Vec6.get c Axis.W)
let s1c n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:n

let%expect_test "Direct: relu pointwise" =
  let module R = Pointwise.Relu.Compute (Direct) in
  let x_shape = s1c 4 in
  let x =
    Tensor.materialize x_shape (fun c -> [| -2.; -0.5; 1.; 3. |].(chan c))
  in
  Format.printf "%a@." Tensor.pp
    (Schedule.evaluate (Pointwise.Relu.output_shape x_shape) (R.pixel x));
  [%expect {| tensor f32 [C=4] {0, 0, 1, 3} |}]

let%expect_test "Direct: add" =
  let module A = Pointwise.Add.Compute (Direct) in
  let x_shape = s1c 3 and y_shape = s1c 3 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (chan c)) in
  let y = Tensor.materialize y_shape (fun _ -> 10.) in
  Format.printf "%a@." Tensor.pp
    (Schedule.evaluate
       (Pointwise.Add.output_shape x_shape y_shape)
       (A.pixel x y));
  [%expect {| tensor f32 [C=3] {10, 11, 12} |}]

let%expect_test "Direct: mul" =
  let module M = Pointwise.Mul.Compute (Direct) in
  let x_shape = s1c 3 and y_shape = s1c 3 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (chan c)) in
  let y = Tensor.materialize y_shape (fun _ -> 10.) in
  Format.printf "%a@." Tensor.pp
    (Schedule.evaluate
       (Pointwise.Mul.output_shape x_shape y_shape)
       (M.pixel x y));
  [%expect {| tensor f32 [C=3] {0, 10, 20} |}]

let%expect_test "Direct: conv2d 2x2 box filter (stride 1, no pad)" =
  let module Cv = Conv.Conv2d.Compute (Direct) in
  (* 3x3 single-channel input, value(h,w) = h*3 + w *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 3) + col c))
  in
  (* Cout=1, 2x2 kernel, Cin=1, all ones -> sum of each 2x2 window *)
  let weight_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let weight = Tensor.materialize weight_shape (fun _ -> 1.) in
  let bias = Tensor.materialize (s1c 1) (fun _ -> 0.) in
  let p =
    {
      Conv.Conv2d.kernel = Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      in_channels = Dim.extent 1;
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
    }
  in
  let out_shape = Conv.Conv2d.output_shape ~x_shape ~weight_shape p in
  Format.printf "%a@." Tensor.pp
    (Schedule.evaluate out_shape (Cv.pixel p ~x_shape ~x ~weight ~bias));
  (* windows: 0+1+3+4=8, 1+2+4+5=12, 3+4+6+7=20, 4+5+7+8=24 *)
  [%expect {| tensor f32 [H=2 W=2 C=1] {8, 12, 20, 24} |}]

let%expect_test
    "Direct: conv2d 3x3 same padding (pad=1) — exercises bound clipping" =
  let module Cv = Conv.Conv2d.Compute (Direct) in
  (* 3x3 single-channel input, value(h,w) = h*3 + w; every output pixel's
     window runs off the edge on at least one side, so this only gets the
     right answer if the clipped kh/kw range — not a guarded read — is what
     keeps the padding region at 0. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 3) + col c))
  in
  let weight_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let weight = Tensor.materialize weight_shape (fun _ -> 1.) in
  let bias = Tensor.materialize (s1c 1) (fun _ -> 0.) in
  let p =
    {
      Conv.Conv2d.kernel = Op_config.Hw.{ h = Dim.extent 3; w = Dim.extent 3 };
      in_channels = Dim.extent 1;
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 1; w = Op_config.Nonneg.of_int 1 };
    }
  in
  let out_shape = Conv.Conv2d.output_shape ~x_shape ~weight_shape p in
  Format.printf "%a@." Tensor.pp
    (Schedule.evaluate out_shape (Cv.pixel p ~x_shape ~x ~weight ~bias));
  (* hand-computed 3x3 zero-padded box sums over [0..8] laid out row-major *)
  [%expect {| tensor f32 [H=3 W=3 C=1] {8, 15, 12, 21, 36, 27, 20, 33, ...} |}]

let%expect_test
    "Direct: max_pool2d 2x2 (stride 1, pad 1) — negative inputs catch a \
     padding=0 bug" =
  let module P = Pool.MaxPool2d.Compute (Direct) in
  (* All-negative input: if the padding region wrongly contributed 0 (as a
     naive guarded-load fallback would), 0 would beat every real value here
     and the result would be wrong everywhere a window touches the border —
     i.e. everywhere, since pad=1 on a 3x3 input with a 2x2 kernel. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c ->
        float_of_int (-((row c * 3) + col c) - 1))
  in
  let p =
    {
      Pool.MaxPool2d.kernel =
        Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 1; w = Op_config.Nonneg.of_int 1 };
    }
  in
  let out_shape = Pool.MaxPool2d.output_shape ~x_shape p in
  Format.printf "%a@." Tensor.pp
    (Schedule.evaluate out_shape (P.pixel p ~x_shape ~x));
  (* hand-computed: input(h,w) = -(h*3+w)-1, i.e. -1..-9 row-major; each
     output position's max over its real (non-padding) taps only *)
  [%expect {| tensor f32 [H=4 W=4 C=1] {-1, -1, -2, -3, -1, -1, -2, -3, ...} |}]

let%expect_test "Direct: avg_pool2d 2x2 (stride 1, pad 0) — box-filter average"
    =
  let module P = Pool.AvgPool2d.Compute (Direct) in
  (* Same 3x3 input/kernel as the conv2d box-filter test, whose hand-computed
     window sums were {8,12,20,24}; dividing by the kernel area (4) is the
     cross-check here. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 3) + col c))
  in
  let p =
    {
      Pool.AvgPool2d.kernel =
        Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
    }
  in
  let out_shape = Pool.AvgPool2d.output_shape ~x_shape p in
  Format.printf "%a@." Tensor.pp
    (Schedule.evaluate out_shape (P.pixel p ~x_shape ~x));
  [%expect {| tensor f32 [H=2 W=2 C=1] {2, 3, 5, 6} |}]

let%expect_test
    "Direct: avg_pool2d 2x2 (stride 1, pad 1) — count_include_pad=true divides \
     by the full kernel area" =
  (* A constant-1 2x2 input: every real tap contributes 1, so the result at
     each output position is exactly (# real taps in its window) / 4 — this
     is what makes count_include_pad=true visible (a count_include_pad=false
     pooling would give 1 everywhere instead). *)
  let module P = Pool.AvgPool2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x = Tensor.materialize x_shape (fun _ -> 1.) in
  let p =
    {
      Pool.AvgPool2d.kernel =
        Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 1; w = Op_config.Nonneg.of_int 1 };
    }
  in
  let out_shape = Pool.AvgPool2d.output_shape ~x_shape p in
  Format.printf "%a@." Tensor.pp
    (Schedule.evaluate out_shape (P.pixel p ~x_shape ~x));
  [%expect
    {| tensor f32 [H=3 W=3 C=1] {0.25, 0.5, 0.25, 0.5, 1, 0.5, 0.25, 0.5, ...} |}]

let%expect_test "Direct: linear (addmm) — out_features mix in_features" =
  let module L = Linear.Linear.Compute (Direct) in
  let chan c = Dim.to_int (Vec6.get c Axis.C) in
  let x_shape = s1c 3 in
  let weight_shape = Vec6.shape ~n:2 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3 in
  let bias_shape = s1c 2 in
  let x = Tensor.materialize x_shape (fun c -> [| 1.; 2.; 3. |].(chan c)) in
  (* weight[0,:] selects x0; weight[1,:] sums x1+x2 *)
  let weight =
    Tensor.materialize weight_shape (fun c ->
        match (Dim.to_int (Vec6.get c Axis.N), chan c) with
        | 0, 0 -> 1.
        | 1, 1 | 1, 2 -> 1.
        | _ -> 0.)
  in
  let bias =
    Tensor.materialize bias_shape (fun c -> [| 10.; 100. |].(chan c))
  in
  let p = { Linear.Linear.in_features = Dim.extent 3 } in
  let out_shape = Linear.Linear.output_shape ~x_shape ~weight_shape in
  Format.printf "%a@." Tensor.pp
    (Schedule.evaluate out_shape (L.pixel p ~x ~weight ~bias));
  [%expect {| tensor f32 [C=2] {11, 105} |}]

let%expect_test
    "Direct: bmm — B=2 batch matrix multiply, exercises batch axis isolation" =
  let module B = Matmul.Bmm.Compute (Direct) in
  (* input[H=2, W=2, C=3]: two 2×3 matrices laid out H=batch, W=row, C=inner.
     batch 0: [[1,2,3],[4,5,6]]   batch 1: [[1,0,0],[0,1,0]] *)
  let input_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:3 in
  let mat0 = [| [| 1.; 2.; 3. |]; [| 4.; 5.; 6. |] |] in
  let mat1 = [| [| 1.; 0.; 0. |]; [| 0.; 1.; 0. |] |] in
  let input =
    Tensor.materialize input_shape (fun c ->
        let b = Dim.to_int (Vec6.get c Axis.H) in
        let r = Dim.to_int (Vec6.get c Axis.W) in
        let k = Dim.to_int (Vec6.get c Axis.C) in
        [| mat0; mat1 |].(b).(r).(k))
  in
  (* mat2[H=2, W=3, C=2]: two 3×2 matrices laid out H=batch, W=inner, C=col.
     batch 0: [[1,0],[0,1],[1,1]]   batch 1: [[2,0],[0,2],[0,0]] *)
  let mat2_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:2 in
  let m2_0 = [| [| 1.; 0. |]; [| 0.; 1. |]; [| 1.; 1. |] |] in
  let m2_1 = [| [| 2.; 0. |]; [| 0.; 2. |]; [| 0.; 0. |] |] in
  let mat2 =
    Tensor.materialize mat2_shape (fun c ->
        let b = Dim.to_int (Vec6.get c Axis.H) in
        let k = Dim.to_int (Vec6.get c Axis.W) in
        let j = Dim.to_int (Vec6.get c Axis.C) in
        [| m2_0; m2_1 |].(b).(k).(j))
  in
  let out_shape = Matmul.Bmm.output_shape ~input_shape ~mat2_shape in
  Format.printf "%a@." Tensor.pp
    (Schedule.evaluate out_shape (B.pixel ~input_shape ~input ~mat2));
  (* batch 0: [1,2,3]·[[1,0],[0,1],[1,1]] = [4,5]; [4,5,6]·same = [10,11]
     batch 1: [1,0,0]·[[2,0],[0,2],[0,0]] = [2,0]; [0,1,0]·same = [0,2] *)
  [%expect {| tensor f32 [H=2 W=2 C=2] {4, 5, 10, 11, 2, 0, 0, 2} |}]

let%expect_test "windowed axis: output_extent and window agree" =
  let module Wa = Window_axis.Compute (Direct) in
  (* The property that's actually guaranteed (not "the window becomes empty
     right at output_extent" — it can still have a stray valid tap one
     position past, depending on config, so that's not asserted here): for
     every out in [0, output_extent), the window names a genuinely non-empty
     range of real (non-padding) taps. If output_extent and the window
     disagreed about the windowed-axis relationship, some output position
     would have an empty window and this would fail. *)
  let check ~in_extent ~kernel ~stride ~pad =
    let kernel = Dim.extent kernel
    and stride = Op_config.Pos.of_int stride
    and pad = Op_config.Nonneg.of_int pad
    and in_extent = Dim.extent in_extent in
    let out_extent =
      Window_axis.output_extent ~kernel ~stride ~pad ~in_extent
    in
    let non_empty out =
      let w = Wa.window ~kernel ~stride ~pad ~in_extent out in
      w.lo < w.hi
    in
    let all_in_range =
      List.for_all non_empty (List.init (out_extent :> int) (fun i -> i))
    in
    Format.printf
      "in_extent=%d kernel=%d stride=%d pad=%d -> output_extent=%d, \
       all_in_range=%b@."
      (in_extent :> int)
      (kernel :> int)
      (stride :> int)
      (pad :> int)
      (out_extent :> int)
      all_in_range
  in
  check ~in_extent:3 ~kernel:2 ~stride:1 ~pad:0;
  check ~in_extent:3 ~kernel:3 ~stride:1 ~pad:1;
  check ~in_extent:7 ~kernel:3 ~stride:2 ~pad:1;
  check ~in_extent:5 ~kernel:1 ~stride:1 ~pad:0;
  [%expect
    {|
    in_extent=3 kernel=2 stride=1 pad=0 -> output_extent=2, all_in_range=true
    in_extent=3 kernel=3 stride=1 pad=1 -> output_extent=3, all_in_range=true
    in_extent=7 kernel=3 stride=2 pad=1 -> output_extent=4, all_in_range=true
    in_extent=5 kernel=1 stride=1 pad=0 -> output_extent=5, all_in_range=true
    |}]

let%expect_test "Direct: mean over spatial (H,W), per channel" =
  let module M = Reduce.Mean.Compute (Direct) in
  (* C passes through, H/W reduce: channel 0 is [1,2,3,4] (mean 2.5), channel 1
     is 10x that (mean 25). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:2 in
  let x =
    Tensor.materialize x_shape (fun c ->
        let base = [| 1.; 2.; 3.; 4. |].((row c * 2) + col c) in
        if chan c = 0 then base else base *. 10.)
  in
  let p = { Reduce.Mean.dims = [ Axis.H; Axis.W ]; keepdim = true } in
  let out_shape = Reduce.Mean.output_shape ~x_shape p in
  Format.printf "%a@." Tensor.pp
    (Schedule.evaluate out_shape (M.pixel p ~x_shape ~x));
  [%expect {| tensor f32 [C=2] {2.5, 25} |}]

let%expect_test
    "Mean output_shape: keepdim true collapses in place, false shifts" =
  (* input [H6 W7 C8] (rank 3); mean over W. keepdim=true leaves W at 1 in place;
     keepdim=false removes W and re-packs H's data onto W (§1d). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:6 ~w:7 ~c:8 in
  let shape ~keepdim =
    Reduce.Mean.output_shape ~x_shape { Reduce.Mean.dims = [ Axis.W ]; keepdim }
  in
  Format.printf "keepdim=true:  %a@." Vec6.pp_shape (shape ~keepdim:true);
  Format.printf "keepdim=false: %a@." Vec6.pp_shape (shape ~keepdim:false);
  [%expect {|
    keepdim=true:  [H=6 W=1 C=8]
    keepdim=false: [W=6 C=8] |}]

let%expect_test "Direct: mean over W, keepdim=false shifts H's data onto W" =
  let module M = Reduce.Mean.Compute (Direct) in
  (* [H2 W2 C1]; value(h,w): row H0 = [1,3] (mean 2), row H1 = [5,7] (mean 6). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c ->
        [| [| 1.; 3. |]; [| 5.; 7. |] |].(row c).(col c))
  in
  let run ~keepdim =
    let p = { Reduce.Mean.dims = [ Axis.W ]; keepdim } in
    Format.printf "%a@." Tensor.pp
      (Schedule.evaluate
         (Reduce.Mean.output_shape ~x_shape p)
         (M.pixel p ~x_shape ~x))
  in
  run ~keepdim:true;
  run ~keepdim:false;
  (* same means {2, 6}; keepdim=true keeps them on H, keepdim=false moves them to W. *)
  [%expect
    {|
    tensor f32 [H=2 W=1 C=1] {2, 6}
    tensor f32 [W=2 C=1] {2, 6} |}]

let%expect_test "Direct: rms_norm over C (channel-wise normalise)" =
  let module R = Norm.RmsNorm.Compute (Direct) in
  let run ~vals ~w ~eps =
    let x_shape = s1c (List.length vals) in
    let xa = Array.of_list vals and wa = Array.of_list w in
    let x = Tensor.materialize x_shape (fun c -> xa.(chan c)) in
    (* weight carries the normalised (C) shape; here all other axes are 1 so its
       shape coincides with x's. *)
    let weight = Tensor.materialize x_shape (fun c -> wa.(chan c)) in
    let p = { Norm.RmsNorm.dims = [ Axis.C ]; eps } in
    Format.printf "%a@." Tensor.pp
      (Schedule.evaluate
         (Norm.RmsNorm.output_shape ~x_shape)
         (R.pixel p ~x_shape ~x ~weight))
  in
  (* distinct |x| so the mean genuinely averages: mean(1²,7²)=mean(1,49)=25,
     sqrt=5, x/5 -> {0.2, 1.4} (eps=0, weight=1). *)
  run ~vals:[ 1.; 7. ] ~w:[ 1.; 1. ] ~eps:0.;
  (* eps + weight, sign preserved through the square: mean(x²)=4, +12=16,
     sqrt=4, x*0.25*w = {0.5, -1, 1.5, -2}. *)
  run ~vals:[ 2.; -2.; 2.; -2. ] ~w:[ 1.; 2.; 3.; 4. ] ~eps:12.;
  [%expect
    {|
    tensor f32 [C=2] {0.2, 1.4}
    tensor f32 [C=4] {0.5, -1, 1.5, -2} |}]

let%expect_test "Direct: permute [W=2 C=3] — swap W and C gives [W=3 C=2]" =
  let module P = Permute.Permute.Compute (Direct) in
  (* [W=2 C=3] tensor: x[w,c] = w*3 + c, row-major values 0..5. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:3 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((col c * 3) + chan c))
  in
  let perm =
    [
      (Axis.N, Axis.N);
      (Axis.T, Axis.T);
      (Axis.D, Axis.D);
      (Axis.H, Axis.H);
      (Axis.W, Axis.C);
      (Axis.C, Axis.W);
    ]
  in
  let out_shape = Permute.Permute.output_shape ~x_shape perm in
  Format.printf "%a@." Tensor.pp (Schedule.evaluate out_shape (P.pixel perm ~x));
  (* transpose of [[0,1,2],[3,4,5]] is [[0,3],[1,4],[2,5]]: row-major 0,3,1,4,2,5 *)
  [%expect {| tensor f32 [W=3 C=2] {0, 3, 1, 4, 2, 5} |}]

let%expect_test "Direct: permute identity — output equals input" =
  let module P = Permute.Permute.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:4 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int (row c + col c + chan c))
  in
  let perm = List.map (fun a -> (a, a)) Axis.all in
  let out_shape = Permute.Permute.output_shape ~x_shape perm in
  (* Shape unchanged *)
  Format.printf "shape: %a@." Vec6.pp_shape out_shape;
  (* Values unchanged: compare first-8 directly with Tensor.pp *)
  Format.printf "%a@." Tensor.pp (Schedule.evaluate out_shape (P.pixel perm ~x));
  [%expect
    {|
    shape: [H=2 W=3 C=4]
    tensor f32 [H=2 W=3 C=4] {0, 1, 2, 3, 1, 2, 3, 4, ...} |}]

let%expect_test "Direct: permute 3D [H=2 W=3 C=4] — cycle H->W->C->H" =
  let module P = Permute.Permute.Compute (Direct) in
  (* Cycle: output H <- input W, output W <- input C, output C <- input H.
     Input shape [H=2 W=3 C=4] -> output shape [H=3 W=4 C=2]. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:3 ~c:4 in
  let x =
    Tensor.materialize x_shape (fun c ->
        (* unique value per element: h*12 + w*4 + c *)
        float_of_int ((row c * 12) + (col c * 4) + chan c))
  in
  let perm =
    [
      (Axis.N, Axis.N);
      (Axis.T, Axis.T);
      (Axis.D, Axis.D);
      (Axis.H, Axis.W);
      (Axis.W, Axis.C);
      (Axis.C, Axis.H);
    ]
  in
  let out_shape = Permute.Permute.output_shape ~x_shape perm in
  Format.printf "out shape: %a@." Vec6.pp_shape out_shape;
  Format.printf "%a@." Tensor.pp (Schedule.evaluate out_shape (P.pixel perm ~x));
  (* output[h,w,c] = input[c, h, w]  (inverse cycle: C->H->W->C)
     (h=0,w=0,c=0): input[c=0,h=0,w=0]=0; (h=0,w=0,c=1): input[c=1,h=0,w=0]=12
     (h=0,w=1,c=0): input[c=0,h=0,w=1]=1; (h=0,w=1,c=1): input[c=1,h=0,w=1]=13 *)
  [%expect
    {|
    out shape: [H=3 W=4 C=2]
    tensor f32 [H=3 W=4 C=2] {0, 12, 1, 13, 2, 14, 3, 15, ...} |}]
