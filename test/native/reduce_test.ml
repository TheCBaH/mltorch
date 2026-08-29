(* mean/amax/vector_norm over spatial axes, rms_norm, and batch_norm's
   per-channel affine. Split from compute_test.ml. *)

open Compute_fixtures

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
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor (Reduce.Mean.output_shape ~x_shape p) (M.pixel p ~x_shape ~x));
  [%expect {| tensor f32 [C=2] {2.5, 25} |}]

let%expect_test
    "Mean output_shape: keepdim true collapses in place, false shifts" =
  (* input [H6 W7 C8] (rank 3); mean over W. keepdim=true leaves W at 1 in place;
     keepdim=false removes W and re-packs H's data onto W (§1d). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:6 ~w:7 ~c:8 in
  let shape ~keepdim =
    Reduce.Mean.output_shape ~x_shape { Reduce.Mean.dims = [ Axis.W ]; keepdim }
    |> Err.or_raise ~pp_error:Shape_error.pp
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
    Format.printf "%a@." (pp_result Tensor.pp)
      (eval_tensor
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

let%expect_test "Direct: amax over spatial (H,W), per channel" =
  let module M = Reduce.Amax.Compute (Direct) in
  (* Same input as the mean fixture above: channel 0 is [1,2,3,4] (max 4),
     channel 1 is 10x that (max 40). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:2 in
  let x =
    Tensor.materialize x_shape (fun c ->
        let base = [| 1.; 2.; 3.; 4. |].((row c * 2) + col c) in
        if chan c = 0 then base else base *. 10.)
  in
  let p = { Reduce.Amax.dims = [ Axis.H; Axis.W ]; keepdim = true } in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor (Reduce.Amax.output_shape ~x_shape p) (M.pixel p ~x_shape ~x));
  [%expect {| tensor f32 [C=2] {4, 40} |}]

let%expect_test "Direct: amax over W, keepdim=false shifts H's data onto W" =
  let module M = Reduce.Amax.Compute (Direct) in
  (* [H2 W2 C1]; value(h,w): row H0 = [1,3] (max 3), row H1 = [5,7] (max 7). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c ->
        [| [| 1.; 3. |]; [| 5.; 7. |] |].(row c).(col c))
  in
  let run ~keepdim =
    let p = { Reduce.Amax.dims = [ Axis.W ]; keepdim } in
    Format.printf "%a@." (pp_result Tensor.pp)
      (eval_tensor
         (Reduce.Amax.output_shape ~x_shape p)
         (M.pixel p ~x_shape ~x))
  in
  run ~keepdim:true;
  run ~keepdim:false;
  (* same maxes {3, 7}; keepdim=true keeps them on H, keepdim=false moves them to W. *)
  [%expect
    {|
    tensor f32 [H=2 W=1 C=1] {3, 7}
    tensor f32 [W=2 C=1] {3, 7} |}]

let%expect_test "Direct: vector_norm over spatial (H,W), per channel" =
  let module M = Reduce.Vector_norm.Compute (Direct) in
  (* Same input as the mean/amax fixtures above: channel 0's L2 norm over
     [1,2,3,4] is sqrt(30), channel 1's over 10x that is sqrt(3000). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:2 in
  let x =
    Tensor.materialize x_shape (fun c ->
        let base = [| 1.; 2.; 3.; 4. |].((row c * 2) + col c) in
        if chan c = 0 then base else base *. 10.)
  in
  let p = { Reduce.Vector_norm.dims = [ Axis.H; Axis.W ]; keepdim = true } in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Reduce.Vector_norm.output_shape ~x_shape p)
       (M.pixel p ~x_shape ~x));
  [%expect {| tensor f32 [C=2] {5.47723, 54.7723} |}]

let%expect_test
    "Direct: vector_norm over W, keepdim=false shifts H's data onto W" =
  let module M = Reduce.Vector_norm.Compute (Direct) in
  (* [H2 W2 C1]; value(h,w): row H0 = [1,3] (norm sqrt(10)), row H1 = [5,7]
     (norm sqrt(74)). *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c ->
        [| [| 1.; 3. |]; [| 5.; 7. |] |].(row c).(col c))
  in
  let run ~keepdim =
    let p = { Reduce.Vector_norm.dims = [ Axis.W ]; keepdim } in
    Format.printf "%a@." (pp_result Tensor.pp)
      (eval_tensor
         (Reduce.Vector_norm.output_shape ~x_shape p)
         (M.pixel p ~x_shape ~x))
  in
  run ~keepdim:true;
  run ~keepdim:false;
  (* same norms {sqrt(10), sqrt(74)}; keepdim=true keeps them on H, keepdim=false
     moves them to W. *)
  [%expect
    {|
    tensor f32 [H=2 W=1 C=1] {3.16228, 8.60233}
    tensor f32 [W=2 C=1] {3.16228, 8.60233} |}]

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
    Format.printf "%a@." (pp_result Tensor.pp)
      (eval_tensor
         (Norm.RmsNorm.output_shape ~x_shape p)
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

let%expect_test
    "Direct: batch_norm per-channel affine over C (broadcast over H)" =
  let module B = Norm.BatchNorm.Compute (Direct) in
  (* [H=2 C=2]: x[h,c] with c0 = [1,3] over h, c1 = [5,7]. The per-channel
     running stats (mean=[1,5], var=[4,4]) apply across the H axis. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:2 in
  let x =
    Tensor.materialize x_shape (fun c ->
        [| [| 1.; 3. |]; [| 5.; 7. |] |].(chan c).(row c))
  in
  let vec2 a0 a1 =
    Tensor.materialize (s1c 2) (fun c -> [| a0; a1 |].(chan c))
  in
  let p = { Norm.BatchNorm.channel = Axis.C; eps = 0. } in
  let run ~w0 ~w1 ~b0 ~b1 =
    Format.printf "%a@." (pp_result Tensor.pp)
      (eval_tensor
         (Norm.BatchNorm.output_shape ~x_shape)
         (B.pixel p ~x ~weight:(vec2 w0 w1) ~bias:(vec2 b0 b1)
            ~running_mean:(vec2 1. 5.) ~running_var:(vec2 4. 4.)))
  in
  (* mean=[1,5], 1/sqrt(4)=0.5, weight=1 bias=0: (x-mean)*0.5 per channel. *)
  run ~w0:1. ~w1:1. ~b0:0. ~b1:0.;
  (* per-channel weight/bias: c0 -> *2 + 1, c1 -> *10 - 1. *)
  run ~w0:2. ~w1:10. ~b0:1. ~b1:(-1.);
  [%expect
    {|
    tensor f32 [H=2 W=1 C=2] {0, 0, 1, 1}
    tensor f32 [H=2 W=1 C=2] {1, -1, 3, 9} |}]
