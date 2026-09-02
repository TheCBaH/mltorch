(* layer_norm and group_norm, hand-computed against values derived
   independently of the [Compute] functor under test. Split from
   compute_test.ml. *)

open Compute_fixtures

(* LayerNorm's arithmetic, against values derived by hand rather than by a
   second run of the same functor. Direct-vs-Symbolic cannot see any of the
   mutations these cover, because both instantiate this same [Compute].

   Every case is chosen so a plausible WRONG implementation prints something
   visibly different:
     - skipping the centring (i.e. RMSNorm) changes every case with a non-zero
       mean, which is all of them;
     - dividing the variance by count - 1 changes every case with count > 2;
     - adding eps outside the sqrt changes the two eps cases;
     - reducing the wrong axis changes the multi-axis case into per-row work;
     - indexing the affine operands by the full output coordinate reads them
       out of range;
     - applying bias before weight changes the affine case's sign pattern.
   A 2-element normalised slice is deliberately NOT used as the main fixture:
   it normalises to +/-1 whatever the divisor is, so it cannot separate them. *)
let%expect_test "Direct: layer_norm over C, centred and scaled" =
  let module L = Norm.LayerNorm.Legacy_pixel (Direct) in
  let ones n = List.init n (fun _ -> 1.)
  and zeros n = List.init n (fun _ -> 0.) in
  let run ?w ?b ~vals ~eps () =
    let n = List.length vals in
    let x_shape = s1c n in
    let arr l = Array.of_list l in
    let xa = arr vals in
    let wa = arr (Option.value w ~default:(ones n)) in
    let ba = arr (Option.value b ~default:(zeros n)) in
    let x = Tensor.materialize x_shape (fun c -> xa.(chan c)) in
    (* Both affine operands carry the normalised (C) shape; every other axis is
       1 here, so their shape coincides with x's. *)
    let weight = Tensor.materialize x_shape (fun c -> wa.(chan c)) in
    let bias = Tensor.materialize x_shape (fun c -> ba.(chan c)) in
    let p = { Norm.LayerNorm.dims = [ Axis.C ]; eps } in
    Format.printf "%a@." (pp_result Tensor.pp)
      (eval_tensor
         (Norm.LayerNorm.output_shape ~x_shape p)
         (L.pixel p ~x_shape ~x ~weight ~bias))
  in
  (* Non-zero mean AND non-unit variance, the two things RMSNorm gets wrong:
     mean(3,3,7,7) = 5, deviations (-2,-2,2,2), var = 16/4 = 4, sqrt = 2,
     so y = (-1,-1,1,1). The unbiased divisor 3 would give var = 16/3,
     sqrt = 2.3094, y = -+0.8660254 instead. *)
  run ~vals:[ 3.; 3.; 7.; 7. ] ~eps:0. ();
  (* The same slice with a NEGATIVE weight and a non-zero bias, and the affine
     step in ATen's order: y * weight + bias.
       (-1*-1)+10 = 11, (-1*2)+20 = 18, (1*-3)+30 = 27, (1*4)+40 = 44.
     Bias before weight would give (y + b) * w = (-9, 38, -81, 164). *)
  run ~vals:[ 3.; 3.; 7.; 7. ] ~w:[ -1.; 2.; -3.; 4. ] ~b:[ 10.; 20.; 30.; 40. ]
    ~eps:0. ();
  (* count = 1: the variance is 0 by construction, so y is 0 whatever eps is
     and only the bias survives. Legal in ATen, so it must be legal here. *)
  run ~vals:[ 7. ] ~w:[ 5. ] ~b:[ 3. ] ~eps:1e-5 ();
  (* A large offset with a small variance. mean = 8388608 = 2^23 and the
     deviations are -+1, both exactly representable in f32, so the answer is
     again (-1,-1,1,1).

     This pins the ANSWER at a large offset; it is not a demonstration that
     E[x^2] - mean^2 breaks, and it cannot be one here -- [Direct] accumulates
     in f64, whose 53-bit mantissa absorbs a 2^46 mean-square with room to
     spare, and no f32-representable input is far enough out to exhaust it. The
     case for two passes is settled against ATen in f32 by the walk, not here. *)
  run ~vals:[ 8388607.; 8388607.; 8388609.; 8388609. ] ~eps:0. ();
  [%expect
    {|
    tensor f32 [C=4] {-1, -1, 1, 1}
    tensor f32 [C=4] {11, 18, 27, 44}
    tensor f32 [C=1] {3}
    tensor f32 [C=4] {-1, -1, 1, 1} |}]

(* Multi-axis: the reduction nests one [sum] per normalised axis, and reducing
   over the wrong subset is a wrong answer that still has the right shape. *)
let%expect_test "Direct: layer_norm over W and C jointly" =
  let module L = Norm.LayerNorm.Legacy_pixel (Direct) in
  (* [W=2 C=2] holding 1, 3, 5, 7 in (w, c) order. Over BOTH axes: mean = 4,
     deviations (-3,-1,1,3), var = 20/4 = 5, sqrt(5) = 2.2360680, so
     y = (-1.3416408, -0.4472136, 0.4472136, 1.3416408).

     Over C alone it would normalise each row separately -- (1,3) and (5,7) --
     and print (-1, 1, -1, 1). Over W alone, (-1, -1, 1, 1). Neither is close
     to the joint answer, which is the point of the fixture. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:2 in
  let vals = [| [| 1.; 3. |]; [| 5.; 7. |] |] in
  let x = Tensor.materialize x_shape (fun c -> vals.(col c).(chan c)) in
  let affine v = Tensor.materialize x_shape (fun _ -> v) in
  let p = { Norm.LayerNorm.dims = [ Axis.W; Axis.C ]; eps = 0. } in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Norm.LayerNorm.output_shape ~x_shape p)
       (L.pixel p ~x_shape ~x ~weight:(affine 1.) ~bias:(affine 0.)));
  [%expect {| tensor f32 [W=2 C=2] {-1.34164, -0.447214, 0.447214, 1.34164} |}]

(* Both epsilon values the corpus contains, on a slice whose variance is the
   same order as eps -- which is the only place the two are distinguishable,
   and also the only place "eps inside the sqrt" is distinguishable from "eps
   outside" it.

   x = (1, 1, 1 + 2^-10, 1 + 2^-10), all exact in f32. mean = 1 + 2^-11,
   deviations -+2^-11, var = 2^-22 = 2.384185791015625e-07. *)
let%expect_test "Direct: layer_norm eps is inside the sqrt" =
  let module L = Norm.LayerNorm.Legacy_pixel (Direct) in
  let run ~eps =
    let x_shape = s1c 4 in
    let hi = 1. +. (2. ** -10.) in
    let xa = [| 1.; 1.; hi; hi |] in
    let x = Tensor.materialize x_shape (fun c -> xa.(chan c)) in
    let affine v = Tensor.materialize x_shape (fun _ -> v) in
    let p = { Norm.LayerNorm.dims = [ Axis.C ]; eps } in
    Format.printf "%a@." (pp_result Tensor.pp)
      (eval_tensor
         (Norm.LayerNorm.output_shape ~x_shape p)
         (L.pixel p ~x_shape ~x ~weight:(affine 1.) ~bias:(affine 0.)))
  in
  run ~eps:1e-6;
  run ~eps:1e-5;
  [%expect
    {|
    tensor f32 [C=4] {-0.438769, -0.438769, 0.438769, 0.438769}
    tensor f32 [C=4] {-0.1526, -0.1526, 0.1526, 0.1526} |}]

(* ---- group_norm -------------------------------------------------------------

   Hand-computed, same attribution discipline as the layer_norm block above.
   Unlike layer_norm, the reduction is over a WINDOW of C (one group's slice)
   plus the FULL extent of every other non-N axis -- so the fixture needs a
   real spatial extent (H > 1) to exercise the "full extent" half at all;
   layer_norm's single-axis-C fixtures would leave it untested. *)
let%expect_test "Direct: group_norm splits C into two groups, reducing H too" =
  let module G = Norm.GroupNorm.Compute (Direct) in
  (* [H=2 W=1 C=4], x[h,c] = h*10+c: group 0 is channels {0,1}, group 1 is
     {2,3}, and each group's window spans both rows of H. Group 0's four
     values {0,1,10,11} have mean 5.5, var 25.25; group 1's {2,3,12,13} have
     mean 7.5, the SAME variance (both groups are shifted copies of one
     another by 2), so a wrong group boundary that swapped which four values
     went together would still print a plausible-looking but different
     result. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:1 ~c:4 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 10) + chan c))
  in
  let ones = Tensor.materialize x_shape (fun _ -> 1.) in
  let zeros = Tensor.materialize x_shape (fun _ -> 0.) in
  let p =
    {
      Norm.GroupNorm.channel = Axis.C;
      groups = Op_config.Pos.of_int 2;
      eps = 0.;
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Norm.GroupNorm.output_shape ~x_shape p)
       (G.pixel p ~x_shape ~x ~weight:ones ~bias:zeros));
  [%expect
    {| tensor f32 [H=2 W=1 C=4] {-1.09454, -0.895533, -1.09454, -0.895533, 0.895533, 1.09454, 0.895533, 1.09454} |}]

(* The per-channel affine, on the trivial-spatial [H=1 W=1] case so the
   reduction itself is not also under test: with two channels per group and
   H=W=1, group 0 is {0,1} and group 1 is {2,3}, each normalising to
   [-1, 1]. weight/bias then apply PER CHANNEL (not per group, unlike
   layer_norm's per-normalised-shape affine) in ATen's [y * weight + bias]
   order. *)
let%expect_test "Direct: group_norm applies weight/bias per channel" =
  let module G = Norm.GroupNorm.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:4 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (chan c)) in
  let wa = [| 10.; 20.; 30.; 40. |] in
  let ba = [| 100.; 200.; 300.; 400. |] in
  let weight = Tensor.materialize x_shape (fun c -> wa.(chan c)) in
  let bias = Tensor.materialize x_shape (fun c -> ba.(chan c)) in
  let p =
    {
      Norm.GroupNorm.channel = Axis.C;
      groups = Op_config.Pos.of_int 2;
      eps = 0.;
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Norm.GroupNorm.output_shape ~x_shape p)
       (G.pixel p ~x_shape ~x ~weight ~bias));
  [%expect {| tensor f32 [C=4] {90, 220, 270, 440} |}]

(* [num_groups] must divide the channel count exactly -- ATen's own contract,
   checked at [output_shape] rather than assumed. *)
let%expect_test "group_norm output_shape: indivisible channel count" =
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:5 in
  let p =
    {
      Norm.GroupNorm.channel = Axis.C;
      groups = Op_config.Pos.of_int 2;
      eps = 0.;
    }
  in
  Format.printf "%a@." (pp_result Vec6.pp_shape)
    (Norm.GroupNorm.output_shape ~x_shape p);
  [%expect {| group_norm: channel count 5 is not divisible by num_groups 2 |}]
