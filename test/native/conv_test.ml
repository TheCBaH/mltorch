(* conv2d.default: box filter, grouped/depthwise reduction, asymmetric
   padding+dilation, and the Same-padding odd-split rule. Split from
   compute_test.ml. *)

open Compute_fixtures

let conv_axis ?(pad_before = 0) ?pad_after ?(dilation = 1) ~kernel ~stride () :
    Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent kernel;
    stride = Op_config.Pos.of_int stride;
    pad_before = Op_config.Nonneg.of_int pad_before;
    pad_after =
      Op_config.Nonneg.of_int (Option.value pad_after ~default:pad_before);
    dilation = Op_config.Pos.of_int dilation;
  }

let conv_params ?(groups = 1) ?(dilation = (1, 1)) ?(pad_after = None)
    ?(pad = (0, 0)) ~kernel ~stride ~in_channels () =
  let kh, kw = kernel
  and sh, sw = stride
  and ph, pw = pad
  and dh, dw = dilation in
  let pah, paw =
    match pad_after with None -> (None, None) | Some (h, w) -> (Some h, Some w)
  in
  {
    Conv.Conv2d.h =
      conv_axis ?pad_after:pah ~pad_before:ph ~dilation:dh ~kernel:kh ~stride:sh
        ();
    w =
      conv_axis ?pad_after:paw ~pad_before:pw ~dilation:dw ~kernel:kw ~stride:sw
        ();
    in_channels = Dim.extent in_channels;
    groups = Op_config.Pos.of_int groups;
  }

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
  let p = conv_params ~kernel:(2, 2) ~stride:(1, 1) ~in_channels:1 () in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Conv.Conv2d.output_shape ~x_shape ~weight_shape p)
       (Cv.pixel p ~x_shape ~weight_shape ~x ~weight ~bias));
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
    conv_params ~kernel:(3, 3) ~stride:(1, 1) ~pad:(1, 1) ~in_channels:1 ()
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Conv.Conv2d.output_shape ~x_shape ~weight_shape p)
       (Cv.pixel p ~x_shape ~weight_shape ~x ~weight ~bias));
  (* hand-computed 3x3 zero-padded box sums over [0..8] laid out row-major *)
  [%expect {| tensor f32 [H=3 W=3 C=1] {8, 15, 12, 21, 36, 27, 20, 33, ...} |}]

let%expect_test "Direct: grouped conv2d reduces only within each channel group"
    =
  let module Cv = Conv.Conv2d.Compute (Direct) in
  let x_shape = s1c 4 in
  let x =
    Tensor.materialize x_shape (fun c -> [| 1.; 2.; 10.; 20. |].(chan c))
  in
  let weight_shape = Vec6.shape ~n:4 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let weight =
    Tensor.materialize weight_shape (fun c ->
        match (Dim.to_int (Vec6.get c Axis.N), chan c) with
        | 0, 0 | 0, 1 -> 1.
        | 1, 0 -> 10.
        | 2, 0 | 2, 1 -> 1.
        | 3, 1 -> 2.
        | _ -> 0.)
  in
  let bias =
    Tensor.materialize (s1c 4) (fun c -> [| 0.; 100.; 1000.; 10000. |].(chan c))
  in
  let p =
    conv_params ~groups:2 ~kernel:(1, 1) ~stride:(1, 1) ~in_channels:4 ()
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Conv.Conv2d.output_shape ~x_shape ~weight_shape p)
       (Cv.pixel p ~x_shape ~weight_shape ~x ~weight ~bias));
  [%expect {| tensor f32 [C=4] {3, 110, 1030, 10040} |}]

let%expect_test
    "Direct: depthwise conv2d as grouped conv with channel multiplier" =
  let module Cv = Conv.Conv2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:2 in
  let x =
    Tensor.materialize x_shape (fun c ->
        match (col c, chan c) with
        | 0, 0 -> 1.
        | 0, 1 -> 10.
        | 1, 0 -> 2.
        | 1, 1 -> 20.
        | _ -> assert false)
  in
  let weight_shape = Vec6.shape ~n:4 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let weight =
    Tensor.materialize weight_shape (fun c ->
        [| 1.; 2.; 3.; 4. |].(Dim.to_int (Vec6.get c Axis.N)))
  in
  let bias = Tensor.materialize (s1c 4) (fun _ -> 0.) in
  let p =
    conv_params ~groups:2 ~kernel:(1, 1) ~stride:(1, 1) ~in_channels:2 ()
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Conv.Conv2d.output_shape ~x_shape ~weight_shape p)
       (Cv.pixel p ~x_shape ~weight_shape ~x ~weight ~bias));
  [%expect {| tensor f32 [W=2 C=4] {1, 2, 30, 40, 2, 4, 60, 80} |}]

let%expect_test "Direct: conv2d asymmetric padding with dilation" =
  let module Cv = Conv.Conv2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:5 ~c:1 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (col c)) in
  let weight_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:3 ~c:1 in
  let weight = Tensor.materialize weight_shape (fun _ -> 1.) in
  let bias = Tensor.materialize (s1c 1) (fun _ -> 0.) in
  let p =
    conv_params ~dilation:(1, 2) ~pad:(0, 1)
      ~pad_after:(Some (0, 2))
      ~kernel:(1, 3) ~stride:(1, 1) ~in_channels:1 ()
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Conv.Conv2d.output_shape ~x_shape ~weight_shape p)
       (Cv.pixel p ~x_shape ~weight_shape ~x ~weight ~bias));
  [%expect {| tensor f32 [W=4 C=1] {4, 6, 4, 6} |}]

(* [Conv1d] delegates its whole arithmetic to [Conv2d] with H pinned to
   [Conv2d.unit_window] (conv_conv1d.ml) — this pins the case that
   delegation could get wrong: a kernel window that only ever moves along W,
   never H. *)
let%expect_test "Direct: conv1d delegates to Conv2d with H pinned unit" =
  let module Cv = Conv.Conv1d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:5 ~c:1 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (col c)) in
  let weight_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:3 ~c:1 in
  let weight = Tensor.materialize weight_shape (fun _ -> 1.) in
  let bias = Tensor.materialize (s1c 1) (fun _ -> 0.) in
  let p =
    {
      Conv.Conv1d.w = conv_axis ~kernel:3 ~stride:1 ();
      in_channels = Dim.extent 1;
      groups = Op_config.Pos.of_int 1;
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Conv.Conv1d.output_shape ~x_shape ~weight_shape p)
       (Cv.pixel p ~x_shape ~weight_shape ~x ~weight ~bias));
  (* windows over [0,1,2,3,4], kernel 3: 0+1+2=3, 1+2+3=6, 2+3+4=9 *)
  [%expect {| tensor f32 [W=3 C=1] {3, 6, 9} |}]

let%expect_test "Direct: grouped conv1d reduces only within each channel group"
    =
  let module Cv = Conv.Conv1d.Compute (Direct) in
  let x_shape = s1c 4 in
  let x =
    Tensor.materialize x_shape (fun c -> [| 1.; 2.; 10.; 20. |].(chan c))
  in
  let weight_shape = Vec6.shape ~n:4 ~t:1 ~d:1 ~h:1 ~w:1 ~c:2 in
  let weight =
    Tensor.materialize weight_shape (fun c ->
        match (Dim.to_int (Vec6.get c Axis.N), chan c) with
        | 0, 0 | 0, 1 -> 1.
        | 1, 0 -> 10.
        | 2, 0 | 2, 1 -> 1.
        | 3, 1 -> 2.
        | _ -> 0.)
  in
  let bias =
    Tensor.materialize (s1c 4) (fun c -> [| 0.; 100.; 1000.; 10000. |].(chan c))
  in
  let p =
    {
      Conv.Conv1d.w = conv_axis ~kernel:1 ~stride:1 ();
      in_channels = Dim.extent 4;
      groups = Op_config.Pos.of_int 2;
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Conv.Conv1d.output_shape ~x_shape ~weight_shape p)
       (Cv.pixel p ~x_shape ~weight_shape ~x ~weight ~bias));
  [%expect {| tensor f32 [C=4] {3, 110, 1030, 10040} |}]

(* [Conv2d_padding.same_padding] splits an ODD total unevenly — [total / 2]
   before and [total - total / 2] after — and nothing else pins which side gets
   the extra cell. Direct-vs-Symbolic agreement cannot: both resolve through the
   same function, so they agree on a reversed split as readily as on the right
   one. Neither can an output-shape check: reversing the split leaves the extent
   untouched. Nor can the ATen oracle, which reaches [constant_pad_nd] for this
   case and is not in this repository's minimal static-dispatch build.

   So the numbers here are hand-computed, and the fixture is built so they
   differ. A 1x2 kernel over a 1x3 row gives total = 1: pad_before = 0,
   pad_after = 1. With the weight [0; 1] each output reads the cell to its
   RIGHT, so y = [x1; x2; pad] = {2, 3, 0}. Reverse the split and every output
   reads itself instead, giving {1, 2, 3}. *)
let%expect_test "Direct: conv2d_padding same splits an odd total to the right" =
  let module Cv = Conv.Conv2d_padding.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:3 ~c:1 in
  let x = Tensor.materialize x_shape (fun c -> float_of_int (col c + 1)) in
  let weight_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:1 in
  let weight =
    Tensor.materialize weight_shape (fun c -> if col c = 0 then 0. else 1.)
  in
  let bias = Tensor.materialize (s1c 1) (fun _ -> 0.) in
  let p =
    {
      Conv.Conv2d_padding.stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      padding = Conv.Conv2d_padding.Same;
      dilation =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      groups = Op_config.Pos.of_int 1;
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Conv.Conv2d_padding.output_shape ~x_shape ~weight_shape p)
       (Cv.pixel p ~x_shape ~weight_shape ~x ~weight ~bias));
  [%expect {| tensor f32 [W=3 C=1] {2, 3, 0} |}]
