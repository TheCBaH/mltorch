(* max_pool2d, max_pool2d_with_indices (including the signed-zero/NaN tie
   pinning against real f32 bit patterns), avg_pool2d, adaptive_avg_pool2d,
   and the shared Window_axis output_extent/window agreement property. Split
   from compute_test.ml. *)

open Compute_fixtures

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
      Pool.MaxPool2d.ceil_mode = false;
      kernel = Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 1; w = Op_config.Nonneg.of_int 1 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.MaxPool2d.output_shape ~x_shape p)
       (P.pixel p ~x_shape ~x));
  (* hand-computed: input(h,w) = -(h*3+w)-1, i.e. -1..-9 row-major; each
     output position's max over its real (non-padding) taps only *)
  [%expect {| tensor f32 [H=4 W=4 C=1] {-1, -1, -2, -3, -1, -1, -2, -3, ...} |}]

let%expect_test "Direct: max_pool2d_with_indices 2x2 stride 2 — values + argmax"
    =
  let module M = Pool.MaxPool2dWithIndices.Compute (Direct) in
  (* 4x4 input, value(h,w) = h*4 + w (0..15). Each 2x2 window's max is its
     bottom-right corner; the argmax flat index is ih*4 + iw of that corner. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:4 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 4) + col c))
  in
  let p =
    {
      Pool.MaxPool2d.ceil_mode = false;
      kernel = Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
    }
  in
  Format.printf "values:  %a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.MaxPool2dWithIndices.output_shape ~x_shape p)
       (M.value_pixel p ~x_shape ~x));
  Format.printf "indices: %a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.MaxPool2dWithIndices.output_shape ~x_shape p)
       (M.index_pixel p ~x_shape ~x));
  [%expect
    {|
    values:  tensor f32 [H=2 W=2 C=1] {5, 7, 13, 15}
    indices: tensor f32 [H=2 W=2 C=1] {5, 7, 13, 15} |}]

let%expect_test
    "Direct: max_pool2d_with_indices ties choose smallest flat index" =
  let module M = Pool.MaxPool2dWithIndices.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x = Tensor.materialize x_shape (fun _ -> 7.) in
  let p =
    {
      Pool.MaxPool2d.ceil_mode = false;
      kernel = Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.MaxPool2dWithIndices.output_shape ~x_shape p)
       (M.index_pixel p ~x_shape ~x));
  [%expect {| tensor f32 [C=1] {0} |}]

(* ---- max-pool signed zero / NaN, against ATen -----------------------------

   [Tensor.pp] renders -0. as "-0" but every NaN as "nan", so it cannot show
   WHICH NaN survived — and these tests turn on exactly those bits. Print the
   f32 storage pattern instead. *)

let bits_of tensor =
  let (Tensor.Tensor t) = tensor in
  Vec6.fold_coords t.Tensor.shape ~init:[] ~f:(fun acc c ->
      Printf.sprintf "%08lx" (Int32.bits_of_float (Tensor.read tensor c)) :: acc)
  |> List.rev |> String.concat " "

let pp_bits ppf tensor = Format.pp_print_string ppf (bits_of tensor)

(* A 1xN window, so one output pixel reduces the whole row. *)
let row_pool n =
  {
    Pool.MaxPool2d.ceil_mode = false;
    kernel = Op_config.Hw.{ h = Dim.extent 1; w = Dim.extent n };
    stride =
      Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int n };
    pad =
      Op_config.Hw.
        { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
  }

let row_input values =
  let n = Array.length values in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:n ~c:1 in
  (x_shape, Tensor.materialize x_shape (fun c -> values.(col c)))

(* Two NaNs the same class but distinct payloads: identical payloads would make
   "the last NaN wins" unobservable. *)
let nan1 = Int32.float_of_bits 0x7FC00001l
let nan2 = Int32.float_of_bits 0x7FC00002l

let%expect_test "Direct: max_pool2d keeps the incumbent on a signed-zero tie" =
  (* ATen selects on [(val > maxval) || isnan(val)] (MaxPoolKernel.cpp:215), and
     [-0. > +0.] and [+0. > -0.] are both false, so a signed-zero tie keeps
     whichever came first. [Float.max] would answer +0. either way, which is the
     bug this pins. *)
  let module P = Pool.MaxPool2d.Compute (Direct) in
  List.iter
    (fun (name, values) ->
      let x_shape, x = row_input values in
      let p = row_pool 2 in
      Format.printf "%s: %a@." name (pp_result pp_bits)
        (eval_tensor
           (Pool.MaxPool2d.output_shape ~x_shape p)
           (P.pixel p ~x_shape ~x)))
    [ ("-0 then +0", [| -0.; 0. |]); ("+0 then -0", [| 0.; -0. |]) ];
  [%expect {|
    -0 then +0: 80000000
    +0 then -0: 00000000 |}]

let%expect_test
    "Direct: max_pool2d_with_indices — the last NaN wins, with its index" =
  (* Every NaN re-satisfies the predicate, so the last one in iteration order
     ends up in both the value and the index. Before [Max_op] the value
     propagated a NaN (via [Float.max]) while the index did not (strict [>]),
     so the two disagreed about which element won. *)
  let module M = Pool.MaxPool2dWithIndices.Compute (Direct) in
  let x_shape, x = row_input [| nan1; 5.; nan2; 7. |] in
  let p = row_pool 4 in
  let shape = Pool.MaxPool2dWithIndices.output_shape ~x_shape p in
  Format.printf "value: %a@." (pp_result pp_bits)
    (eval_tensor shape (M.value_pixel p ~x_shape ~x));
  Format.printf "index: %a@." (pp_result Tensor.pp)
    (eval_tensor shape (M.index_pixel p ~x_shape ~x));
  [%expect {|
    value: 7fc00002
    index: tensor f32 [C=1] {2} |}]

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
      Pool.AvgPool2d.ceil_mode = false;
      count_include_pad = true;
      kernel = Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.AvgPool2d.output_shape ~x_shape p)
       (P.pixel p ~x_shape ~x));
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
      Pool.AvgPool2d.ceil_mode = false;
      count_include_pad = true;
      kernel = Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 1; w = Op_config.Nonneg.of_int 1 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.AvgPool2d.output_shape ~x_shape p)
       (P.pixel p ~x_shape ~x));
  [%expect
    {| tensor f32 [H=3 W=3 C=1] {0.25, 0.5, 0.25, 0.5, 1, 0.5, 0.25, 0.5, ...} |}]

let%expect_test
    "Direct: avg_pool2d 2x2 (stride 1, pad 1) — count_include_pad=false \
     divides by the real (non-padding) window area" =
  (* Same constant-1 input as the count_include_pad=true test above, same
     window geometry -- every real tap still contributes 1, so this should be
     1 everywhere, unlike the padded divisor's <1 corners/edges. *)
  let module P = Pool.AvgPool2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x = Tensor.materialize x_shape (fun _ -> 1.) in
  let p =
    {
      Pool.AvgPool2d.ceil_mode = false;
      count_include_pad = false;
      kernel = Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 1; w = Op_config.Nonneg.of_int 1 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.AvgPool2d.output_shape ~x_shape p)
       (P.pixel p ~x_shape ~x));
  [%expect {| tensor f32 [H=3 W=3 C=1] {1, 1, 1, 1, 1, 1, 1, 1, ...} |}]

let%expect_test
    "Direct: avg_pool2d 2x2 (stride 2, pad 0, ceil_mode=true) — the trailing \
     window's count_include_pad=true divisor shrinks below the kernel area" =
  (* 3x3 input, kernel 2, stride 2, no padding: floor mode gives one output
     row/col ({0,1},{0,1}); ceil_mode admits a second, output position 1,
     whose window only has ONE real+padded column/row left (a kernel of 2 but
     only 1 column remaining) -- so its divisor is 1*1, not 2*2, even though
     count_include_pad=true. All values are 1, so a wrong (too-large) divisor
     would show up directly as a wrong (too-small) average. *)
  let module P = Pool.AvgPool2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let x = Tensor.materialize x_shape (fun _ -> 1.) in
  let p =
    {
      Pool.AvgPool2d.ceil_mode = true;
      count_include_pad = true;
      kernel = Op_config.Hw.{ h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        Op_config.Hw.{ h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
      pad =
        Op_config.Hw.
          { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.AvgPool2d.output_shape ~x_shape p)
       (P.pixel p ~x_shape ~x));
  [%expect {| tensor f32 [H=2 W=2 C=1] {1, 1, 1, 1} |}]

let%expect_test
    "Direct: adaptive_avg_pool2d uses ATen's overlapping non-divisible bins" =
  let module P = Pool.AdaptiveAvgPool2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:5 ~w:5 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 5) + col c + 1))
  in
  let p =
    {
      Pool.AdaptiveAvgPool2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 3; w = Op_config.Pos.of_int 3 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.AdaptiveAvgPool2d.output_shape ~x_shape p)
       (P.pixel p ~x_shape ~x));
  [%expect
    {| tensor f32 [H=3 W=3 C=1] {4, 5.5, 7, 11.5, 13, 14.5, 19, 20.5, ...} |}]

let%expect_test "Direct: adaptive_avg_pool2d global [1,1]" =
  let module P = Pool.AdaptiveAvgPool2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:4 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 4) + col c + 1))
  in
  let p =
    {
      Pool.AdaptiveAvgPool2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.AdaptiveAvgPool2d.output_shape ~x_shape p)
       (P.pixel p ~x_shape ~x));
  [%expect {| tensor f32 [C=1] {8.5} |}]

let%expect_test
    "Direct: adaptive_max_pool2d_with_indices uses ATen's overlapping \
     non-divisible bins — values + argmax" =
  (* Same 5x5 input/3x3 output as [adaptive_avg_pool2d]'s own "overlapping
     non-divisible bins" test above, so the bin ranges are already verified by
     that test's hand-computed averages: bin0=[0,2), bin1=[1,4), bin2=[3,5) on
     each axis. Values increase monotonically in both H and W, so each bin's
     max is always its bottom-right corner, and the argmax is that corner's
     flat index [ih*5+iw]. *)
  let module M = Pool.AdaptiveMaxPool2dWithIndices.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:5 ~w:5 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((row c * 5) + col c + 1))
  in
  let p =
    {
      Pool.AdaptiveMaxPool2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 3; w = Op_config.Pos.of_int 3 };
    }
  in
  let shape = Pool.AdaptiveMaxPool2dWithIndices.output_shape ~x_shape p in
  Format.printf "values:  %a@." (pp_result Tensor.pp)
    (eval_tensor shape (M.value_pixel p ~x_shape ~x));
  Format.printf "indices: %a@." (pp_result Tensor.pp)
    (eval_tensor shape (M.index_pixel p ~x_shape ~x));
  [%expect
    {|
    values:  tensor f32 [H=3 W=3 C=1] {7, 9, 10, 17, 19, 20, 22, 24, ...}
    indices: tensor f32 [H=3 W=3 C=1] {6, 8, 9, 16, 18, 19, 21, 23, ...} |}]

let%expect_test
    "Direct: adaptive_max_pool2d_with_indices ties choose smallest flat index" =
  let module M = Pool.AdaptiveMaxPool2dWithIndices.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x = Tensor.materialize x_shape (fun _ -> 7.) in
  let p =
    {
      Pool.AdaptiveMaxPool2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Pool.AdaptiveMaxPool2dWithIndices.output_shape ~x_shape p)
       (M.index_pixel p ~x_shape ~x));
  [%expect {| tensor f32 [C=1] {0} |}]

let%expect_test
    "adaptive_avg_pool2d rejects an index-scale aggregate before allocation" =
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:(1 lsl 30) ~w:1 ~c:1 in
  let p =
    {
      Pool.AdaptiveAvgPool2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 1 };
    }
  in
  Format.printf "%a@." (pp_result Vec6.pp_shape)
    (Pool.AdaptiveAvgPool2d.output_shape ~x_shape p);
  [%expect
    {| adaptive_avg_pool2d: input extent 1073741824 times output_size 2 on axis H is 2147483648, which must be below the engine maximum of 2147483648 |}]

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
      Window_axis.output_extent ~ceil_mode:false ~kernel ~stride ~pad_before:pad
        ~pad_after:pad ~dilation:(Op_config.Pos.of_int 1) ~in_extent
      |> Err.or_raise ~pp_error:Shape_error.pp
    in
    let non_empty out =
      let w =
        Wa.window ~kernel ~stride ~pad_before:pad
          ~dilation:(Op_config.Pos.of_int 1) ~in_extent out
      in
      (w.lo :> int) < (w.hi :> int)
    in
    let all_in_range =
      List.for_all non_empty
        (List.init (out_extent :> int) (fun i -> Dim.index i))
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
