(* upsample_bilinear2d, both align_corners values; upsample_nearest2d; plus
   each op's own index-scale overflow rejection. Split from compute_test.ml. *)

open Compute_fixtures

let%expect_test "Direct: upsample_bilinear2d align_corners=true, 2x2 -> 3x3" =
  let module R = Resize.Bilinear2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  (* v(0,0)=0 v(0,1)=2 / v(1,0)=4 v(1,1)=6 *)
  let x =
    Tensor.materialize x_shape (fun c ->
        float_of_int ((4 * row c) + (2 * col c)))
  in
  let p =
    {
      Resize.Bilinear2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 3; w = Op_config.Pos.of_int 3 };
      align_corners = true;
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Resize.Bilinear2d.output_shape ~x_shape p)
       (R.pixel p ~x_shape ~x));
  (* Hand-derived: scale = (2-1)/(3-1) = 0.5, so rows/cols 0,1,2 read source
     0, 0.5, 1 -- corners are exact, the center averages all four inputs. *)
  [%expect {| tensor f32 [H=3 W=3 C=1] {0, 1, 2, 2, 3, 4, 4, 5, ...} |}]

let%expect_test "Direct: upsample_bilinear2d align_corners=false, 2x2 -> 4x4" =
  let module R = Resize.Bilinear2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c ->
        float_of_int ((4 * row c) + (2 * col c)))
  in
  let p =
    {
      Resize.Bilinear2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 4; w = Op_config.Pos.of_int 4 };
      align_corners = false;
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Resize.Bilinear2d.output_shape ~x_shape p)
       (R.pixel p ~x_shape ~x));
  [%expect {| tensor f32 [H=4 W=4 C=1] {0, 0.5, 1.5, 2, 1, 1.5, 2.5, 3, ...} |}]

let%expect_test "Direct: upsample_bilinear2d align_corners=true, out_extent=1" =
  let module R = Resize.Bilinear2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:4 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((4 * row c) + col c + 1))
  in
  let p =
    {
      Resize.Bilinear2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      align_corners = true;
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Resize.Bilinear2d.output_shape ~x_shape p)
       (R.pixel p ~x_shape ~x));
  (* out_extent=1 always reads source index 0 on that axis (ATen's own
     scale-0 rule), regardless of the input extent. *)
  [%expect {| tensor f32 [C=1] {1} |}]

let%expect_test
    "upsample_bilinear2d rejects an index-scale aggregate before allocation" =
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:(1 lsl 30) ~w:1 ~c:1 in
  let p =
    {
      Resize.Bilinear2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 1 };
      align_corners = true;
    }
  in
  Format.printf "%a@." (pp_result Vec6.pp_shape)
    (Resize.Bilinear2d.output_shape ~x_shape p);
  [%expect
    {| upsample_bilinear2d: (input extent 1073741824 - 1) times (output_size 2 - 1) on axis H is 4294967296, which must be below the engine maximum of 2147483648 |}]

let%expect_test
    "Direct: upsample_nearest2d, 2x2 -> 4x4 -- every tap is a copy, no blend" =
  let module R = Resize.Nearest2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  (* v(0,0)=0 v(0,1)=2 / v(1,0)=4 v(1,1)=6 *)
  let x =
    Tensor.materialize x_shape (fun c ->
        float_of_int ((4 * row c) + (2 * col c)))
  in
  let p =
    {
      Resize.Nearest2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 4; w = Op_config.Pos.of_int 4 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Resize.Nearest2d.output_shape ~x_shape p)
       (R.pixel p ~x_shape ~x));
  (* Hand-derived: src = floor(out_idx * 2 / 4), so rows/cols 0,1,2,3 all read
     source row/col 0,0,1,1 -- every 2x2 quadrant is a constant block, unlike
     bilinear's smooth ramp above. *)
  [%expect {| tensor f32 [H=4 W=4 C=1] {0, 0, 2, 2, 0, 0, 2, 2, ...} |}]

let%expect_test
    "Direct: upsample_nearest2d, 3x3 -> 2x2 -- downsampling drops taps rather \
     than averaging them" =
  let module R = Resize.Nearest2d.Compute (Direct) in
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:1 in
  let x =
    Tensor.materialize x_shape (fun c -> float_of_int ((3 * row c) + col c))
  in
  let p =
    {
      Resize.Nearest2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
    }
  in
  Format.printf "%a@." (pp_result Tensor.pp)
    (eval_tensor
       (Resize.Nearest2d.output_shape ~x_shape p)
       (R.pixel p ~x_shape ~x));
  (* src = floor(out_idx * 3 / 2): out 0 -> 0, out 1 -> 1 -- row/col 2 of the
     3x3 input (value 6, 7, 8) is never read on either axis. *)
  [%expect {| tensor f32 [H=2 W=2 C=1] {0, 1, 3, 4} |}]

let%expect_test
    "upsample_nearest2d rejects an index-scale aggregate before allocation" =
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:(1 lsl 30) ~w:1 ~c:1 in
  let p =
    {
      Resize.Nearest2d.output_size =
        Op_config.Hw.{ h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 1 };
    }
  in
  Format.printf "%a@." (pp_result Vec6.pp_shape)
    (Resize.Nearest2d.output_shape ~x_shape p);
  [%expect
    {| upsample_nearest2d: input extent 1073741824 times output_size 2 on axis H is 2147483648, which must be below the engine maximum of 2147483648 |}]
