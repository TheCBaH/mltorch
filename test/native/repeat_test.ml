(* repeat.default: tile a tensor by an integer multiplier along every axis,
   wrapping each axis independently modulo its own source extent. Split from
   compute_test.ml, alongside reshape_test.ml -- [Repeat] shares its "mod x d
   = x - d*(x/d)" per-axis idiom with [Reshape.delinearize], without
   [Reshape]'s flatten-then-redistribute, since every axis keeps its own
   identity here. *)

open Compute_fixtures

let repeat_eval ~x_shape ~x (p : Repeat.Repeat.params) =
  let module R = Repeat.Repeat.Compute (Direct) in
  let open Err.Syntax in
  let r =
    let* out_shape = Repeat.Repeat.output_shape ~x_shape p in
    Err.return (out_shape, Schedule.evaluate out_shape (R.pixel ~x_shape x))
  in
  Format.printf "@[<v>%a@]@."
    (pp_result (fun ppf (shape, t) -> pp_grid shape ppf t))
    r

let%expect_test
    "Direct: repeat [H=2 W=3] by [H:2 W:2] wraps each axis independently" =
  let x_shape, x = coord_hw 2 3 in
  (* Row h reads source row (h mod 2); each row repeats its own 0,1,2 twice
     across W (w mod 3) -- a wrong axis pairing (e.g. reading H's repeat
     count against W) would show up as a misaligned row or a non-repeating
     tail. *)
  repeat_eval ~x_shape ~x
    { Repeat.Repeat.repeats = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 };
  [%expect
    {|
    [H=4 W=6 C=1]
    0 1 2 0 1 2
    10 11 12 10 11 12
    0 1 2 0 1 2
    10 11 12 10 11 12 |}]

let%expect_test "Direct: repeat by 1 on every axis is the identity" =
  let x_shape, x = coord_hw 2 3 in
  repeat_eval ~x_shape ~x
    { Repeat.Repeat.repeats = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 };
  [%expect {|
    [H=2 W=3 C=1]
    0 1 2
    10 11 12 |}]

(* repeat_interleave.self_int: duplicate each element [repeats] times along
   one named axis, contiguously -- the opposite composition from [Repeat]'s
   wraparound tiling. *)

let repeat_interleave_eval ~x_shape ~x (p : Repeat.RepeatInterleave.params) =
  let module R = Repeat.RepeatInterleave.Compute (Direct) in
  let open Err.Syntax in
  let r =
    let* out_shape = Repeat.RepeatInterleave.output_shape ~x_shape p in
    Err.return (out_shape, Schedule.evaluate out_shape (R.pixel p x))
  in
  Format.printf "@[<v>%a@]@."
    (pp_result (fun ppf (shape, t) -> pp_grid shape ppf t))
    r

let%expect_test
    "Direct: repeat_interleave [H=2 W=3] by 2 on H duplicates each row" =
  let x_shape, x = coord_hw 2 3 in
  (* Row h_out reads source row (h_out / 2): rows 0,1 both read source row 0;
     rows 2,3 both read source row 1 -- CONTIGUOUS duplication, unlike
     [Repeat]'s wraparound. *)
  repeat_interleave_eval ~x_shape ~x
    { Repeat.RepeatInterleave.axis = Axis.H; repeats = Op_config.Pos.of_int 2 };
  [%expect
    {|
    [H=4 W=3 C=1]
    0 1 2
    0 1 2
    10 11 12
    10 11 12 |}]

let%expect_test
    "Direct: repeat_interleave [H=2 W=3] by 2 on W duplicates each column" =
  let x_shape, x = coord_hw 2 3 in
  repeat_interleave_eval ~x_shape ~x
    { Repeat.RepeatInterleave.axis = Axis.W; repeats = Op_config.Pos.of_int 2 };
  [%expect {|
    [H=2 W=6 C=1]
    0 0 1 1 2 2
    10 10 11 11 12 12 |}]

let%expect_test "Direct: repeat_interleave by 1 is the identity" =
  let x_shape, x = coord_hw 2 3 in
  repeat_interleave_eval ~x_shape ~x
    { Repeat.RepeatInterleave.axis = Axis.H; repeats = Op_config.Pos.of_int 1 };
  [%expect {|
    [H=2 W=3 C=1]
    0 1 2
    10 11 12 |}]
