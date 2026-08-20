(* Right-aligned, positional ATen<->6D bridge. See .ai/native_tensor_design.md §1d. *)

let pp_ints fmt a =
  Format.fprintf fmt "[%s]"
    (String.concat ";" (Array.to_list (Array.map string_of_int a)))

(* [of_aten] now returns a result; these cases use only valid shapes. *)
let of_aten a =
  Aten_shape.of_aten a |> Err.or_raise ~pp_error:Aten_shape.pp_error

let%expect_test "of_aten: right-aligned into innermost axes" =
  Format.printf "%a@." Vec6.pp_shape (of_aten [| 6; 7; 8 |]);
  [%expect {| [H=6 W=7 C=8] |}];
  (* rank 4 [n;h;w;c]: n lands on D, not N (positional, not semantic). *)
  Format.printf "%a@." Vec6.pp_shape (of_aten [| 2; 3; 4; 5 |]);
  [%expect {| [D=2 H=3 W=4 C=5] |}];
  Format.printf "%a@." Vec6.pp_shape (of_aten [| 9 |]);
  [%expect {| [C=9] |}]

let%expect_test "to_aten: reverse needs the rank, round-trips of_aten" =
  let s = of_aten [| 6; 7; 8 |] in
  Format.printf "%a@." pp_ints (Aten_shape.to_aten ~rank:3 s);
  [%expect {| [6;7;8] |}];
  (* same frame shape read at a different rank yields a different ATen shape. *)
  Format.printf "%a@." pp_ints (Aten_shape.to_aten ~rank:2 s);
  [%expect {| [7;8] |}];
  List.iter
    (fun a ->
      assert (Aten_shape.to_aten ~rank:(Array.length a) (of_aten a) = a))
    [ [||]; [| 5 |]; [| 6; 7; 8 |]; [| 2; 3; 4; 5; 6; 7 |] ]

let%expect_test "axis_of_dim: positional, negative dims count from the end" =
  let show ~rank dim =
    Format.printf "%a " Axis.pp (Aten_shape.axis_of_dim ~rank dim)
  in
  show ~rank:3 0;
  show ~rank:3 1;
  show ~rank:3 2;
  show ~rank:3 (-1);
  show ~rank:4 0;
  Format.printf "@.";
  [%expect {| H W C C D |}]

let%expect_test "used_axes: innermost rank axes" =
  let show rank =
    Format.printf "%d:%s@." rank
      (String.concat "" (List.map Axis.to_string (Aten_shape.used_axes ~rank)))
  in
  List.iter show [ 0; 1; 2; 4; 6 ];
  [%expect {|
    0:
    1:C
    2:WC
    4:DHWC
    6:NTDHWC |}]

(* ---- resolve_slice: the canonical bounds both importers will share -------- *)

(* PyTorch's contract, restated as a table so each rule is falsifiable on its
   own. Read the output as "start,stop,step -> out": [out] is the extent the
   shape rule will derive, printed here so a floor/ceiling mistake is visible in
   the same place the bounds are.

   [out] is computed by this TEST, not by the function under test -- the
   function deliberately does not decide emptiness (that is the shape rule's
   job, so a graph built through [Graph_builder] meets it too). The column is
   here to show which rows produce nothing. *)
let slice ~extent ?start ?stop ?(step = 1) () =
  match
    Aten_shape.resolve_slice ~extent:(Dim.extent extent) ~start ~stop ~step
  with
  | Error e ->
      Format.printf "error: %a@." Aten_shape.pp_error (Err.Error.kind e)
  | Ok ({ start; stop; step } as b) ->
      let step = (step :> int) in
      let out = (stop - start + step - 1) / step in
      Format.printf "%a -> out %d@." Aten_shape.Slice_bounds.pp b out

let%expect_test "resolve_slice: absent bounds select the whole axis" =
  slice ~extent:8 ();
  slice ~extent:8 ~start:0 ();
  slice ~extent:8 ~stop:8 ();
  [%expect
    {|
    [0, 8) step 1 -> out 8
    [0, 8) step 1 -> out 8
    [0, 8) step 1 -> out 8 |}]

let%expect_test "resolve_slice: negative bounds count from the end" =
  slice ~extent:8 ~start:(-3) ();
  slice ~extent:8 ~stop:(-1) ();
  slice ~extent:8 ~start:(-6) ~stop:(-2) ();
  (* -8 is exactly the first element; -9 is past it and clamps to 0, which is
     what ATen does rather than refusing. *)
  slice ~extent:8 ~start:(-8) ();
  slice ~extent:8 ~start:(-9) ();
  [%expect
    {|
    [5, 8) step 1 -> out 3
    [0, 7) step 1 -> out 7
    [2, 6) step 1 -> out 4
    [0, 8) step 1 -> out 8
    [0, 8) step 1 -> out 8 |}]

let%expect_test "resolve_slice: out-of-range bounds clamp, they do not refuse" =
  (* A trailing bound far past the axis is how exporters spell "to the end". *)
  slice ~extent:8 ~stop:1000 ();
  slice ~extent:8 ~start:1000 ();
  (* An inverted interval collapses to empty at [start], never to a negative
     count -- the case a subtraction alone would get wrong. *)
  slice ~extent:8 ~start:6 ~stop:2 ();
  [%expect
    {|
    [0, 8) step 1 -> out 8
    [8, 8) step 1 -> out 0
    [6, 6) step 1 -> out 0 |}]

let%expect_test "resolve_slice: step > 1 rounds the count UP" =
  (* 7 elements by 2 is 4, not 3: the last index (6) is selected. A floor would
     read 3 here and 3 on the exact case below, which is why both are present --
     an exact division cannot distinguish floor from ceiling. *)
  slice ~extent:8 ~start:0 ~stop:7 ~step:2 ();
  slice ~extent:8 ~start:0 ~stop:8 ~step:2 ();
  slice ~extent:8 ~start:1 ~stop:8 ~step:3 ();
  slice ~extent:8 ~step:8 ();
  [%expect
    {|
    [0, 7) step 2 -> out 4
    [0, 8) step 2 -> out 4
    [1, 8) step 3 -> out 3
    [0, 8) step 8 -> out 1 |}]

let%expect_test "resolve_slice: a non-positive step is the one refusal" =
  slice ~extent:8 ~step:0 ();
  slice ~extent:8 ~step:(-1) ();
  (* Refused BEFORE the bounds are touched: a negative start that would
     otherwise normalize is irrelevant to the answer. *)
  slice ~extent:8 ~start:(-3) ~step:0 ();
  [%expect
    {|
    error: slice step must be >= 1, got 0
    error: slice step must be >= 1, got -1
    error: slice step must be >= 1, got 0 |}]
