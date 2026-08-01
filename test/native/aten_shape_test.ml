(* Right-aligned, positional ATen<->6D bridge. See .ai/native_tensor_design.md §1d. *)

let pp_ints fmt a =
  Format.fprintf fmt "[%s]"
    (String.concat ";" (Array.to_list (Array.map string_of_int a)))

(* [of_aten] now returns a result; these cases use only valid shapes. *)
let of_aten a = Aten_shape.of_aten a |> Core.or_raise Aten_shape.pp_error

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
