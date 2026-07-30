(* [Axis4] and [Shape4]: that the dialect's invariant is a type property rather
   than a check someone has to remember to call. *)

open Native4d

let pp_shape_result =
  Core.Pretty.core_result ~ok:Shape4.pp ~error:(Shape4.pp_error :> _ Fmt.t)

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c

(* ---- Axis4 ---------------------------------------------------------------- *)

(* [of_axis] is partial, and that partiality is the design: a caller converting a
   Native axis has to say what it does with the two the dialect cannot name. *)
let%expect_test "axis4: T and D have no four-axis name" =
  List.iter
    (fun axis ->
      Format.printf "%a -> %a@." Axis.pp axis
        (Core.Pretty.option_or ~none:"outside the dialect" Axis4.pp)
        (Axis4.of_axis axis))
    Axis.all;
  [%expect
    {|
    N -> N
    T -> outside the dialect
    D -> outside the dialect
    H -> H
    W -> W
    C -> C |}]

(* [all] is in FRAME order, not the alphabetical order the constructors are
   declared in — every consumer iterating axes cares about the frame. *)
let%expect_test "axis4: all is in frame order" =
  Format.printf "%a@." (Fmt.list ~sep:(Fmt.any " ") Axis4.pp) Axis4.all;
  [%expect {| N H W C |}]

(* ---- Shape4 --------------------------------------------------------------- *)

let%expect_test "shape4: what of_vec6 accepts" =
  List.iter
    (fun (label, shape) ->
      Format.printf "%-22s %a@." label pp_shape_result (Shape4.of_vec6 shape))
    [
      ("four-axis", s 2 1 1 4 4 3);
      ("all unit", s 1 1 1 1 1 1);
      ("non-unit T", s 1 2 1 4 4 3);
      ("non-unit D", s 1 1 2 4 4 3);
      ("both", s 1 2 2 4 4 3);
    ];
  [%expect
    {|
    four-axis              [N=2 H=4 W=4 C=3]
    all unit               [N=1 H=1 W=1 C=1]
    non-unit T             shape has extent on T or D: [T=2 D=1 H=4 W=4 C=3]
    non-unit D             shape has extent on T or D: [D=2 H=4 W=4 C=3]
    both                   shape has extent on T or D: [T=2 D=2 H=4 W=4 C=3] |}]

(* [Shape4.pp] deliberately does not trim leading 1s the way [Vec6.pp_shape]
   does: a test reading the output can tell which type produced it. *)
let%expect_test "shape4: printing is distinguishable from Vec6" =
  let v = s 1 1 1 1 1 8 in
  let s4 = Shape4.of_vec6 v |> Result.get_ok in
  Format.printf "Vec6:   %a@." Vec6.pp_shape v;
  Format.printf "Shape4: %a@." Shape4.pp s4;
  [%expect {|
    Vec6:   [C=8]
    Shape4: [N=1 H=1 W=1 C=8] |}]

(* [set] is total, unlike a [Vec6.set] on T or D — [Axis4.t] cannot name those,
   so no sequence of sets can take a shape out of the dialect. *)
let%expect_test "shape4: set cannot leave the dialect" =
  let s4 = Shape4.of_ints ~n:1 ~h:4 ~w:4 ~c:3 in
  let moved =
    List.fold_left
      (fun acc axis -> Shape4.set acc axis (Dim.extent 7))
      s4 Axis4.all
  in
  Format.printf "%a -> %a@." Shape4.pp s4 Shape4.pp moved;
  Format.printf "still four-axis: %a@." Vec6.pp_shape (Shape4.to_vec6 moved);
  [%expect
    {|
    [N=1 H=4 W=4 C=3] -> [N=7 H=7 W=7 C=7]
    still four-axis: [N=7 T=1 D=1 H=7 W=7 C=7] |}]

let%expect_test "shape4: round-trips through JSON" =
  let s4 = Shape4.of_ints ~n:2 ~h:3 ~w:5 ~c:7 in
  let json = Json_util.enc Shape4.jsont s4 in
  let back = Json_util.dec Shape4.jsont json in
  Format.printf "equal: %b (%a)@." (Shape4.equal s4 back) Shape4.pp back;
  [%expect {| equal: true ([N=2 H=3 W=5 C=7]) |}]

(* A six-axis shape outside the dialect must not decode into one. Encoding
   cannot produce such a document, but a hand-written or older one can. *)
let%expect_test "shape4: decoding rejects a non-four-axis document" =
  let json = Json_util.enc Vec6.shape_jsont (s 1 1 9 4 4 3) in
  Format.printf "%s@."
    (match Json_util.dec Shape4.jsont json with
    | exception Jsont.Error _ -> "rejected"
    | s4 -> Format.asprintf "accepted %a" Shape4.pp s4);
  [%expect {| rejected |}]
