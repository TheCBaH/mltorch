let%expect_test "vec6: offset, numel, pp" =
  let s = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:3 in
  let c = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:1 ~w:0 ~c:2 in
  Format.printf "%a @@ %a -> off %a, numel %a@." Vec6.pp_shape s Vec6.pp_coord c
    Dim.pp (Vec6.offset s c) Dim.pp (Vec6.numel s);
  [%expect {| [H=2 W=2 C=3] @ (1,0,2) -> off 8, numel 12 |}]

let%expect_test "vec6: broadcast read_coord (extent-1 axes -> 0)" =
  let bias = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3 in
  let c = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:5 ~w:9 ~c:2 in
  Format.printf "%a@." Vec6.pp_coord (Vec6.read_coord bias c);
  [%expect {| (2) |}]

let%expect_test "vec6: iter visits dense offsets 0..numel-1 (C innermost)" =
  let s = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:2 in
  Vec6.iter s (fun c -> Format.printf "%d " (Vec6.offset s c :> int));
  Format.printf "@.";
  [%expect {| 0 1 2 3 4 5 6 7 |}]
