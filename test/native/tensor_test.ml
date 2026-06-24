let%expect_test "tensor: materialize + pp" =
  let t =
    Tensor.materialize (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3) (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.C)))
  in
  Format.printf "%a@." Tensor.pp t;
  [%expect {| tensor f32 [N=1 T=1 D=1 H=1 W=1 C=3] {0, 1, 2} |}]

let%expect_test "tensor: broadcast read of a [..,C] source" =
  let t =
    Tensor.materialize (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3) (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.C)))
  in
  (* far-away spatial coord still reads channel 2 (extent-1 axes broadcast) *)
  Format.printf "%g@."
    (Tensor.read t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:5 ~w:9 ~c:2));
  [%expect {| 2 |}]

let%expect_test "tensor: shift_in_bounds guards the pad region" =
  (* a 1x1x1x3x1x1 column over H; tap relative to base h=0 *)
  let t =
    Tensor.materialize (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:1 ~c:1) (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.H)))
  in
  let base = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0 in
  let probe d =
    match Tensor.shift_in_bounds t base [ (Axis.H, Dim.delta d) ] with
    | Some c -> Printf.sprintf "%g" (Tensor.read t c)
    | None -> "pad"
  in
  Format.printf "%s %s %s %s@." (probe (-1)) (probe 0) (probe 2) (probe 3);
  [%expect {| pad 0 2 pad |}]
