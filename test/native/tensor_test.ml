let%expect_test "tensor: materialize + pp" =
  let t =
    Tensor.materialize (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3) (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.C)))
  in
  Format.printf "%a@." Tensor.pp t;
  [%expect {| tensor f32 [C=3] {0, 1, 2} |}]

let%expect_test "tensor: read is strict — an in-bounds index reads, OOB raises"
    =
  let t =
    Tensor.materialize (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3) (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.C)))
  in
  (* the only valid spatial coord is 0 on every extent-1 axis *)
  Format.printf "%g@."
    (Tensor.read t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:2));
  [%expect {| 2 |}];
  (* an index past an extent-1 axis is out of bounds — [load] does NOT broadcast *)
  let oob () = Tensor.read t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:5 ~w:9 ~c:2) in
  (match oob () with
  | _ -> print_string "no error"
  | exception Invalid_argument msg -> Printf.printf "raised: %s\n" msg);
  [%expect
    {| raised: Tensor.read: coord (5,9,2) out of bounds for shape [C=3] |}];
  (* likewise a channel index at/over the extent *)
  (match Tensor.read t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:3) with
  | _ -> print_string "no error"
  | exception Invalid_argument msg -> Printf.printf "raised: %s\n" msg);
  [%expect {| raised: Tensor.read: coord (3) out of bounds for shape [C=3] |}]

let%expect_test "tensor: read_at raises on an out-of-bounds index" =
  let t =
    Tensor.materialize (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3) (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.C)))
  in
  let idx (a : Axis.t) = match a with Axis.C -> 5 | _ -> 0 in
  (match Tensor.read_at t idx with
  | _ -> print_string "no error"
  | exception Invalid_argument msg -> Printf.printf "raised: %s\n" msg);
  [%expect {| raised: Tensor.read: coord (5) out of bounds for shape [C=3] |}]

let%expect_test "tensor: shift_in_bounds guards the pad region" =
  (* a 1x1x1x3x1x1 column over H; tap relative to base h=0 *)
  let t =
    Tensor.materialize (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:1 ~c:1) (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.H)))
  in
  let base = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0 in
  let probe d =
    let deltas = Vec6.deltas ~n:0 ~t:0 ~d:0 ~h:d ~w:0 ~c:0 in
    match Tensor.shift_in_bounds t base deltas with
    | Some c -> Printf.sprintf "%g" (Tensor.read t c)
    | None -> "pad"
  in
  Format.printf "%s %s %s %s@." (probe (-1)) (probe 0) (probe 2) (probe 3);
  [%expect {| pad 0 2 pad |}]
