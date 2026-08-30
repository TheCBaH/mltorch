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
  let idx (a : Axis.t) =
    match a with Axis.C -> Dim.index 5 | _ -> Dim.index 0
  in
  (match Tensor.read_at t idx with
  | _ -> print_string "no error"
  | exception Invalid_argument msg -> Printf.printf "raised: %s\n" msg);
  [%expect {| raised: Tensor.read: coord (5) out of bounds for shape [C=3] |}]

(* [read_i64_at6]: the exact single-cell [I64] accessor `index.Tensor`'s
   gather value reads through. Round 15's split test contract — this is the
   half that verifies "the [2^53] test": reading a large [int64] value back
   out returns it EXACTLY, with no precision loss at the storage-read
   boundary. Bounds-checking against a gather extent is
   [Eval.resolve_gather_index]'s job (test/expr/index_test.ml), not this
   one's. *)
let%expect_test
    "tensor: read_i64_at6 reads I64 exactly, including 2^53-scale and Int64 \
     extrema" =
  let t =
    Tensor.materialize_i64 (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3) (fun c ->
        match Dim.to_int (Vec6.get c Axis.C) with
        | 0 -> Int64.shift_left 1L 53
        | 1 -> Int64.max_int
        | _ -> Int64.min_int)
  in
  let idx c (a : Axis.t) = match a with Axis.C -> c | _ -> 0 in
  let show c =
    match Tensor.read_i64_at6 t (idx c) with
    | Ok v -> Fmt.pr "%d -> %Ld@." c v
    | Error _ -> Fmt.pr "%d -> error@." c
  in
  show 0;
  show 1;
  show 2;
  [%expect
    {|
    0 -> 9007199254740992
    1 -> 9223372036854775807
    2 -> -9223372036854775808
    |}]

let%expect_test "tensor: read_i64_at6 typed-rejects a non-I64 format" =
  let t =
    Tensor.materialize (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1) (fun _ -> 1.0)
  in
  (match Tensor.read_i64_at6 t (fun _ -> 0) with
  | Ok _ -> print_string "ok"
  | Error e -> (
      match Err.Error.kind e with
      | `Wrong_format (Payload.Fmt fmt) ->
          Fmt.pr "wrong format: %a@." Payload.pp_fmt fmt));
  [%expect {| wrong format: f32 |}]

let%expect_test "tensor: shift_in_bounds guards the pad region" =
  (* a 1x1x1x3x1x1 column over H; tap relative to base h=0 *)
  let t =
    Tensor.materialize (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:1 ~c:1) (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.H)))
  in
  let base = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0 in
  let probe d =
    let deltas = Vec6.deltas ~n:0 ~t:0 ~d:0 ~h:d ~w:0 ~c:0 in
    Tensor.shift_in_bounds t base deltas
    |> Option.fold ~none:"pad" ~some:(fun c ->
        Printf.sprintf "%g" (Tensor.read t c))
  in
  Format.printf "%s %s %s %s@." (probe (-1)) (probe 0) (probe 2) (probe 3);
  [%expect {| pad 0 2 pad |}]
