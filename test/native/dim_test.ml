let%expect_test "dim: non-negative construction + guard + coercion" =
  Format.printf "%a@." Dim.pp (Dim.extent 224);
  [%expect {| 224 |}];
  let rejects f =
    match f () with exception Invalid_argument _ -> true | _ -> false
  in
  Format.printf "%b %b@."
    (rejects (fun () -> Dim.index (-1)))
    (rejects (fun () -> Dim.extent (-7)));
  [%expect {| true true |}];
  (* a signed delta becomes an index only inside the extent: 3 in [0,5), -1 not *)
  let g d = Dim.index_of ~extent:(Dim.extent 5) (Dim.delta d) in
  Format.printf "%b %b %b@." (g 3 <> None) (g (-1) = None) (g 5 = None);
  [%expect {| true true true |}]

let%expect_test "dim: numel fold + horner offset" =
  let open Dim in
  let numel = one_count *@ extent 2 *@ extent 3 *@ extent 4 in
  (* dense offset of (i0=1, i1=2) under extents (3, 4): (0*3+1)*4+2 = 6 *)
  let off = lin (lin zero_offset (extent 3) (index 1)) (extent 4) (index 2) in
  Format.printf "numel=%a off=%a raw=%d@." pp numel pp off (off :> int);
  [%expect {| numel=24 off=6 raw=6 |}]
