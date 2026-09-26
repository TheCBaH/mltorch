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

let extent = Dim.extent
let idx = Dim.index
let fence = Dim.fence

let show_opt = function
  | Some e -> Format.asprintf "%a" Dim.pp e
  | None -> "none"

let%expect_test "dim: fence and span" =
  let s a b = show_opt (Dim.span (fence a) (fence b)) in
  Printf.printf "%s %s %s %s\n" (s 2 5) (s 0 1) (s 3 3) (s 4 2);
  [%expect {| 3 1 none none |}];
  (* a fence of an index / an extent is the same number *)
  let f = Dim.fence_of_extent (extent 7) in
  Format.printf "%a %a@." Dim.pp f Dim.pp (Dim.fence_of_index (idx 0));
  [%expect {| 7 0 |}];
  (match Dim.fence (-1) with
  | exception Invalid_argument _ -> print_endline "rejected"
  | _ -> print_endline "accepted");
  [%expect {| rejected |}]

let%expect_test "dim: div_exact, divides, unlin" =
  Printf.printf "%s %s %b %b\n"
    (show_opt (Dim.div_exact (extent 12) ~by:(extent 4)))
    (show_opt (Dim.div_exact (extent 12) ~by:(extent 5)))
    (Dim.divides ~by:(extent 3) (extent 9))
    (Dim.divides ~by:(extent 2) (extent 9));
  [%expect {| 3 none true false |}];
  (* unlin is the dual of lin *)
  let o =
    Dim.lin (Dim.lin Dim.zero_offset (extent 3) (idx 1)) (extent 4) (idx 2)
  in
  let o', i = Dim.unlin o (extent 4) in
  Printf.printf "%d %d\n" (o' :> int) (i :> int);
  [%expect {| 1 2 |}]

let%expect_test "dim: Delta" =
  let d = Dim.delta in
  let n x = (x : Dim.delta Dim.t :> int) in
  let f = Dim.Delta.floor_div_pos and c = Dim.Delta.ceil_div_pos in
  let by = extent 3 in
  Printf.printf "add=%d neg=%d min=%d max=%d scale=%d of_extent=%d\n"
    (n (Dim.Delta.add (d 2) (d (-5))))
    (n (Dim.Delta.neg (d 4)))
    (n (Dim.Delta.min (d 2) (d (-1))))
    (n (Dim.Delta.max (d 2) (d (-1))))
    (n (Dim.Delta.scale (-3) (d 4)))
    (n (Dim.Delta.of_extent (extent 9)));
  [%expect {| add=-3 neg=-4 min=-1 max=2 scale=-12 of_extent=9 |}];
  (* floor rounds toward -inf, ceil toward +inf, for either sign *)
  Printf.printf "floor: %d %d %d %d | ceil: %d %d %d %d\n"
    (n (f (d 7) ~by))
    (n (f (d (-7)) ~by))
    (n (f (d (-6)) ~by))
    (n (f (d 0) ~by))
    (n (c (d 7) ~by))
    (n (c (d (-7)) ~by))
    (n (c (d (-6)) ~by))
    (n (c (d 0) ~by));
  [%expect {| floor: 2 -3 -2 0 | ceil: 3 -2 -2 0 |}]

let%expect_test "dim: to_int64" =
  Printf.printf "%Ld %Ld\n" (Dim.to_int64 (extent 5)) (Dim.to_int64 (idx 0));
  [%expect {| 5 0 |}]

(* Three extents that each pass every per-axis limit but whose product, 2^33,
   wraps a 32-bit [int] to 0 under js_of_ocaml. A plain [*] would report a
   product of 0 there and let it through; [product_bounded] divides each factor
   into the ceiling first, so the same answer holds on both backends. *)
let%expect_test "dim: product_bounded rejects what a wrapped product accepts" =
  let limit = Int64.shift_left 1L 31 in
  let show r =
    match Err.payload r with
    | Ok (e : Dim.extent Dim.t) -> Printf.sprintf "ok %d" (e :> int)
    | Error (`Product_over_limit w) ->
        Format.asprintf "over: %a" Dim.Product_witness.pp w
  in
  let p es = show (Dim.product_bounded ~limit (List.map extent es)) in
  print_endline (p [ 2048; 2048; 2048 ]);
  [%expect {| over: 4194304 * 2048 does not fit below 2147483648 |}];
  (* limit is exclusive: 2^31 is refused, 2^31 - 1 fits *)
  print_endline (p [ 65536; 32768 ]);
  print_endline (p [ 46341; 46341 ]);
  print_endline (p []);
  print_endline (p [ 3; 5; 7 ]);
  [%expect
    {|
    over: 65536 * 32768 does not fit below 2147483648
    over: 46341 * 46341 does not fit below 2147483648
    ok 1
    ok 105 |}];
  let big =
    Dim.product_bounded ~limit:(Int64.pred limit) [ extent 32767; extent 65537 ]
  in
  print_endline (show big);
  [%expect {| ok 2147450879 |}]

let%expect_test "dim_arith: typed factors" =
  let pos = Op_config.Pos.of_int in
  let n (x : Dim.delta Dim.t) = (x :> int) in
  Printf.printf "%d %d %d\n"
    (n (Dim_arith.Delta.scale ~by:(pos 3) (Dim.delta (-4))))
    (n (Dim_arith.Delta.floor_div_pos (Dim.delta (-7)) ~by:(pos 3)))
    (n (Dim_arith.Delta.ceil_div_pos (Dim.delta (-7)) ~by:(pos 3)));
  [%expect {| -12 -3 -2 |}];
  let limit = Int64.shift_left 1L 31 in
  let show r =
    match Err.payload r with
    | Ok (e : Dim.extent Dim.t) -> Printf.sprintf "ok %d" (e :> int)
    | Error (`Product_over_limit w) ->
        Format.asprintf "over: %a" Dim.Product_witness.pp w
  in
  (* 2^20 channels x 2^12 groups: 2^32 wraps a 32-bit int to 0 *)
  print_endline
    (show (Dim_arith.Extent.scale ~limit ~by:(pos 4096) (extent 1048576)));
  print_endline (show (Dim_arith.Extent.scale ~limit ~by:(pos 4) (extent 16)));
  [%expect
    {|
    over: 1048576 * 4096 does not fit below 2147483648
    ok 64 |}]

let%expect_test "dim: window composition and wrap" =
  let start = fence 3 in
  Format.printf "%a %a@." Dim.pp
    (Dim.advance ~start (idx 2))
    Dim.pp
    (Dim.fence_after start (extent 4));
  [%expect {| 5 7 |}];
  (* window [3, 7): 2 and 7 fall outside, 3 and 6 are its ends *)
  let local i = show_opt (Dim.local_in ~start ~extent:(extent 4) (idx i)) in
  Printf.printf "%s %s %s %s\n" (local 2) (local 3) (local 6) (local 7);
  [%expect {| none 0 3 none |}];
  Printf.printf "%d %d\n"
    (Dim.wrap (idx 7) (extent 3) :> int)
    (Dim.wrap (idx 2) (extent 3) :> int);
  [%expect {| 1 2 |}]
