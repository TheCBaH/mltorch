open Native4d

(* The product the lowerers form over a tensor's extents is bounded where it is
   formed. N=T=D=2^11 passes every per-axis limit and its product, 2^33, wraps a
   32-bit [int] to 0 under js_of_ocaml: an unchecked [*] would hand
   [Dim.extent] a 0 there (a raise) or, for other shapes, a silently wrong fused
   extent. *)
let show = function
  | Some e -> string_of_int (e : Dim.extent Dim.t :> int)
  | None -> "declined"

let%expect_test "extent product: bounded below the numel ceiling" =
  let p es = show (Extent_product.bounded (List.map Dim.extent es)) in
  print_endline (p [ 2048; 2048; 2048 ]);
  print_endline (p [ 4; 5; 6 ]);
  print_endline (p []);
  print_endline (p [ 65536; 32768 ]);
  print_endline (p [ 65536; 32767 ]);
  [%expect {|
    declined
    120
    1
    declined
    2147418112 |}]
