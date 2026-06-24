let%expect_test "bf16 decode = high 16 bits of f32; round-trip exact values" =
  (* 0x3F80 << 16 = 0x3F800000 = 1.0 *)
  Format.printf "%a %a@." Half.Bf16.pp 0x3F80 Half.Bf16.pp 0x4040;
  [%expect {| 0x3f80(1) 0x4040(3) |}];
  List.iter
    (fun f -> Format.printf "%g " (Half.Bf16.to_float (Half.Bf16.of_float f)))
    [ 1.0; 1.5; -3.0; 0.5 ];
  Format.printf "@.";
  [%expect {| 1 1.5 -3 0.5 |}]

let%expect_test "half decode known + round-trip exact values" =
  (* 0x3C00 = 1.0, 0x4000 = 2.0, 0x3800 = 0.5 *)
  Format.printf "%g %g %g@."
    (Half.Half.to_float 0x3C00)
    (Half.Half.to_float 0x4000)
    (Half.Half.to_float 0x3800);
  [%expect {| 1 2 0.5 |}];
  List.iter
    (fun f -> Format.printf "%g " (Half.Half.to_float (Half.Half.of_float f)))
    [ 1.0; 1.5; -3.0; 0.5; 100.0 ];
  Format.printf "@.";
  [%expect {| 1 1.5 -3 0.5 100 |}]
