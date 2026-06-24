open Payload

let%expect_test "format names" =
  Format.printf "%a %a %a %a %a %a %a@." pp_fmt F32 pp_fmt I64 pp_fmt I32 pp_fmt
    F16 pp_fmt BF16 pp_fmt I16 pp_fmt I8;
  [%expect {| f32 i64 i32 f16 bf16 i16 i8 |}]

let%expect_test "decode f32 and bf16 (real, no metadata)" =
  let f32 =
    {
      fmt = F32;
      quant = No_quant;
      data = Bigarray.(Array1.of_array float32 c_layout [| 1.5; -2.0 |]);
    }
  in
  let bf16 =
    (* raw bf16 cells: 0x3F80 = 1.0, 0x4040 = 3.0 *)
    {
      fmt = BF16;
      quant = No_quant;
      data =
        Bigarray.(Array1.of_array int16_unsigned c_layout [| 0x3F80; 0x4040 |]);
    }
  in
  Format.printf "%a %g %g | %a %g %g@." pp f32 (get_float f32 ~c:0 ~i:0)
    (get_float f32 ~c:0 ~i:1) pp bf16 (get_float bf16 ~c:0 ~i:0)
    (get_float bf16 ~c:0 ~i:1);
  [%expect {| f32 1.5 -2 | bf16 1 3 |}]

let%expect_test
    "decode i8 (quantized: needs metadata) + round-trip via set_float" =
  let q = Quant.Per_tensor { scale = 0.5; zero_point = 0 } in
  let p =
    {
      fmt = I8;
      quant = Quant q;
      data = Bigarray.(Array1.of_array int8_signed c_layout [| 2; -4 |]);
    }
  in
  Format.printf "%a: %g %g@." pp p (get_float p ~c:0 ~i:0)
    (get_float p ~c:0 ~i:1);
  [%expect {| i8[Per_tensor s=0.5 zp=0]: 1 -2 |}];
  set_float p ~c:0 ~i:0 3.0;
  (* 3.0 / 0.5 = 6 stored, decodes back to 3.0 *)
  Format.printf "stored=%d deq=%g@." p.data.{0} (get_float p ~c:0 ~i:0);
  [%expect {| stored=6 deq=3 |}]

(* The quant discipline is a compile-time guarantee; this must NOT compile:
   let _bad = { fmt = I8; quant = No_quant;
                data = Bigarray.(Array1.of_array int8_signed c_layout [| 0 |]) }
   -> Error: No_quant has type [`Real] quantization but [`Quant] was expected. *)
