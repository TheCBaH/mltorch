let%expect_test "per-tensor quant round-trip + pp" =
  let q = Quant.Per_tensor { scale = 0.5; zero_point = 0 } in
  Format.printf "%a deq(4)=%g req(2.0)=%d@." Quant.pp q
    (Quant.dequantize q ~c:0 ~q:4)
    (Quant.quantize q ~c:0 ~qmin:(-128) ~qmax:127 2.0);
  [%expect {| Per_tensor s=0.5 zero_point=0 deq(4)=2 req(2.0)=4 |}]

let%expect_test "per-tensor with zero_point + clamping" =
  let q = Quant.Per_tensor { scale = 0.1; zero_point = 128 } in
  (* dequant of 138 = 0.1 * (138 - 128) = 1.0 *)
  Format.printf "deq=%g req(1.0)=%d clamp_hi=%d@."
    (Quant.dequantize q ~c:0 ~q:138)
    (Quant.quantize q ~c:0 ~qmin:0 ~qmax:255 1.0)
    (Quant.quantize q ~c:0 ~qmin:0 ~qmax:255 1000.0);
  [%expect {| deq=1 req(1.0)=138 clamp_hi=255 |}]

let%expect_test "per-channel picks scale/zp by channel" =
  let q =
    Quant.Per_channel { scale = [| 0.5; 0.25 |]; zero_point = [| 0; 1 |] }
  in
  Format.printf "%a ch0=%g ch1=%g@." Quant.pp q
    (Quant.dequantize q ~c:0 ~q:4)
    (Quant.dequantize q ~c:1 ~q:5);
  (* ch1 = 0.25 * (5 - 1) = 1.0 *)
  [%expect {| Per_channel(2) ch0=2 ch1=1 |}]
