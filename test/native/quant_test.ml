let%expect_test "per-tensor quant round-trip + pp" =
  let q = Quant.per_tensor ~scale:0.5 ~zero_point:0 in
  Format.printf "%a deq(4)=%g req(2.0)=%d@." Quant.pp q
    (Quant.dequantize q ~c:0 ~q:4)
    (Quant.quantize q ~c:0 ~qmin:(-128) ~qmax:127 2.0);
  [%expect {| Per_tensor s=0.5 zero_point=0 deq(4)=2 req(2.0)=4 |}]

let%expect_test "per-tensor with zero_point + clamping" =
  let q = Quant.per_tensor ~scale:0.1 ~zero_point:128 in
  (* dequant of 138 = 0.1 * (138 - 128) = 1.0 *)
  Format.printf "deq=%g req(1.0)=%d clamp_hi=%d@."
    (Quant.dequantize q ~c:0 ~q:138)
    (Quant.quantize q ~c:0 ~qmin:0 ~qmax:255 1.0)
    (Quant.quantize q ~c:0 ~qmin:0 ~qmax:255 1000.0);
  [%expect {| deq=1 req(1.0)=138 clamp_hi=255 |}]

let per_channel ~scale ~zero_point =
  Core.or_raise Quant.pp_error (Quant.per_channel ~scale ~zero_point)

let%expect_test "per-channel picks scale/zp by channel" =
  let q = per_channel ~scale:[| 0.5; 0.25 |] ~zero_point:[| 0; 1 |] in
  Format.printf "%a ch0=%g ch1=%g@." Quant.pp q
    (Quant.dequantize q ~c:0 ~q:4)
    (Quant.dequantize q ~c:1 ~q:5);
  (* ch1 = 0.25 * (5 - 1) = 1.0 *)
  [%expect {| Per_channel(2) ch0=2 ch1=1 |}]

let%expect_test "per_channel rejects unequal array lengths" =
  Format.printf "%a@."
    (Core.Pretty.core_result ~ok:Quant.pp ~error:Quant.pp_error)
    (Quant.per_channel ~scale:[| 0.5; 0.25 |] ~zero_point:[| 0 |]);
  [%expect {| quant scale/zero_point lengths differ: 2 vs 1 |}]

(* [per_channel] COPIES. Switching the call sites above to the constructor only
   proves it exists; this is what proves it copied, and it is the test that
   makes the whole abstraction worth its cost — a [Quant.t] reachable through a
   validated [Tensor_sig.t] must not change under its owner. *)
let%expect_test "per_channel copies the caller's arrays" =
  let scale = [| 0.5; 0.25 |] and zero_point = [| 0; 1 |] in
  let q = per_channel ~scale ~zero_point in
  let before =
    Format.asprintf "%a ch1=%g" Quant.pp q (Quant.dequantize q ~c:1 ~q:5)
  in
  scale.(1) <- 99.0;
  zero_point.(1) <- 99;
  let after =
    Format.asprintf "%a ch1=%g" Quant.pp q (Quant.dequantize q ~c:1 ~q:5)
  in
  Format.printf "before: %s@.after:  %s@.unchanged: %b@." before after
    (String.equal before after);
  (* [equal] must also be unmoved by the mutation. *)
  Format.printf "equal to a fresh copy: %b@."
    (Quant.equal q (per_channel ~scale:[| 0.5; 0.25 |] ~zero_point:[| 0; 1 |]));
  [%expect
    {|
    before: Per_channel(2) ch1=1
    after:  Per_channel(2) ch1=1
    unchanged: true
    equal to a fresh copy: true |}]

let%expect_test "equal compares granularity, lengths and bits" =
  let p = per_channel ~scale:[| 0.5; 0.25 |] ~zero_point:[| 0; 1 |] in
  let same = per_channel ~scale:[| 0.5; 0.25 |] ~zero_point:[| 0; 1 |] in
  let other_scale = per_channel ~scale:[| 0.5; 0.5 |] ~zero_point:[| 0; 1 |] in
  let other_zp = per_channel ~scale:[| 0.5; 0.25 |] ~zero_point:[| 0; 2 |] in
  let shorter = per_channel ~scale:[| 0.5 |] ~zero_point:[| 0 |] in
  let tensor = Quant.per_tensor ~scale:0.5 ~zero_point:0 in
  Format.printf "same=%b scale=%b zp=%b length=%b granularity=%b@."
    (Quant.equal p same)
    (Quant.equal p other_scale)
    (Quant.equal p other_zp) (Quant.equal p shorter) (Quant.equal tensor p);
  (* -0. and 0. are distinct bit patterns and must not compare equal, the same
     rule [Tensor.equal_bits] states. [Float.equal] would say true here. *)
  Format.printf "signed zero: %b@."
    (Quant.equal
       (Quant.per_tensor ~scale:0.0 ~zero_point:0)
       (Quant.per_tensor ~scale:(-0.0) ~zero_point:0));
  [%expect
    {|
    same=true scale=false zp=false length=false granularity=false
    signed zero: false |}]

let%expect_test "channel_count reports granularity" =
  let p = per_channel ~scale:[| 0.5; 0.25; 0.125 |] ~zero_point:[| 0; 1; 2 |] in
  Format.printf "per_channel=%s per_tensor=%s@."
    (match Quant.channel_count p with
    | Some n -> string_of_int n
    | None -> "none")
    (match Quant.channel_count (Quant.per_tensor ~scale:0.5 ~zero_point:0) with
    | Some n -> string_of_int n
    | None -> "none");
  [%expect {| per_channel=3 per_tensor=none |}]

let%expect_test "jsont rejects mismatched arrays at the decode boundary" =
  (* Through the public decoder: the constructor's [Core.result] has to become
     Jsont's ordinary [Error _], not an exception escaping [decode_string]. *)
  let decode s =
    match Jsont_bytesrw.decode_string Quant.jsont s with
    | Ok q -> Format.asprintf "ok %a" Quant.pp q
    | Error m -> "error: " ^ m
  in
  print_endline
    (decode {|{"Per_channel":{"scale":[0.5,0.25],"zero_point":[0,1]}}|});
  print_endline
    (decode {|{"Per_channel":{"scale":[0.5,0.25],"zero_point":[0]}}|});
  [%expect
    {|
    ok Per_channel(2)
    error: quant scale/zero_point lengths differ: 2 vs 1 |}]
