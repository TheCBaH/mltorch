open Inference_compare_schema
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module C = Codec_fixtures

(* Resource limits are printed without their platform-dependent bound. *)
let show_error name = function
  | Ok _ -> Format.printf "%s: ok@." name
  | Error e -> (
      match Err.Error.kind e with
      | Codecs.Codec_error.Resource_exceeded _ ->
          Format.printf "%s: resource exceeded@." name
      | _ -> Format.printf "%s: %a@." name C.codec_error_pp e)

let%expect_test "construction is checked" =
  let ctx = C.context () in
  show_error "exact length" (L.make ctx (C.f32 [ 2L ]) (String.make 8 '\000'));
  show_error "short" (L.make ctx (C.f32 [ 2L ]) (String.make 7 '\000'));
  show_error "long" (L.make ctx (C.f32 [ 2L ]) (String.make 9 '\000'));
  show_error "empty tensor" (L.make ctx (C.f32 [ 3L; 0L ]) "");
  show_error "count overflow"
    (L.make ctx (C.f32 [ 4294967296L; 4294967296L ]) "");
  show_error "payload beyond the string bound"
    (L.make ctx (C.f32 [ 4611686018427387904L ]) "");
  show_error "negative extent" (L.make ctx (C.f32 [ -1L ]) "");
  let odd = C.spec (Codecs.Element_codec.id "bf16") [ 1L ] in
  show_error "unknown codec" (L.make ctx odd "\000\000");
  [%expect
    {|
    exact length: ok
    short: byte length: expected 8, got 7
    long: byte length: expected 8, got 9
    empty tensor: ok
    count overflow: element count overflow
    payload beyond the string bound: resource exceeded
    negative extent: negative extent -1
    unknown codec: unknown codec bf16 |}]

let%expect_test "exact bits survive" =
  let ctx = C.context () in
  (* 2^53 + 1 is not representable as a double: an integer that went through
     floating point would come back as 2^53. *)
  let big = Int64.add (Int64.shift_left 1L 53) 1L in
  let ints =
    C.integers ctx (C.i64 [ 3L ]) [ big; Int64.min_int; Int64.max_int ]
  in
  Format.printf "i64: %a@." C.pp_ints (C.to_integers ctx ints);
  let signalling_nan = 0x7FA00001L and neg_zero = 0x80000000L in
  let fs =
    let b = Bytes.create 12 in
    List.iteri
      (fun i bits -> Bytes.set_int32_le b (i * 4) (Int64.to_int32 bits))
      [ signalling_nan; neg_zero; 0x7F800000L ];
    C.ok (L.make ctx (C.f32 [ 3L ]) (Bytes.to_string b))
  in
  Format.printf "f32 bits: %a@." C.pp_hex_bits (C.to_bits ctx fs);
  let u = C.integers ctx (C.u8 [ 3L ]) [ 0L; 255L; 256L ] in
  Format.printf "u8 truncates at the codec width: %a@." C.pp_ints
    (C.to_integers ctx u);
  [%expect
    {|
    i64: [9007199254740993; -9223372036854775808; 9223372036854775807]
    f32 bits: [0x7fa00001; 0x80000000; 0x7f800000]
    u8 truncates at the codec width: [0; 255; 0] |}]

let%expect_test "digest identifies spec and bytes, equality compares bytes" =
  let ctx = C.context () in
  let a = C.integers ctx (C.i32 [ 2L ]) [ 1L; 2L ] in
  let b = C.integers ctx (C.i32 [ 2L ]) [ 1L; 2L ] in
  let reshaped = C.integers ctx (C.i32 [ 1L; 2L ]) [ 1L; 2L ] in
  let other = C.integers ctx (C.i32 [ 2L ]) [ 1L; 3L ] in
  Format.printf "same value equal: %b@." (L.equal a b);
  Format.printf "same digest: %b@."
    (Digest.Logical_value.equal (L.digest a) (L.digest b));
  Format.printf "reshape changes the digest: %b@."
    (not (Digest.Logical_value.equal (L.digest a) (L.digest reshaped)));
  Format.printf "different payload: %b@." (not (L.equal a other));
  [%expect
    {|
    same value equal: true
    same digest: true
    reshape changes the digest: true
    different payload: true |}]

let%expect_test "ledger limits and balance" =
  let ctx = C.context ~limit:16L () in
  let a = C.integers ctx (C.i32 [ 2L ]) [ 1L; 2L ] in
  Format.printf "live: %Ld bytes, %d blobs@."
    (Codecs.Owned_blob.Ledger.live_bytes ctx.ledger)
    (Codecs.Owned_blob.Ledger.live_blobs ctx.ledger);
  let b = C.integers ctx (C.i32 [ 2L ]) [ 3L; 4L ] in
  show_error "over the limit" (L.of_integers ctx (C.i32 [ 1L ]) [ 5L ]);
  Format.printf "peak: %Ld, live after refusal: %Ld@."
    (Codecs.Owned_blob.Ledger.peak_bytes ctx.ledger)
    (Codecs.Owned_blob.Ledger.live_bytes ctx.ledger);
  let lease = Codecs.Owned_blob.lease (L.storage a) in
  L.drop a;
  Format.printf "after drop, leased: %Ld bytes live, %d leases@."
    (Codecs.Owned_blob.Ledger.live_bytes ctx.ledger)
    (Codecs.Owned_blob.Ledger.outstanding_leases ctx.ledger);
  Format.printf "lease still reads: %b@."
    (String.length (Codecs.Owned_blob.Lease.contents lease) = 8);
  Codecs.Owned_blob.Lease.release lease;
  L.drop b;
  Format.printf "balanced: %b@." (Codecs.Owned_blob.Ledger.balanced ctx.ledger);
  (try ignore (Codecs.Owned_blob.contents (L.storage a))
   with Invalid_argument m -> Format.printf "use after release: %s@." m);
  [%expect
    {|
    live: 8 bytes, 1 blobs
    over the limit: resource exceeded
    peak: 16, live after refusal: 16
    after drop, leased: 16 bytes live, 1 leases
    lease still reads: true
    balanced: true
    use after release: Owned_blob.contents: released |}]
