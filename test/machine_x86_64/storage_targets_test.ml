open Ssa_ir
module S = Machine_source_test.Mir_storage_test

(* The storage programs through both targets' selected routes and AArch64's
   allocated route and x86-64's realized one: bfloat16 and binary16 samples,
   bool loads and stores, per-tensor and per-channel quantized decodes. *)

let cases =
  let per_channel =
    Ssa_format.Per_channel
      { scale = [| 0.5; -2.; 1e-2 |]; zero_point = [| 0; 5; -9 |] }
  in
  [
    ( "bf16",
      S.copy ~w:8L ~decode:Ssa_op.Decode.Bf16_to_f64
        ~encode:Ssa_op.Encode.F32_round Ssa_format.Bf16 Ssa_format.F32,
      [
        ( 0,
          Ssa_memory.Ints
            [| 0; 0x8000; 0x3F80; 0x7F80; 0xFF80; 0x7FC1; 0x0001; 0xC2F7 |] );
      ] );
    ( "f16",
      S.copy ~w:8L ~decode:Ssa_op.Decode.F16_to_f64
        ~encode:Ssa_op.Encode.F32_round Ssa_format.F16 Ssa_format.F32,
      [
        ( 0,
          Ssa_memory.Ints
            [| 0; 0x8000; 0x3C00; 0x7C00; 0xFC00; 0x7E01; 0x0001; 0x83FF |] );
      ] );
    ( "bool",
      S.copy ~w:6L ~decode:Ssa_op.Decode.F32_to_f64
        ~encode:Ssa_op.Encode.Bool_nonzero Ssa_format.F32 Ssa_format.Bool,
      [ (0, Ssa_memory.Floats [| 0.; -0.; 1.; Float.nan; -2.5; 1e-45 |]) ] );
    ( "i8 per channel",
      S.copy ~w:4L ~c:3L ~decode:Ssa_op.Decode.I8_dequant
        ~encode:Ssa_op.Encode.F32_round (Ssa_format.I8 per_channel)
        Ssa_format.F32,
      [ (0, Ssa_memory.Ints (Array.init 12 (fun i -> (i * 37 mod 256) - 128))) ]
    );
  ]

let%expect_test "storage on both targets" =
  List.iter
    (fun (name, p, inputs) ->
      Fmt.pr "%s: aarch64 %s, allocated %s | x86_64 %s, realized %s@." name
        (Machine_aarch64_test.A64_harness.program p ~inputs)
        (Machine_alloc_test.Alloc_harness.program p ~inputs)
        (X64_harness.program p ~inputs)
        (X64_harness.program ~stage:X64_harness.Realized p ~inputs))
    cases;
  [%expect
    {|
    bf16: aarch64 ok, allocated ok | x86_64 ok, realized ok
    f16: aarch64 ok, allocated ok | x86_64 ok, realized ok
    bool: aarch64 ok, allocated ok | x86_64 ok, realized ok
    i8 per channel: aarch64 ok, allocated ok | x86_64 ok, realized ok |}]
