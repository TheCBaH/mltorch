open Ssa_ir
module S = Machine_source_test.Mir_storage_test

(* The storage programs through both targets' selected routes and AArch64's
   allocated route and x86-64's realized one: bfloat16 and binary16 samples,
   bool loads and stores, per-tensor and per-channel quantized decodes. *)

let cases =
  let per_channel =
    Ssa_format.Per_channel
      {
        scale = [| 0.5; -2.; fst (Quant.round_scale 1e-2) |];
        zero_point = [| 0.; 5.; -9. |];
      }
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

(* The binary32 dequantization on both targets: [q] to float, minus the zero
   point, times the scale, in binary32 ([f32] subtraction and multiplication in
   registers), against the structured interpreter, with fractional zero
   points. *)
let%expect_test "binary32 quantized decodes on both targets" =
  let precision = Machine_ir.Mir_planning.Precision.F32 in
  let f32 x = fst (Quant.round_scale x) in
  let per_channel =
    Ssa_format.Per_channel
      {
        scale = [| 0.5; -2.; f32 1e-2 |];
        zero_point = [| 0.5; 5.; f32 (-9.25) |];
      }
  in
  let cases =
    [
      ( "i8 per tensor, fractional zero point",
        S.copy ~w:6L ~decode:Ssa_op.Decode.I8_dequant32
          ~encode:Ssa_op.Encode.F32_exact
          (Ssa_format.I8
             (Ssa_format.Per_tensor { scale = f32 0.37; zero_point = 2.5 }))
          Ssa_format.F32,
        [ (0, Ssa_memory.Ints [| -128; -1; 0; 1; 3; 127 |]) ] );
      ( "i16 per tensor, tiny zero point",
        S.copy ~w:5L ~decode:Ssa_op.Decode.I16_dequant32
          ~encode:Ssa_op.Encode.F32_exact
          (Ssa_format.I16
             (Ssa_format.Per_tensor { scale = 1.; zero_point = f32 0.001 }))
          Ssa_format.F32,
        [ (0, Ssa_memory.Ints [| -32768; -1; 0; 1; 32767 |]) ] );
      ( "i8 per channel, fractional zero points",
        S.copy ~w:4L ~c:3L ~decode:Ssa_op.Decode.I8_dequant32
          ~encode:Ssa_op.Encode.F32_exact (Ssa_format.I8 per_channel)
          Ssa_format.F32,
        [
          (0, Ssa_memory.Ints (Array.init 12 (fun i -> (i * 37 mod 256) - 128)));
        ] );
    ]
  in
  List.iter
    (fun (name, p, inputs) ->
      Fmt.pr "%s: aarch64 %s, allocated %s | x86_64 %s, realized %s@." name
        (Machine_aarch64_test.A64_harness.program ~precision p ~inputs)
        (Machine_alloc_test.Alloc_harness.program ~precision p ~inputs)
        (X64_harness.program ~precision p ~inputs)
        (X64_harness.program ~precision ~stage:X64_harness.Realized p ~inputs))
    cases;
  [%expect
    {|
    i8 per tensor, fractional zero point: aarch64 ok, allocated ok | x86_64 ok, realized ok
    i16 per tensor, tiny zero point: aarch64 ok, allocated ok | x86_64 ok, realized ok
    i8 per channel, fractional zero points: aarch64 ok, allocated ok | x86_64 ok, realized ok |}]
