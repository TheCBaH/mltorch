open Ssa_bridge
open Loop_ir_test
open Loop_fixtures
open Loop_programs

(* Every storage format a Kernel input can have, decoded by a load, and the Bool
   boundary, direct: the cases are the Loop suite's own, against the reference. *)

let show name ~fmt ?quant cells =
  let n = Array.length cells in
  let plan = Fusion_plan.default (format_kernel ~fmt ?quant n) in
  let bind id =
    if Tensor_id.equal id (tid 0) then Some (raw_tensor fmt ?quant n cells)
    else None
  in
  Fmt.pr "%s: %a@." name Ssa_check.pp_verdict (Ssa_check.run plan ~bind)

let fmt f = Payload.Fmt f
let bits = Int64.bits_of_float
let per_tensor scale zero_point = Quant.per_tensor ~scale ~zero_point

let per_channel scale zero_point =
  Err.or_raise ~pp_error:Quant.pp_error (Quant.per_channel ~scale ~zero_point)

let%expect_test
    "half and bfloat16 decode, including subnormals, infinities and NaN" =
  show "f16" ~fmt:(fmt Payload.F16) [| 0x3c00L; 0x0001L; 0x7c00L; 0xfe00L |];
  show "f16 negative zero and largest finite" ~fmt:(fmt Payload.F16)
    [| 0x8000L; 0x7bffL; 0x03ffL; 0x0400L |];
  show "bf16" ~fmt:(fmt Payload.BF16) [| 0x3f80L; 0x0001L; 0x7f80L; 0xffc0L |];
  [%expect
    {|
    f16: agree
    f16 negative zero and largest finite: agree
    bf16: agree
    |}]

let%expect_test
    "quantized cells are scale * (q - zero point), per tensor and per channel" =
  show "i8 per tensor" ~fmt:(fmt Payload.I8) ~quant:(per_tensor 0.1 (-3))
    [| -128L; 127L; 0L; -3L |];
  show "i8 per channel, last channel" ~fmt:(fmt Payload.I8)
    ~quant:(per_channel [| 0.5; 0.25; 2.; 0.1 |] [| 0; 1; -2; 5 |])
    [| -128L; 127L; 3L; -4L |];
  show "i16 per tensor" ~fmt:(fmt Payload.I16) ~quant:(per_tensor 0.001 7)
    [| -32768L; 32767L; 0L; 1L |];
  show "i16 per channel" ~fmt:(fmt Payload.I16)
    ~quant:(per_channel [| 1e-3; 2e-3; 3e-3; 4e-3 |] [| 0; -7; 7; 32767 |])
    [| -32768L; 32767L; 5L; -5L |];
  [%expect
    {|
    i8 per tensor: agree
    i8 per channel, last channel: agree
    i16 per tensor: agree
    i16 per channel: agree
    |}]

let%expect_test "wide formats load exactly to the nearest double" =
  show "i32 extremes" ~fmt:(fmt Payload.I32)
    [| Int64.of_int32 Int32.min_int; Int64.of_int32 Int32.max_int; -1L; 0L |];
  show "f64" ~fmt:(fmt Payload.F64)
    [| bits 1e300; bits 5e-324; bits (-0.); bits nan |];
  show "i64 read as a float" ~fmt:(fmt Payload.I64)
    [|
      Int64.add (Int64.shift_left 1L 53) 1L; Int64.min_int; Int64.max_int; -1L;
    |];
  show "bool bytes" ~fmt:(fmt Payload.Bool) [| 0L; 1L; 2L; 255L |];
  [%expect
    {|
    i32 extremes: agree
    f64: agree
    i64 read as a float: agree
    bool bytes: agree
    |}]

let%expect_test
    "a Bool value is nonzero on the working value, read back as 0 or 1" =
  let run name ~input_fmt cells =
    let plan = Fusion_plan.default (bool_kernel ~input_fmt) in
    let bind id =
      if Tensor_id.equal id (tid 0) then
        let shape = shape_w 4 in
        Some
          (match input_fmt with
          | Payload.Fmt Payload.F64 ->
              Tensor.materialize_fmt input_fmt shape (fun c ->
                  Int64.float_of_bits cells.((Vec6.offset shape c :> int)))
          | _ ->
              Tensor.materialize shape (fun c ->
                  Int64.float_of_bits cells.((Vec6.offset shape c :> int))))
      else None
    in
    Fmt.pr "%s: %a@." name Ssa_check.pp_verdict (Ssa_check.run plan ~bind)
  in
  (* NaN and 1e-40 are true, both zeros false *)
  run "f32" ~input_fmt:(fmt Payload.F32)
    [| bits nan; bits (-0.); bits 0.; bits 1e-40 |];
  (* below binary32's smallest subnormal the working value is nonzero, but
     rounding to f32 first would say false *)
  run "f64 below binary32's range" ~input_fmt:(fmt Payload.F64)
    [| bits 1e-50; bits 1e-300; bits (-0.); bits 5e-324 |];
  [%expect {|
    f32: agree
    f64 below binary32's range: agree
    |}]

let%expect_test
    "a filled input folds to its decoded value, after its bounds check" =
  List.iter
    (fun v ->
      let plan = Fusion_plan.default (bool_fill_kernel v) in
      Fmt.pr "bool fill %g: %a@." v Ssa_check.pp_verdict
        (Ssa_check.run plan ~bind:bind_none))
    [ 0.; -0.; 0.5; nan; 1e-50; 3. ];
  (* the fill is materialized as f32 by the reference: 2^24 + 1 reads back as 2^24 *)
  Fmt.pr "f32 fill: %a@." Ssa_check.pp_verdict
    (Ssa_check.run
       (Fusion_plan.default (filled_kernel 16777217.))
       ~bind:(Ssa_fixtures.bind_data ~shape:(shape_w 4) [| 0.; 1.; 2.; 3. |]));
  [%expect
    {|
    bool fill 0: agree
    bool fill -0: agree
    bool fill 0.5: agree
    bool fill nan: agree
    bool fill 1e-50: agree
    bool fill 3: agree
    f32 fill: agree
    |}]

let%expect_test "the 16-bit decodes agree with the native codec on every cell" =
  let differing decode native =
    List.length
      (List.filter
         (fun bits ->
           not (Core.Float_bits.equal_portable (decode bits) (native bits)))
         (List.init 0x10000 Fun.id))
  in
  Fmt.pr "f16: %d differ, bf16: %d differ@."
    (differing Ssa_ir.Ssa_half.f16_to_float Half.Half.to_float)
    (differing Ssa_ir.Ssa_half.bf16_to_float Half.Bf16.to_float);
  [%expect {| f16: 0 differ, bf16: 0 differ |}]
