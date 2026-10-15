open Ssa_ir
module B = Ssa_builder

(* the first index of [sub] in [s] *)
module Str_find = struct
  let find s sub =
    let n = String.length s and m = String.length sub in
    let rec go i =
      if i + m > n then None
      else if String.sub s i m = sub then Some i
      else go (i + 1)
    in
    go 0
end

module F = Ssa_ir_test.Ssa_fixtures

(* M4.2: narrow and quantized storage through the generic route, against the
   SSA interpreter's codecs (Ssa_half, Ssa_format.dequantize): every binary16
   and bfloat16 encoding, every i8 value under per-tensor parameters, every
   channel's parameters under per-channel ones, bool loads and stores and i32
   widening; a decode reading the wrong channel is detected. *)

let buffer id ~w ?(c = 1L) format role =
  {
    Ssa_buffer.id = F.buf id;
    extents = Expr.Coord.make ~n:1L ~t:1L ~d:1L ~h:1L ~w ~c;
    format;
    role;
  }

let at bld ~w ~c =
  let zero = B.index bld 0L in
  Ssa_builder.Coord (Expr.Coord.make ~n:zero ~t:zero ~d:zero ~h:zero ~w ~c)

(* out[w, c] <- encode (decode in[w, c]) over every cell *)
let copy ~w ?(c = 1L) ~decode ~encode in_format out_format =
  F.build
    ~buffers:
      [
        buffer 0 ~w ~c in_format Ssa_buffer.Input;
        buffer 1 ~w ~c out_format Ssa_buffer.Output;
      ]
    (fun bld ->
      let B.Nil =
        B.for_ bld ~lo:(B.index bld 0L) ~hi:(B.index bld w) ~init:B.Nil
          (fun bld x B.Nil ->
            B.for_ bld ~lo:(B.index bld 0L) ~hi:(B.index bld c) ~init:B.Nil
              (fun bld k B.Nil ->
                let v = B.load_f64 bld (F.buf 0) ~decode (at bld ~w:x ~c:k) in
                B.store_f64 bld (F.buf 1) ~encode (at bld ~w:x ~c:k) v;
                B.Nil))
      in
      ())

let run ?mutation ?fuel ?precision label p inputs =
  let r = Mir_source.check_program ?mutation ?fuel ?precision p ~inputs in
  (* every cell's value would be noise: the status, then any disagreement *)
  let status =
    match String.index_opt r ' ' with Some i -> String.sub r 0 i | None -> r
  in
  let tail =
    match Str_find.find r " DISAGREE " with
    | Some i -> String.sub r i (String.length r - i)
    | None -> ""
  in
  Fmt.pr "%s: %s%s@." label status tail

let%expect_test "every binary16 and bfloat16 encoding" =
  let all = Array.init 65536 Fun.id in
  run ~fuel:100_000_000L "f16"
    (copy ~w:65536L ~decode:Ssa_op.Decode.F16_to_f64
       ~encode:Ssa_op.Encode.F32_round Ssa_format.F16 Ssa_format.F32)
    [ (0, Ssa_memory.Ints all) ];
  run "bf16"
    (copy ~w:65536L ~decode:Ssa_op.Decode.Bf16_to_f64
       ~encode:Ssa_op.Encode.F32_round Ssa_format.Bf16 Ssa_format.F32)
    [ (0, Ssa_memory.Ints all) ];
  [%expect {|
    f16: ok
    bf16: ok |}]

let%expect_test "bool loads and stores, i32 widening" =
  run "bool -> f32"
    (copy ~w:4L ~decode:Ssa_op.Decode.Bool_to_f64
       ~encode:Ssa_op.Encode.F32_round Ssa_format.Bool Ssa_format.F32)
    [ (0, Ssa_memory.Floats [| 0.; 1.; 1.; 0. |]) ];
  run "f32 -> bool"
    (copy ~w:6L ~decode:Ssa_op.Decode.F32_to_f64
       ~encode:Ssa_op.Encode.Bool_nonzero Ssa_format.F32 Ssa_format.Bool)
    [ (0, Ssa_memory.Floats [| 0.; -0.; 1.; Float.nan; -2.5; 1e-45 |]) ];
  run "i32 -> f32"
    (copy ~w:6L ~decode:Ssa_op.Decode.I32_to_f64 ~encode:Ssa_op.Encode.F32_round
       Ssa_format.I32 Ssa_format.F32)
    [
      (0, Ssa_memory.Ints [| 0; -1; 0x7FFF_FFFF; -0x8000_0000; 16777217; 123 |]);
    ];
  [%expect {|
    bool -> f32: ok
    f32 -> bool: ok
    i32 -> f32: ok |}]

(* A producer's double, rounded once at the boundary into the format. *)
let f32 x = fst (Quant.round_scale x)

let%expect_test "quantized decodes" =
  let i8_all = Array.init 256 (fun i -> i - 128) in
  let per_tensor =
    Ssa_format.Per_tensor { scale = f32 0.37; zero_point = -3. }
  in
  run "i8 per tensor, all values"
    (copy ~w:256L ~decode:Ssa_op.Decode.I8_dequant
       ~encode:Ssa_op.Encode.F32_round (Ssa_format.I8 per_tensor) Ssa_format.F32)
    [ (0, Ssa_memory.Ints i8_all) ];
  run "i16 per tensor"
    (copy ~w:6L ~decode:Ssa_op.Decode.I16_dequant
       ~encode:Ssa_op.Encode.F32_round
       (Ssa_format.I16
          (Ssa_format.Per_tensor { scale = f32 1e-3; zero_point = 7. }))
       Ssa_format.F32)
    [ (0, Ssa_memory.Ints [| -32768; -1; 0; 1; 7; 32767 |]) ];
  let per_channel =
    Ssa_format.Per_channel
      { scale = [| 0.5; -2.; f32 1e-2 |]; zero_point = [| 0.; 5.; -9. |] }
  in
  let p =
    copy ~w:4L ~c:3L ~decode:Ssa_op.Decode.I8_dequant
      ~encode:Ssa_op.Encode.F32_round (Ssa_format.I8 per_channel) Ssa_format.F32
  in
  let cells =
    Ssa_memory.Ints (Array.init 12 (fun i -> (i * 37 mod 256) - 128))
  in
  run "i8 per channel" p [ (0, cells) ];
  run ~mutation:Machine_lower.Mir_lower.Mutation.Channel_zero
    "i8 per channel, channel 0 always" p
    [ (0, cells) ];
  [%expect
    {|
    i8 per tensor, all values: ok
    i16 per tensor: ok
    i8 per channel: ok
    i8 per channel, channel 0 always: ok DISAGREE structured vs generic: output t1[1]: 0x1.8p+7:f32 vs -0x1.6cp+5:f32 |}]

(* A flat access names no channel, so a per-channel decode through one keeps
   its refusal: the SSA verifier rejects the program before it reaches the
   lowering. *)
let%expect_test "per-channel decode through a flat access" =
  let per_channel =
    Ssa_format.Per_channel { scale = [| 0.5; -2. |]; zero_point = [| 0.; 5. |] }
  in
  match
    F.build
      ~buffers:
        [
          buffer 0 ~w:1L ~c:2L (Ssa_format.I8 per_channel) Ssa_buffer.Input;
          buffer 1 ~w:1L ~c:2L Ssa_format.F32 Ssa_buffer.Output;
        ]
      (fun bld ->
        let k = B.index bld 1L in
        let v =
          B.load_f64 bld (F.buf 0) ~decode:Ssa_op.Decode.I8_dequant
            (Ssa_builder.Flat k)
        in
        B.store_f64 bld (F.buf 1) ~encode:Ssa_op.Encode.F32_round
          (Ssa_builder.Flat k) v)
  with
  | p -> run "flat per channel" p [ (0, Ssa_memory.Ints [| 3; 4 |]) ]
  | exception Err.Exn.E e ->
      Fmt.pr "rejected: %a@." Err.Exn.pp_kind e;
      [%expect
        {| rejected: r0, stmt1: b0 is per-channel quantized and takes no flat access |}]

(* The binary32 dequantization through the generic route: [q] to float, minus
   the zero point, times the scale, each operation in binary32, against the SSA
   interpreter's [Ssa_format.dequantize32]. Fractional zero points, and a case
   whose difference needs more than 24 bits. *)
let%expect_test "quantized decodes in binary32" =
  let f32_run label p inputs =
    run ~precision:Machine_ir.Mir_planning.Precision.F32 label p inputs
  in
  let i16_all = Array.init 65536 (fun i -> i - 32768) in
  f32_run "i16 per tensor, tiny zero point, all values"
    (copy ~w:65536L ~decode:Ssa_op.Decode.I16_dequant32
       ~encode:Ssa_op.Encode.F32_exact
       (Ssa_format.I16
          (Ssa_format.Per_tensor { scale = f32 0.25; zero_point = f32 0.001 }))
       Ssa_format.F32)
    [ (0, Ssa_memory.Ints i16_all) ];
  f32_run "i8 per tensor, fractional zero point"
    (copy ~w:256L ~decode:Ssa_op.Decode.I8_dequant32
       ~encode:Ssa_op.Encode.F32_exact
       (Ssa_format.I8
          (Ssa_format.Per_tensor { scale = f32 0.37; zero_point = 2.5 }))
       Ssa_format.F32)
    [ (0, Ssa_memory.Ints (Array.init 256 (fun i -> i - 128))) ];
  let per_channel =
    Ssa_format.Per_channel
      {
        scale = [| 0.5; -2.; f32 1e-2 |];
        zero_point = [| 0.5; 5.; f32 (-9.25) |];
      }
  in
  let p =
    copy ~w:4L ~c:3L ~decode:Ssa_op.Decode.I8_dequant32
      ~encode:Ssa_op.Encode.F32_exact (Ssa_format.I8 per_channel) Ssa_format.F32
  in
  let cells =
    Ssa_memory.Ints (Array.init 12 (fun i -> (i * 37 mod 256) - 128))
  in
  f32_run "i8 per channel, fractional zero points" p [ (0, cells) ];
  run ~precision:Machine_ir.Mir_planning.Precision.F32
    ~mutation:Machine_lower.Mir_lower.Mutation.Channel_zero
    "i8 per channel, channel 0 always" p
    [ (0, cells) ];
  [%expect
    {|
    i16 per tensor, tiny zero point, all values: ok
    i8 per tensor, fractional zero point: ok
    i8 per channel, fractional zero points: ok
    i8 per channel, channel 0 always: ok DISAGREE structured vs generic: output t1[1]: 0x1.8p+7:f32 vs -0x1.6ep+5:f32 |}]

(* Narrow storage under [Fp32_ordered]: the exact form reads a binary16,
   bfloat16, bool or i32 cell straight to binary32 ([F16_to_f32],
   [Bf16_to_f32], [Bool_to_f32], [I32_to_f32]), scalar and, for the forced
   neon128 target, as a vector built lane by lane. Each plan agrees with the
   SSA interpreter, which agrees with the binary64 decode narrowed once, and
   its generic Machine IR holds no binary64 instruction (the binary64 decode
   is a decode and a narrowing: two conversions a read). *)
let%expect_test "narrow storage read straight to binary32" =
  let census p =
    match
      Err.payload
        (Machine_lower.Mir_lower.program
           ~planning:
             (Some
                (Machine_ir.Mir_planning.make
                   ~subject:(Machine_lower.Mir_lower.subject p)
                   ~policy:"fp32_ordered" ~schedule:"scalar"
                   ~precision:Machine_ir.Mir_planning.Precision.F32
                   ~lanes:(Machine_ir.Mir_type.Lanes.of_int 1)
                   ~fma:Machine_ir.Mir_planning.Fma.Forbidden ~capabilities:[]))
           p)
    with
    | Error _ -> "refused"
    | Ok l ->
        let c =
          Machine_ir.Mir_f64_census.of_program l.Machine_lower.Mir_lower.program
        in
        Fmt.str "conversion %d, total %d"
          (Machine_ir.Mir_f64_census.count c
             Machine_ir.Mir_f64_census.Role.Conversion)
          (List.fold_left
             (fun n r -> n + Machine_ir.Mir_f64_census.count c r)
             0 Machine_ir.Mir_f64_census.Role.all)
  in
  let decodes (p : Ssa_program.t) =
    let names = ref [] in
    let rec region (r : Ssa_region.t) = List.iter stmt r.Ssa_region.body
    and stmt = function
      | Ssa_stmt.Instr
          {
            Ssa_instr.op =
              ( Ssa_op.Load_in_bounds { decode; _ }
              | Ssa_op.Load { decode; _ }
              | Ssa_op.Vec_load { decode; _ } );
            _;
          } ->
          names := Ssa_op.Decode.name decode :: !names
      | Ssa_stmt.Instr _ -> ()
      | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
          region body
      | Ssa_stmt.If { then_; else_; _ } ->
          region then_;
          region else_
    in
    region p.Ssa_program.entry;
    String.concat "," (List.sort_uniq compare !names)
  in
  let all = Array.init 65536 Fun.id in
  List.iter
    (fun (label, w, decode, format, cells) ->
      let p =
        copy ~w ~decode ~encode:Ssa_op.Encode.F32_round format Ssa_format.F32
      in
      let alias = Ssa_effects.Distinct_buffers in
      let scalar =
        (Ssa_plan.resolve ~alias ~target:Ssa_target.scalar
           ~numerics:Ssa_numerics.Fp32_ordered p)
          .Ssa_plan.program
      in
      let vector =
        match
          Ssa_plan.fp32_vector ~alias
            ~target:(Ssa_target.forced Ssa_target.neon128)
            p
        with
        | Ok t -> t.Ssa_plan.program
        | Error _ -> assert false
      in
      let interpreted q =
        let memory =
          F.memory
            [
              ( 0,
                match cells with
                | Ssa_memory.Ints a -> Ssa_memory.Ints (Array.copy a)
                | Ssa_memory.Floats a -> Ssa_memory.Floats (Array.copy a)
                | Ssa_memory.Int64s a -> Ssa_memory.Int64s (Array.copy a) );
              (1, F.floats (Array.make (Int64.to_int w) 0.));
            ]
        in
        F.run q ~memory;
        F.output_floats memory 1
      in
      let reference = interpreted p in
      List.iter
        (fun (form, q) ->
          Fmt.pr "%s %s [%s] %s; = binary64 decode %b: " label form (decodes q)
            (census q)
            (Array.for_all2 Core.Float_bits.equal_portable reference
               (interpreted q));
          run ~fuel:100_000_000L
            ~precision:Machine_ir.Mir_planning.Precision.F32 "MIR" q
            [ (0, cells) ])
        [ ("binary64 decode", p); ("scalar", scalar); ("vector", vector) ])
    [
      ( "f16",
        65536L,
        Ssa_op.Decode.F16_to_f64,
        Ssa_format.F16,
        Ssa_memory.Ints all );
      ( "bf16",
        65536L,
        Ssa_op.Decode.Bf16_to_f64,
        Ssa_format.Bf16,
        Ssa_memory.Ints all );
      ( "bool",
        4L,
        Ssa_op.Decode.Bool_to_f64,
        Ssa_format.Bool,
        Ssa_memory.Floats [| 0.; 1.; 1.; 0. |] );
      ( "i32",
        32L,
        Ssa_op.Decode.I32_to_f64,
        Ssa_format.I32,
        Ssa_memory.Ints
          (Array.init 32 (fun k ->
               [|
                 0;
                 -1;
                 0x7FFF_FFFF;
                 -0x8000_0000;
                 16777217;
                 123;
                 -16777219;
                 8388609;
               |].(k mod 8))) );
    ];
  [%expect
    {|
    f16 binary64 decode [f16_to_f64] conversion 4, total 10; = binary64 decode true: MIR: ok
    f16 scalar [f16_to_f32] conversion 0, total 0; = binary64 decode true: MIR: ok
    f16 vector [f16_to_f32] conversion 0, total 0; = binary64 decode true: MIR: ok
    bf16 binary64 decode [bf16_to_f64] conversion 2, total 2; = binary64 decode true: MIR: ok
    bf16 scalar [bf16_to_f32] conversion 0, total 0; = binary64 decode true: MIR: ok
    bf16 vector [bf16_to_f32] conversion 0, total 0; = binary64 decode true: MIR: ok
    bool binary64 decode [bool_to_f64] conversion 1, total 4; = binary64 decode true: MIR: ok
    bool scalar [bool_to_f32] conversion 0, total 0; = binary64 decode true: MIR: ok
    bool vector [bool_to_f32] conversion 0, total 0; = binary64 decode true: MIR: ok
    i32 binary64 decode [i32_to_f64] conversion 2, total 2; = binary64 decode true: MIR: ok
    i32 scalar [i32_to_f32] conversion 0, total 0; = binary64 decode true: MIR: ok
    i32 vector [i32_to_f32] conversion 0, total 0; = binary64 decode true: MIR: ok |}]
