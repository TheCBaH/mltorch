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

let run ?mutation ?fuel label p inputs =
  let r = Mir_source.check_program ?mutation ?fuel p ~inputs in
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

let%expect_test "quantized decodes" =
  let i8_all = Array.init 256 (fun i -> i - 128) in
  let per_tensor = Ssa_format.Per_tensor { scale = 0.37; zero_point = -3 } in
  run "i8 per tensor, all values"
    (copy ~w:256L ~decode:Ssa_op.Decode.I8_dequant
       ~encode:Ssa_op.Encode.F32_round (Ssa_format.I8 per_tensor) Ssa_format.F32)
    [ (0, Ssa_memory.Ints i8_all) ];
  run "i16 per tensor"
    (copy ~w:6L ~decode:Ssa_op.Decode.I16_dequant
       ~encode:Ssa_op.Encode.F32_round
       (Ssa_format.I16 (Ssa_format.Per_tensor { scale = 1e-3; zero_point = 7 }))
       Ssa_format.F32)
    [ (0, Ssa_memory.Ints [| -32768; -1; 0; 1; 7; 32767 |]) ];
  let per_channel =
    Ssa_format.Per_channel
      { scale = [| 0.5; -2.; 1e-2 |]; zero_point = [| 0; 5; -9 |] }
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
    Ssa_format.Per_channel { scale = [| 0.5; -2. |]; zero_point = [| 0; 5 |] }
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
