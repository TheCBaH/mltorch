open Ssa_bridge
open Ssa_ir
module B = Ssa_builder
module Fx = Ssa_ir_test.Ssa_fixtures
module Lf = Loop_ir_test.Loop_fixtures
module Lp = Loop_ir_test.Loop_programs

(* The binary32 dequantization through generated WebAssembly from SSA, against
   the structured interpreter: [q] to float, minus the zero point, times the
   scale, each rounded once to binary32, with per-channel parameters held as
   [f32] tables and fractional zero points. *)

let f32 x = fst (Quant.round_scale x)
let cells_n = 12
let channels = 3

let program format =
  let buffers =
    [
      {
        Ssa_buffer.id = Fx.buf 0;
        extents =
          Expr.Coord.make ~n:1L ~t:1L ~d:1L ~h:1L ~w:4L
            ~c:(Int64.of_int channels);
        format;
        role = Ssa_buffer.Input;
      };
      {
        Ssa_buffer.id = Fx.buf 1;
        extents =
          Expr.Coord.make ~n:1L ~t:1L ~d:1L ~h:1L ~w:4L
            ~c:(Int64.of_int channels);
        format = Ssa_format.F32;
        role = Ssa_buffer.Output;
      };
    ]
  in
  Err.or_raise ~pp_error:Ssa_verify.pp_error
    (B.program ~buffers (fun bld ->
         let idx n = B.index bld (Int64.of_int n) in
         let at x k =
           let z = idx 0 in
           B.Coord (Expr.Coord.make ~n:z ~t:z ~d:z ~h:z ~w:x ~c:k)
         in
         let decode =
           match format with
           | Ssa_format.I16 _ -> Ssa_op.Decode.I16_dequant32
           | _ -> Ssa_op.Decode.I8_dequant32
         in
         let B.Nil =
           B.for_ bld ~lo:(idx 0) ~hi:(idx 4) ~init:B.Nil (fun bld x B.Nil ->
               B.for_ bld ~lo:(idx 0) ~hi:(idx channels) ~init:B.Nil
                 (fun bld k B.Nil ->
                   let v = B.load_f64 bld (Fx.buf 0) ~decode (at x k) in
                   B.store_f64 bld (Fx.buf 1) ~encode:Ssa_op.Encode.F32_exact
                     (at x k) v;
                   B.Nil))
         in
         ()))

let cells ~lo ~hi =
  Array.init cells_n (fun i -> lo + (i * 37 mod (hi - lo + 1)))

let structured p cells =
  let out = Array.make cells_n 0. in
  let memory =
    List.fold_left
      (fun m (b : Ssa_buffer.t) ->
        let c =
          match (b.Ssa_buffer.id :> int) with
          | 0 -> Ssa_memory.Ints (Array.copy cells)
          | _ -> Ssa_memory.Floats out
        in
        Ssa_id.Buffer.Map.add b.Ssa_buffer.id c m)
      Ssa_id.Buffer.Map.empty p.Ssa_program.buffers
  in
  (match Err.payload (Ssa_interp.run p ~memory) with
  | Ok () -> ()
  | Error _ -> failwith "structured run failed");
  out

let wasm p format cells =
  let lowered =
    match Ssa_wasm.lower ~numerics:Loop_ir.Loop_numerics.Fp32_ordered p with
    | Ok k -> k
    | Error e -> Fmt.failwith "%a" Ssa_wasm.pp_error e
  in
  let loop =
    Lf.program
      ~buffers:(List.map Loop_of_ssa.loop_buffer (Ssa_wasm.arguments p))
      []
  in
  let fmt, quant =
    match format with
    | Ssa_format.I8 q -> (Payload.Fmt Payload.I8, q)
    | Ssa_format.I16 q -> (Payload.Fmt Payload.I16, q)
    | _ -> assert false
  in
  let quant = Ssa_lower.Ssa_sig.to_quant quant in
  let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:4 ~c:channels in
  let bind id =
    if Tensor_id.equal id (Lf.tid 0) then
      Some
        (Lp.raw_tensor fmt ~quant ~shape cells_n (Array.map Int64.of_int cells))
    else None
  in
  match Err.payload (Loop_wasm_exec.exec_module ~lowered loop ~bind) with
  | Error e -> Fmt.failwith "%a" Loop_wasm_exec.pp_error e
  | Ok outputs -> (
      match Tensor_id.Map.find_opt (Lf.tid 1) outputs with
      | Some t ->
          Array.init cells_n (fun i ->
              Tensor.read t
                (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:(i / channels)
                   ~c:(i mod channels)))
      | None -> failwith "no output")

let%expect_test "binary32 SSA Wasm dequantization is the interpreter's" =
  let check name format ~lo ~hi =
    let p = program format in
    let cells = cells ~lo ~hi in
    let s = structured p cells and w = wasm p format cells in
    Fmt.pr "%-40s %s@." name
      (if Array.for_all2 Core.Float_bits.equal_exact s w then "equal"
       else "DIFFERS")
  in
  check "i8 per tensor, fractional zero point"
    (Ssa_format.I8
       (Ssa_format.Per_tensor { scale = f32 0.37; zero_point = 2.5 }))
    ~lo:(-128) ~hi:127;
  check "i16 per tensor, tiny zero point"
    (Ssa_format.I16
       (Ssa_format.Per_tensor { scale = 1.; zero_point = f32 0.001 }))
    ~lo:(-32768) ~hi:32767;
  check "i8 per channel, fractional zero points"
    (Ssa_format.I8
       (Ssa_format.Per_channel
          {
            scale = [| 0.5; -2.; f32 1e-2 |];
            zero_point = [| 0.5; 5.; f32 (-9.25) |];
          }))
    ~lo:(-128) ~hi:127;
  check "i16 per channel, fractional zero points"
    (Ssa_format.I16
       (Ssa_format.Per_channel
          {
            scale = [| f32 1e-3; f32 2e-3; 3. |];
            zero_point = [| f32 0.001; -7.5; 32767.5 |];
          }))
    ~lo:(-32768) ~hi:32767;
  [%expect
    {|
    i8 per tensor, fractional zero point     equal
    i16 per tensor, tiny zero point          equal
    i8 per channel, fractional zero points   equal
    i16 per channel, fractional zero points  equal |}]
