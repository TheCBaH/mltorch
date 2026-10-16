open Loop_ir
open Loop_fixtures
open Loop_programs

(* Quantized reads under the binary32 policy: [q] to float, minus the zero
   point, times the scale, each operation rounded once to binary32
   ([Quant.dequantize32]). The generated Wasm (binary32 tables, [f32]
   operations) is bitwise the binary32 interpreter; and the fixtures include
   fractional zero points and values whose difference needs more than 24 bits,
   where the binary32 meaning differs from the binary64 reference, so a check
   that compared only with the reference would pass for the wrong reason. *)

let r32 x = fst (Quant.round_scale x)

let c_at i =
  Expr.Coord.make ~n:(Loop_index.Const 0) ~t:(Loop_index.Const 0)
    ~d:(Loop_index.Const 0) ~h:(Loop_index.Const 0) ~w:(Loop_index.Const 0) ~c:i

let program ~fmt ~quant n =
  let input =
    {
      Loop_buffer.id = tid 0;
      sg =
        Tensor_sig.create ~id:(tid 0) ~name:"" ~shape:(shape_c n) ~fmt ~quant ();
      role = Loop_buffer.Input;
    }
  in
  let output = buffer 1 (shape_c n) f32 Loop_buffer.Output in
  let i = Loop_index.Var (v 0) in
  ( input,
    program ~buffers:[ input; output ]
      [
        Loop_stmt.For
          {
            var = v 0;
            lo = Loop_index.Const 0;
            hi = Loop_index.Const n;
            body =
              [
                Loop_stmt.Store_flat
                  {
                    buffer = output;
                    offset = i;
                    value = Loop_stored.F32 (Loop_expr.Load (input, c_at i));
                  };
              ];
          };
      ] )

let fixtures =
  let t scale zp = Quant.per_tensor ~scale:(r32 scale) ~zero_point:(r32 zp) in
  let ch scale zp =
    Err.or_raise ~pp_error:Quant.pp_error
      (Quant.per_channel ~scale:(Array.map r32 scale)
         ~zero_point:(Array.map r32 zp))
  in
  [
    ( "i8 per tensor, integer zp",
      Payload.Fmt Payload.I8,
      t 0.1 (-3.),
      [| -128L; 127L; 0L; -3L |] );
    ( "i8 per tensor, fractional zp",
      Payload.Fmt Payload.I8,
      t 0.37 2.5,
      [| -128L; 127L; 0L; 3L |] );
    ( "i16 per tensor, tiny zp (difference rounds)",
      Payload.Fmt Payload.I16,
      t 0.001 0.001,
      [| -32768L; 32767L; 5L; -5L |] );
    ( "i16 per tensor, integer zp",
      Payload.Fmt Payload.I16,
      t 0.001 7.,
      [| -32768L; 32767L; 0L; 1L |] );
    ( "i8 per channel, integer zp",
      Payload.Fmt Payload.I8,
      ch [| 0.5; 0.25; 2.; 0.1 |] [| 0.; 1.; -2.; 5. |],
      [| -128L; 127L; 3L; -4L |] );
    ( "i8 per channel, fractional zp",
      Payload.Fmt Payload.I8,
      ch [| 0.5; 0.25; 2.; 0.1 |] [| 0.5; -1.25; 2.75; 100.125 |],
      [| -128L; 127L; 3L; -4L |] );
    ( "i16 per channel, fractional zp",
      Payload.Fmt Payload.I16,
      ch [| 1e-3; 2e-3; 3e-3; 4e-3 |] [| 0.001; -7.5; 7.25; 32767.5 |],
      [| -32768L; 32767L; 5L; -5L |] );
  ]

let verdict a b =
  match (Err.payload a, Err.payload b) with
  | Ok a, Ok b ->
      if Tensor_id.Map.equal Loop_check.tensors_equal a b then "equal"
      else "DIFFERS"
  | _ -> "FAILED"

let%expect_test "binary32 Wasm dequantization is the binary32 oracle" =
  List.iter
    (fun (name, fmt, quant, cells) ->
      let n = Array.length cells in
      let input, p = program ~fmt ~quant n in
      let bind id =
        if Tensor_id.equal id input.Loop_buffer.id then
          Some (raw_tensor fmt ~quant n cells)
        else None
      in
      let precision = Loop_numerics.Precision.F32 in
      let oracle = Loop_interp.run ~precision p ~bind in
      let wasm = Loop_wasm_exec.exec ~precision p ~bind in
      let reference = Loop_interp.run p ~bind in
      Fmt.pr "%-44s wasm %s the f32 oracle; f32 oracle %s the f64 reference@."
        name (verdict oracle wasm)
        (if verdict oracle reference = "equal" then "equals" else "differs from"))
    fixtures;
  [%expect
    {|
    i8 per tensor, integer zp                    wasm equal the f32 oracle; f32 oracle equals the f64 reference
    i8 per tensor, fractional zp                 wasm equal the f32 oracle; f32 oracle equals the f64 reference
    i16 per tensor, tiny zp (difference rounds)  wasm equal the f32 oracle; f32 oracle differs from the f64 reference
    i16 per tensor, integer zp                   wasm equal the f32 oracle; f32 oracle equals the f64 reference
    i8 per channel, integer zp                   wasm equal the f32 oracle; f32 oracle equals the f64 reference
    i8 per channel, fractional zp                wasm equal the f32 oracle; f32 oracle equals the f64 reference
    i16 per channel, fractional zp               wasm equal the f32 oracle; f32 oracle equals the f64 reference |}]
