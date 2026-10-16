open Loop_ir

(* The binary64 census of the Wasm modules the Loop emitter builds for the
   shared program corpus: each program as the binary64 reference module and as
   a forced scalar binary32 module. The baseline for the binary32 policy: what a
   binary32 module still computes in binary64, and what it imports. *)

module P = Loop_vector_programs

let census ?precision p =
  match Err.payload (Loop_wasm.lower ?precision p) with
  | Ok l ->
      let c = Wasm_census.of_module l.Loop_wasm.module_ in
      Fmt.str "%a" Wasm_census.pp c
  | Error e -> Fmt.failwith "%a" Loop_wasm.pp_error e

let%expect_test "binary64 census of the corpus, reference and forced binary32" =
  List.iter
    (fun (name, p) ->
      Fmt.pr "%s@.  f64: %s@." name (census p);
      match Loop_numerics.admit p with
      | Error _ -> Fmt.pr "  f32: refused@."
      | Ok () ->
          Fmt.pr "  f32: %s@." (census ~precision:Loop_numerics.Precision.F32 p))
    (P.all @ Loop_fp32_programs.extra);
  [%expect
    {|
    double rounding
      f64: arithmetic 2, conversion 6, helper_call 0, memory_move 2; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    arith chain
      f64: arithmetic 4, conversion 5, helper_call 0, memory_move 1; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    offset view
      f64: arithmetic 0, conversion 2, helper_call 0, memory_move 0; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    broadcast and invariant
      f64: arithmetic 2, conversion 3, helper_call 0, memory_move 1; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    strided
      f64: arithmetic 1, conversion 2, helper_call 0, memory_move 1; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    maximum
      f64: arithmetic 1, conversion 3, helper_call 0, memory_move 0; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    select and compare
      f64: arithmetic 3, conversion 4, helper_call 0, memory_move 2; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    pool_better
      f64: arithmetic 2, conversion 5, helper_call 0, memory_move 6; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    not and or
      f64: arithmetic 2, conversion 4, helper_call 0, memory_move 3; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    index value
      f64: arithmetic 1, conversion 3, helper_call 0, memory_move 0; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    temporaries
      f64: arithmetic 3, conversion 2, helper_call 0, memory_move 7; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    sqrt and trunc
      f64: arithmetic 3, conversion 2, helper_call 0, memory_move 1; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    transcendentals
      f64: arithmetic 2, conversion 3, helper_call 2, memory_move 1; imports 2
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    int32 source
      f64: arithmetic 1, conversion 3, helper_call 0, memory_move 0; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    bool store
      f64: arithmetic 2, conversion 2, helper_call 0, memory_move 1; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    nested loops, vector inner
      f64: arithmetic 1, conversion 3, helper_call 0, memory_move 0; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    matmul: reduction per output
      f64: arithmetic 5, conversion 3, helper_call 0, memory_move 16; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    reduction with a triangular inner bound
      f64: arithmetic 2, conversion 3, helper_call 0, memory_move 5; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    uniform index temporary
      f64: arithmetic 2, conversion 3, helper_call 0, memory_move 3; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    extents around the width: 3
      f64: arithmetic 1, conversion 2, helper_call 0, memory_move 1; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    extents around the width: 4
      f64: arithmetic 1, conversion 2, helper_call 0, memory_move 1; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    extents around the width: 5
      f64: arithmetic 1, conversion 2, helper_call 0, memory_move 1; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    matvec: 21 outputs, k 4
      f64: arithmetic 2, conversion 5, helper_call 0, memory_move 5; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    matmul: 3 rows, 19 outputs, k 4
      f64: arithmetic 2, conversion 5, helper_call 0, memory_move 5; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    matvec: exactly one vector
      f64: arithmetic 2, conversion 5, helper_call 0, memory_move 5; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    int64 to float, once rounded
      f64: arithmetic 1, conversion 3, helper_call 0, memory_move 0; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    index to float
      f64: arithmetic 1, conversion 3, helper_call 0, memory_move 0; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    erf
      f64: arithmetic 19, conversion 2, helper_call 2, memory_move 26; imports 1
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    log, cos, sqrt
      f64: arithmetic 4, conversion 5, helper_call 2, memory_move 0; imports 2
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    packed float scratch
      f64: arithmetic 3, conversion 3, helper_call 0, memory_move 10; imports 0
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    f32 cos over the ramp
      f64: arithmetic 2, conversion 2, helper_call 1, memory_move 2; imports 1
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    f32 cos over the spread
      f64: arithmetic 2, conversion 3, helper_call 1, memory_move 1; imports 1
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    f32 erf over the ramp
      f64: arithmetic 21, conversion 2, helper_call 2, memory_move 28; imports 1
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    f32 erf over the spread
      f64: arithmetic 21, conversion 3, helper_call 2, memory_move 27; imports 1
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    f32 exp over the ramp
      f64: arithmetic 2, conversion 2, helper_call 1, memory_move 2; imports 1
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    f32 exp over the spread
      f64: arithmetic 2, conversion 3, helper_call 1, memory_move 1; imports 1
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    f32 log over the ramp
      f64: arithmetic 2, conversion 2, helper_call 1, memory_move 2; imports 1
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    f32 log over the spread
      f64: arithmetic 2, conversion 3, helper_call 1, memory_move 1; imports 1
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    f32 sin over the ramp
      f64: arithmetic 2, conversion 2, helper_call 1, memory_move 2; imports 1
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    f32 sin over the spread
      f64: arithmetic 2, conversion 3, helper_call 1, memory_move 1; imports 1
      f32: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0 |}]

(* The check can fail: one [f64] promotion injected into a binary32 kernel's
   emitted body is counted, as a conversion, and a binary32 module with an
   [f64.add] is counted as arithmetic. *)
let%expect_test "an injected f64 operation is caught by the census" =
  let _, p = List.hd Loop_vector_programs.all in
  let m =
    match
      Err.payload (Loop_wasm.lower ~precision:Loop_numerics.Precision.F32 p)
    with
    | Ok l -> l.Loop_wasm.module_
    | Error e -> Fmt.failwith "%a" Loop_wasm.pp_error e
  in
  Fmt.pr "emitted:  %a@." Wasm_census.pp (Wasm_census.of_module m);
  let inject extra =
    {
      m with
      Wasm.Module.funcs =
        List.map
          (fun (f : Wasm.Func.t) ->
            { f with Wasm.Func.body = f.Wasm.Func.body @ extra })
          m.Wasm.Module.funcs;
    }
  in
  Fmt.pr "promote:  %a@." Wasm_census.pp
    (Wasm_census.of_module
       (inject
          [
            Wasm.Instr.F32_const 0l;
            Wasm.Instr.Numeric Wasm_op.F64_promote_f32;
            Wasm.Instr.Drop;
          ]));
  Fmt.pr "f64 add:  %a@." Wasm_census.pp
    (Wasm_census.of_module
       (inject
          [
            Wasm.Instr.F64_const 0L;
            Wasm.Instr.F64_const 0L;
            Wasm.Instr.Numeric Wasm_op.F64_add;
            Wasm.Instr.Drop;
          ]));
  [%expect
    {|
    emitted:  arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    promote:  arithmetic 0, conversion 1, helper_call 0, memory_move 0; imports 0
    f64 add:  arithmetic 1, conversion 0, helper_call 0, memory_move 2; imports 0 |}]
