open Machine_ir
open Machine_interp
open Mir_fixtures
open Mir_harness
module B = Mir_builder

(* C1, continued: comparator-detected mutations, the outcome taxonomy, helper
   models and calls, and the independently derived numeric primitives. *)

let i32v x = Val (Mir_datum.i32 x)
let i64v x = Val (Mir_datum.i64 x)

let ok results =
  {
    Mir_observation.status = Mir_observation.Status.Success;
    outputs =
      [
        {
          Mir_observation.Output.source = Expr.Source.create 0;
          cells = Array.of_list (List.map Option.some results);
        };
      ];
    events = List.map (fun e -> (e, 0L)) Mir_event.all;
  }

let failed f =
  {
    Mir_observation.status =
      Mir_observation.Status.Failure
        {
          Mir_observation.Row.failure = f;
          payload = [];
          invocation = None;
          site = None;
        };
    outputs = [];
    events = List.map (fun e -> (e, 0L)) Mir_event.all;
  }

let%expect_test "mutations the comparison detects" =
  (* transfer order: a swap repeated five times against its reference *)
  let swap_obs p =
    let r, _, _ = run p [ i32v 5L ] in
    observe r ~results:[ i64; i64 ]
  in
  let expected = ok [ Mir_const.i64 2L; Mir_const.i64 1L ] in
  Fmt.pr "swap: %s@." (verdict ~expected ~actual:(swap_obs (swap ())));
  Fmt.pr "sequential: %s@."
    (verdict ~expected ~actual:(swap_obs (swap ~sequential:true ())));
  (* failure selection: the overflow block reports the zero-divisor kind *)
  let wrong =
    Mir_verify_test.map_blocks
      (fun (b : (_, _) Mir_block.t) ->
        match b.Mir_block.terminator with
        | Mir_terminator.Fail
            ({ Mir_fail.failure = Mir_failure.I64_division_overflow; _ } as f)
          ->
            {
              b with
              Mir_block.terminator =
                Mir_terminator.Fail
                  { f with Mir_fail.failure = Mir_failure.I64_division_by_zero };
            }
        | _ -> b)
      (div ())
  in
  let div_obs p =
    let r, _, _ = run p [ i64v Int64.min_int; i64v (-1L) ] in
    observe r ~results:[ i64 ]
  in
  let expected = failed Mir_failure.I64_division_overflow in
  Fmt.pr "division: %s@." (verdict ~expected ~actual:(div_obs (div ())));
  Fmt.pr "wrong kind: %s@." (verdict ~expected ~actual:(div_obs wrong));
  (* endian decode: the four bytes assembled big-endian from byte loads *)
  let decode ~big =
    let bld = B.create () in
    let e = B.new_block bld [ Mir_type.Ptr ] in
    let base = List.hd (B.param e) in
    let byte k =
      let a =
        B.emit bld e (Mir_op.Ptr_add (base, k64 bld e (Int64.of_int k)))
      in
      let b =
        B.emit bld e
          (Mir_op.Load
             { Mir_op.Access.width = Mir_width.W8; addr = a; align = 1L })
      in
      B.emit bld e (Mir_op.Iext (Mir_op.Iext.Zext, Mir_width.W32, b))
    in
    let acc =
      List.fold_left
        (fun acc k ->
          let sh =
            B.emit bld e (Mir_op.Iarith (Mir_op.Iarith.Shl, acc, k32 bld e 8L))
          in
          B.emit bld e (Mir_op.Iarith (Mir_op.Iarith.Or, sh, byte k)))
        (k32 bld e 0L)
        (if big then [ 0; 1; 2; 3 ] else [ 3; 2; 1; 0 ])
    in
    let f = B.emit bld e (Mir_op.Bitcast (Mir_type.F32, acc)) in
    B.return e
      [ B.emit bld e (Mir_op.Fconvert (Mir_op.Fconvert.F32_to_f64, f)) ];
    with_objects
      (program
         (B.func bld ~id:fn ~name:"decode" ~entry:e ~results:[ Mir_type.F64 ]))
      ~regions:[ region 0 4L ]
      ~views:[ view 0 ~region:0 4L ]
  in
  let expected = ok [ Mir_const.f64 1.5 ] in
  List.iter
    (fun big ->
      let r, _, _ =
        run (decode ~big) ~bound:[ (0, f32_bytes [ 1.5 ]) ] [ Ptr 0 ]
      in
      Fmt.pr "%s: %s@."
        (if big then "big-endian" else "little-endian")
        (verdict ~expected ~actual:(observe r ~results:[ Mir_type.F64 ])))
    [ false; true ];
  [%expect
    {|
    swap: agree
    sequential: mismatch: output t0[1]: 1:i64 vs 2:i64
    division: agree
    wrong kind: mismatch: failure row: i64_division_overflow() vs i64_division_by_zero()
    little-endian: agree
    big-endian: mismatch: output t0[0]: 0x1.8p+0:f64 vs 0x1.807ep-134:f64 |}]

let%expect_test "distinct outcomes: failure, defect, unsupported, fuel" =
  (* a loop that never ends: test fuel, not a scan budget *)
  let spin =
    let bld = B.create () in
    let e = B.new_block bld [] in
    let l = B.new_block bld [] in
    B.jump e l [];
    B.jump l l [];
    program (B.func bld ~id:fn ~name:"spin" ~entry:e ~results:[])
  in
  let r, _, _ = run ~fuel:1000L spin [] in
  Fmt.pr "%s after %Ld steps@." (show_outcome r) r.Mir_interp.steps;
  (* a helper the program needs and no model provides *)
  let exp_helper =
    {
      Mir_helper.id = Mir_id.Helper.of_int 0;
      name = "exp";
      version = 1;
      params = [ Mir_type.F64 ];
      results = [ Mir_type.F64 ];
      effects = Mir_helper.Effect.Pure;
      failures = [];
    }
  in
  let calls_exp =
    let bld = B.create () in
    let e = B.new_block bld [ Mir_type.F64 ] in
    let sigs = function
      | Mir_op.Callee.Helper _ ->
          Some
            {
              Mir_typing.Signature.params = [ Mir_type.F64 ];
              results = [ Mir_type.F64 ];
            }
      | _ -> None
    in
    let rs =
      Result.get_ok
        (B.op bld e ~signature:sigs
           (Mir_op.Call
              (Mir_op.Callee.Helper (Mir_id.Helper.of_int 0), B.param e)))
    in
    B.return e rs;
    program ~helpers:[ exp_helper ]
      (B.func bld ~id:fn ~name:"calls_exp" ~entry:e ~results:[ Mir_type.F64 ])
  in
  let r, _, _ = run calls_exp [ Val (Mir_datum.f64 1.) ] in
  print_endline (show_outcome r);
  (* the same program with a deterministic model of the helper *)
  let model =
    {
      Mir_helper_model.name = "exp";
      version = 1;
      run =
        (fun _ -> function
          | [ Mir_datum.Bits b ] ->
              Mir_helper_model.Returns
                [ Mir_datum.f64 (Float.exp (Int64.float_of_bits b)) ]
          | _ -> assert false);
    }
  in
  let r, _, _ = run ~models:[ model ] calls_exp [ Val (Mir_datum.f64 1.) ] in
  print_endline (show_outcome r);
  (* a model raising a failure its descriptor does not declare is a defect *)
  let lying =
    {
      model with
      Mir_helper_model.run =
        (fun _ _ ->
          Mir_helper_model.Fails (Mir_failure.I64_division_by_zero, []));
    }
  in
  let r, _, _ = run ~models:[ lying ] calls_exp [ Val (Mir_datum.f64 1.) ] in
  print_endline (show_outcome r);
  (* declared, it propagates unchanged *)
  let declared =
    {
      calls_exp with
      Mir_program.helpers =
        [
          {
            exp_helper with
            Mir_helper.failures = [ Mir_failure.I64_division_by_zero ];
          };
        ];
    }
  in
  let r, _, _ = run ~models:[ lying ] declared [ Val (Mir_datum.f64 1.) ] in
  print_endline (show_outcome r);
  (* a partial operation outside its domain, reached through a bad proof *)
  let narrow =
    let bld = B.create () in
    let e = B.new_block bld [ i64 ] in
    B.return e
      [ B.emit bld e (Mir_op.Narrow (Mir_width.W32, List.hd (B.param e))) ];
    program (B.func bld ~id:fn ~name:"narrow" ~entry:e ~results:[ i32 ])
  in
  List.iter
    (fun x ->
      let r, _, _ = run narrow [ i64v x ] in
      print_endline (show_outcome r))
    [ -0x8000_0000L; 0x8000_0000L ];
  [%expect
    {|
    fuel exhausted after 1000 steps
    unsupported exp
    success [0x4005bf0a8b145769]
    defect invalid_program at fn0 bb0 i0
    failure i64_division_by_zero()
    success [0x80000000]
    defect domain at fn0 bb0 i0 |}]

let%expect_test "calls and failure prefixes" =
  (* main counts one event, calls [div], counts another: a failing callee ends
     the invocation after the first event only *)
  let callee = div () in
  let callee_f = List.hd callee.Mir_program.funcs in
  let callee_f = { callee_f with Mir_func.id = Mir_id.Func.of_int 1 } in
  let bld = B.create () in
  let e = B.new_block bld [ i64; i64 ] in
  B.emit_unit bld e (Mir_op.Event (Mir_event.Key, 1L));
  let sigs = function
    | Mir_op.Callee.Func _ ->
        Some { Mir_typing.Signature.params = [ i64; i64 ]; results = [ i64 ] }
    | _ -> None
  in
  let q =
    Result.get_ok
      (B.op bld e ~signature:sigs
         (Mir_op.Call (Mir_op.Callee.Func (Mir_id.Func.of_int 1), B.param e)))
  in
  B.emit_unit bld e (Mir_op.Event (Mir_event.Key, 1L));
  B.return e q;
  let main = B.func bld ~id:fn ~name:"main" ~entry:e ~results:[ i64 ] in
  let p = Mir_builder.program [ main; callee_f ] ~main:fn in
  List.iter
    (fun (x, y) ->
      let r, _, _ = run p [ i64v x; i64v y ] in
      print_endline (show_outcome r))
    [ (9L, 2L); (9L, 0L) ];
  [%expect
    {|
    success [0x4] events key=2
    failure i64_division_by_zero() events key=1 |}]

(* Independently derived numerics, checked against the SSA library's own
   independent derivations and exhibited where the naive double rounding
   differs. *)
let%expect_test "binary32 FMA and i64-to-binary32 round once" =
  let b32 x =
    Int64.logand (Int64.of_int32 (Int32.bits_of_float x)) 0xFFFF_FFFFL
  in
  let f32 b = Int32.float_of_bits (Int64.to_int32 b) in
  let st = Random.State.make [| 20261007 |] in
  let random32 () =
    let m = Random.State.float st 2. -. 1. in
    let e = Random.State.int st 40 - 20 in
    b32 (Float.ldexp m e)
  in
  let disagreements = ref 0 and double_rounding = ref None in
  for _ = 1 to 200_000 do
    let a = random32 () and b = random32 () and c = random32 () in
    let mine = Mir_numeric.fma32 a b c in
    let ssa = b32 (Ssa_ir.Ssa_numerics.fma32 (f32 a) (f32 b) (f32 c)) in
    if not (Int64.equal mine ssa) then incr disagreements;
    let naive = b32 (Float.fma (f32 a) (f32 b) (f32 c)) in
    if Option.is_none !double_rounding && not (Int64.equal naive mine) then
      double_rounding := Some (a, b, c, mine, naive)
  done;
  (* exact 1 + 2^-24 + 2^-60: binary64 lands on the binary32 midpoint first *)
  (let a = b32 (-.Float.ldexp (1. +. Float.ldexp 1. (-18)) (-24))
   and b = b32 (1. -. Float.ldexp 1. (-18))
   and c = b32 (1. +. Float.ldexp 1. (-23)) in
   double_rounding :=
     Some
       ( a,
         b,
         c,
         Mir_numeric.fma32 a b c,
         b32 (Float.fma (f32 a) (f32 b) (f32 c)) );
   Fmt.pr "constructed case agrees with SSA: %b@."
     (Int64.equal (Mir_numeric.fma32 a b c)
        (b32 (Ssa_ir.Ssa_numerics.fma32 (f32 a) (f32 b) (f32 c)))));
  Fmt.pr "fma32 disagreements with the SSA derivation: %d@." !disagreements;
  (match !double_rounding with
  | Some (a, b, c, mine, naive) ->
      Fmt.pr "double rounding at %h*%h+%h: once %h, twice %h@." (f32 a) (f32 b)
        (f32 c) (f32 mine) (f32 naive)
  | None -> print_endline "no double-rounding case found");
  let cases =
    [
      0L;
      1L;
      -1L;
      16777217L;
      0x20000020000001L;
      Int64.add (Int64.shift_left 1L 62) (Int64.add (Int64.shift_left 1L 38) 1L);
      Int64.max_int;
      Int64.min_int;
      Int64.neg 0x20000020000001L;
    ]
  in
  List.iter
    (fun n ->
      let mine = f32 (Mir_numeric.s64_to_f32 n) in
      Fmt.pr "%Ld -> %h (ssa %h, via binary64 %h)@." n mine
        (Ssa_ir.Ssa_const.round32_of_i64 n)
        (f32 (b32 (Int64.to_float n))))
    cases;
  let bad = ref 0 in
  for _ = 1 to 100_000 do
    let n = Random.State.int64 st Int64.max_int in
    let n = if Random.State.bool st then Int64.neg n else n in
    if
      not
        (Float.equal
           (f32 (Mir_numeric.s64_to_f32 n))
           (Ssa_ir.Ssa_const.round32_of_i64 n))
    then incr bad
  done;
  Fmt.pr "i64->f32 disagreements: %d@." !bad;
  [%expect
    {|
    constructed case agrees with SSA: true
    fma32 disagreements with the SSA derivation: 0
    double rounding at -0x1.00004p-24*0x1.ffff8p-1+0x1.000002p+0: once 0x1.000002p+0, twice 0x1p+0
    0 -> 0x0p+0 (ssa 0x0p+0, via binary64 0x0p+0)
    1 -> 0x1p+0 (ssa 0x1p+0, via binary64 0x1p+0)
    -1 -> -0x1p+0 (ssa -0x1p+0, via binary64 -0x1p+0)
    16777217 -> 0x1p+24 (ssa 0x1p+24, via binary64 0x1p+24)
    9007199791611905 -> 0x1.000002p+53 (ssa 0x1.000002p+53, via binary64 0x1p+53)
    4611686293305294849 -> 0x1.000002p+62 (ssa 0x1.000002p+62, via binary64 0x1p+62)
    9223372036854775807 -> 0x1p+63 (ssa 0x1p+63, via binary64 0x1p+63)
    -9223372036854775808 -> -0x1p+63 (ssa -0x1p+63, via binary64 -0x1p+63)
    -9007199791611905 -> -0x1.000002p+53 (ssa -0x1.000002p+53, via binary64 -0x1p+53)
    i64->f32 disagreements: 0 |}]

let%expect_test "maximum, conversions and integer edges" =
  let show = Fmt.pr "%h@." in
  show (Mir_numeric.fmax (-0.) 0.);
  show (Mir_numeric.fmax 0. (-0.));
  show (Mir_numeric.fmax Float.nan 1.);
  show (Mir_numeric.fmax 1. Float.nan);
  show (Mir_numeric.fmax (-1.) (-2.));
  List.iter
    (fun x ->
      Fmt.pr "%h -> %a@." x
        Fmt.(option ~none:(any "outside") int64)
        (Mir_numeric.f64_to_s64 x))
    [
      -9223372036854775808.;
      9223372036854775808.;
      9223372036854774784.;
      -0.5;
      Float.nan;
      Float.infinity;
    ];
  let w = Mir_width.W32 in
  Fmt.pr "sdiv -7/2=%Ld srem=%Ld@."
    (Mir_width.signed w
       (Option.get
          (Mir_numeric.int_div Mir_op.Idiv.Div_s w
             (Mir_width.normalize w (-7L))
             2L)))
    (Mir_width.signed w
       (Option.get
          (Mir_numeric.int_div Mir_op.Idiv.Rem_s w
             (Mir_width.normalize w (-7L))
             2L)));
  Fmt.pr "min/-1 defined: %b@."
    (Option.is_some
       (Mir_numeric.int_div Mir_op.Idiv.Div_s w 0x8000_0000L 0xFFFF_FFFFL));
  Fmt.pr "ashr -8>>1=%Ld lshr=%Lx@."
    (Mir_width.signed w
       (Mir_numeric.int_binary Mir_op.Iarith.Shr_s w
          (Mir_width.normalize w (-8L))
          1L))
    (Mir_numeric.int_binary Mir_op.Iarith.Shr_u w
       (Mir_width.normalize w (-8L))
       1L);
  Fmt.pr "add wraps: %Lx@."
    (Mir_numeric.int_binary Mir_op.Iarith.Add w 0x7FFF_FFFFL 1L);
  Fmt.pr "slt -1<0: %b ult: %b@."
    (Mir_numeric.int_compare Mir_op.Icmp.Slt w 0xFFFF_FFFFL 0L)
    (Mir_numeric.int_compare Mir_op.Icmp.Ult w 0xFFFF_FFFFL 0L);
  [%expect
    {|
    0x0p+0
    0x0p+0
    nan
    nan
    -0x1p+0
    -0x1p+63 -> -9223372036854775808
    0x1p+63 -> outside
    0x1.fffffffffffffp+62 -> 9223372036854774784
    -0x1p-1 -> 0
    nan -> outside
    infinity -> outside
    sdiv -7/2=-3 srem=-1
    min/-1 defined: false
    ashr -8>>1=-4 lshr=7ffffffc
    add wraps: 80000000
    slt -1<0: true ult: false |}]
