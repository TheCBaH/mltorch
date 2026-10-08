open Machine_ir
open Machine_interp
open Machine_target_x86_64
open X64_op
module Sel = X64_stage.Sel
module H = X64_harness
module Loc = Mir_phys.Loc

(* x86-64 negatives: a condition read from bits its producer leaves
   undefined, a form whose feature the program lacks, an IDIV divisor placed
   where CQO writes; and calls through the selected, allocated and realized
   routes with return addresses pushed on the stack. *)

let v n ty = { Mir_value.id = Mir_id.Value.of_int n; ty }
let o n = v n Mir_type.Order

let program ?(features = [ Mir_target.Feature.Sse2 ]) params body results =
  let instrs =
    List.mapi
      (fun k (rs, op) ->
        {
          Mir_instr.id = Mir_id.Instr.of_int k;
          results = rs;
          op = Mir_sel.Op.Machine op;
          order = None;
          origin = Mir_origin.unknown;
        })
      body
  in
  {
    Sel.features;
    program =
      {
        Mir_program.data_model = Mir_layout.Data_model.Lp64_le;
        regions = [];
        views = [];
        helpers = [];
        funcs =
          [
            {
              Mir_func.id = Mir_id.Func.of_int 0;
              name = "t";
              entry = Mir_id.Block.of_int 0;
              results =
                List.map (fun (r : Mir_value.t) -> r.Mir_value.ty) results;
              blocks =
                [
                  {
                    Mir_block.id = Mir_id.Block.of_int 0;
                    params;
                    order = o 999;
                    body = instrs;
                    terminator =
                      Mir_sel.Terminator.Return
                        { Mir_return.values = results; order = o 999 };
                  };
                ];
            };
          ];
        main = Mir_id.Func.of_int 0;
        planning = None;
        revision = Mir_id.Revision.of_int 0;
      };
  }

let run sel args =
  match Err.payload (Sel.verify sel) with
  | Error d -> Fmt.pr "rejected: %a@." Mir_diagnostic.pp d
  | Ok vsel ->
      let memory = Mir_memory.create () in
      let binding =
        Result.get_ok
          (Mir_interp.instantiate sel.Sel.program memory ~bound:(fun _ -> None))
      in
      Fmt.pr "%a@." Mir_interp.Outcome.pp
        (X64_stage.Interp.run vsel memory binding ~args)
          .X64_stage.Interp.outcome

let%expect_test "x86-64 forms and malformed variants" =
  let a = v 0 Mir_type.i64 and b = v 1 Mir_type.i64 and x = v 2 Mir_type.F64 in
  let f = v 10 Mir_type.Flags and p = v 11 Mir_type.Pred in
  (* BT defines CF only *)
  run
    (program [ a; b; x ]
       [ ([ f ], Bt (Sz.Q, a, 3)); ([ p ], Setcc_zx (Cond.B, f)) ]
       [ p ])
    [ Mir_datum.Bits 8L; Mir_datum.Bits 0L; Mir_datum.f64 0. ];
  run
    (program [ a; b; x ]
       [ ([ f ], Bt (Sz.Q, a, 3)); ([ p ], Setcc_zx (Cond.L, f)) ]
       [ p ])
    [];
  (* IDIV: the quotient and remainder of 7 / -2 *)
  run
    (program [ a; b; x ]
       [ ([ v 12 Mir_type.i64; v 13 Mir_type.i64 ], Cqo_idiv (a, b)) ]
       [ v 12 Mir_type.i64; v 13 Mir_type.i64 ])
    [ Mir_datum.Bits 7L; Mir_datum.Bits (-2L); Mir_datum.f64 0. ];
  (* FMA3 without the feature *)
  run
    (program [ a; b; x ]
       [ ([ v 14 Mir_type.F64 ], Fmadd231 (Fsz.D, x, x, x)) ]
       [ v 14 Mir_type.F64 ])
    [];
  (* MAXSD: the source operand for a NaN or two zeros *)
  let y = v 3 Mir_type.F64 in
  List.iter
    (fun (p, q) ->
      run
        (program [ x; y ]
           [ ([ v 15 Mir_type.F64 ], Fbin (Fop.Max, Fsz.D, x, y)) ]
           [ v 15 Mir_type.F64 ])
        [ Mir_datum.f64 p; Mir_datum.f64 q ])
    [ (Float.nan, 1.); (1., Float.nan); (-0., 0.); (0., -0.) ];
  [%expect
    {|
    success [0x1]
    rejected: selected fn0 bb0 i1: target constraint: reads a condition bit its producer leaves undefined
    success [0xfffffffffffffffd, 0x1]
    rejected: selected fn0 bb0 i0: target constraint: needs feature fma
    success [0x3ff0000000000000]
    success [0x7ff0000000000001]
    success [0x0]
    success [0x8000000000000000] |}]

let%expect_test "x86-64 immediates and their reach" =
  let a = v 0 Mir_type.i64 and w = v 1 Mir_type.i32 in
  let f = v 10 Mir_type.Flags and p = v 11 Mir_type.Pred in
  let r = v 12 Mir_type.i64 in
  let args = [ Mir_datum.Bits 6L; Mir_datum.Bits 0xFFFF_FFFFL ] in
  List.iter
    (fun op -> run (program [ a; w ] [ ([ r ], op) ] [ r ]) args)
    [
      (* the imm32 is sign-extended at 64 bits *)
      Alu_imm (Alu.Add, Sz.Q, a, -7L);
      Imul_imm (Sz.Q, a, -3L);
      Alu_imm (Alu.Add, Sz.Q, a, 0x8000_0000L);
    ];
  (* at 32 bits, every 32-bit value: -1 compares equal to 0xFFFFFFFF *)
  run
    (program [ a; w ]
       [
         ([ f ], Cmp_imm (Sz.L, w, 0xFFFF_FFFFL)); ([ p ], Setcc_zx (Cond.E, f));
       ]
       [ p ])
    args;
  [%expect
    {|
    success [0xffffffffffffffff]
    success [0xffffffffffffffee]
    rejected: selected fn0 bb0 i0: target constraint: alu immediate
    success [0x1] |}]

let%expect_test "an IDIV divisor where CQO writes is rejected" =
  let div =
    Machine_aarch64_test.A64_select_test.i64_program (fun bld x y ->
        Ssa_ir.Ssa_builder.i64_div bld x y)
  in
  let case =
    Result.get_ok
      (Machine_source_test.Mir_source.case_of_program div
         ~inputs:[ (0, Ssa_ir.Ssa_memory.Int64s [| 7L; -2L |]) ]
         ())
  in
  let res =
    Result.get_ok
      (Err.payload
         (X64_select.program
            case.Machine_source_test.Mir_source.Case.lowered
              .Machine_lower.Mir_lower.program))
  in
  let phys = H.A.allocate res.X64_select.selected in
  let rdx = Loc.Reg (X64_reg.q X64_reg.rdx) in
  let edited =
    {
      phys with
      Mir_phys.Program.funcs =
        List.map
          (fun (fn : (_, _) Mir_phys.Func.t) ->
            {
              fn with
              Mir_phys.Func.blocks =
                List.map
                  (fun (b : (_, _) Mir_phys.Block.t) ->
                    let divisor = ref None in
                    let body =
                      List.map
                        (function
                          | Mir_phys.Instr.Exec
                              ({
                                 instr =
                                   {
                                     Mir_instr.op =
                                       Mir_sel.Op.Machine (Cqo_idiv _);
                                     _;
                                   };
                                 uses = [ u0; u1 ];
                                 _;
                               } as e) ->
                              divisor := Some u1;
                              Mir_phys.Instr.Exec { e with uses = [ u0; rdx ] }
                          | i -> i)
                        b.Mir_phys.Block.body
                    in
                    let body =
                      match !divisor with
                      | Some u1 ->
                          List.map
                            (function
                              | Mir_phys.Instr.Move ({ dst; _ } as m)
                                when Loc.equal dst u1 ->
                                  Mir_phys.Instr.Move { m with dst = rdx }
                              | i -> i)
                            body
                      | None -> body
                    in
                    { b with Mir_phys.Block.body })
                  fn.Mir_phys.Func.blocks;
            })
          phys.Mir_phys.Program.funcs;
    }
  in
  (match Err.payload (H.V.verify phys) with
  | Ok _ -> print_endline "original: accepted"
  | Error d -> Fmt.pr "original: %a@." Mir_diagnostic.pp d);
  (match Err.payload (H.V.verify edited) with
  | Ok _ -> print_endline "edited: accepted"
  | Error d -> Fmt.pr "edited: %a@." Mir_diagnostic.pp d);
  [%expect
    {|
    original: accepted
    edited: allocated fn0 bb28 i677: target constraint: an early-clobber result overlaps a use |}]

module Calls = Machine_alloc_test.Calls_test

let%expect_test "calls on x86-64: selected, allocated, realized" =
  let g = Result.get_ok (Err.payload (Mir_verify.generic Calls.program)) in
  let res = Result.get_ok (Err.payload (X64_select.program g)) in
  let record = H.record_region res in
  let sel = X64_stage.Sel.Verified.selected res.X64_select.selected in
  let phys = H.A.allocate res.X64_select.selected in
  let real = Result.get_ok (H.Fr.realize phys) in
  let checked p =
    match Err.payload (H.V.verify p) with
    | Error d -> Fmt.str "%a" Mir_diagnostic.pp d
    | Ok p -> (
        match Err.payload (H.C.check res.X64_select.selected p) with
        | Ok () -> "ok"
        | Error e -> Fmt.str "%a" Machine_check.Mir_checker.pp_error e)
  in
  Fmt.pr "checks: allocated %s, realized %s@." (checked phys) (checked real);
  List.iter
    (fun (x, n) ->
      let args = [ Mir_datum.f64 x; Mir_datum.Bits n ] in
      let memory = Mir_memory.create () in
      let binding =
        Result.get_ok
          (Mir_interp.instantiate sel.X64_stage.Sel.program memory
             ~bound:(fun _ -> None))
      in
      let s =
        Calls.staged ~record memory binding
          (X64_stage.Interp.run ~models:Calls.models res.X64_select.selected
             memory binding ~args)
            .X64_stage.Interp.outcome
      in
      let phys_run realized p =
        let memory = Mir_memory.create () in
        let binding =
          Result.get_ok
            (Mir_interp.instantiate (H.regions_program p) memory
               ~bound:(fun _ -> None))
        in
        Calls.staged ~record memory binding
          (H.P.run ~realized
             ~seed:(if realized then H.caller_state else fun _ -> ())
             ~models:Calls.models p memory binding ~args)
            .H.P.outcome
      in
      Fmt.pr "selected %s | allocated %s | realized %s@." s
        (phys_run false phys) (phys_run true real))
    [ (1.5, 3L); (0.1, 0L); (-2., -4L) ];
  [%expect
    {|
    checks: allocated ok, realized ok
    selected ok 0x1.6p+2 | allocated ok 0x1.6p+2 | realized ok 0x1.6p+2
    selected i64_division_by_zero | allocated i64_division_by_zero | realized i64_division_by_zero
    selected i64_division_overflow | allocated i64_division_overflow | realized i64_division_overflow |}]
