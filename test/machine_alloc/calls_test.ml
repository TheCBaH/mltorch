open Machine_ir
open Machine_interp
open Machine_target_aarch64
module B = Mir_builder
module Loc = Mir_phys.Loc
module H = Alloc_harness

(* Calls end to end: a pure helper, a helper and an internal function that may
   fail, through the generic, selected and allocated routes; the failure record
   a callee stores reaches the caller untouched; and a value kept in a
   register across a call survives only where AAPCS64 preserves those bits. *)

let i64 = Mir_type.i64
let f64 = Mir_type.F64
let fn0 = Mir_id.Func.of_int 0
let fn1 = Mir_id.Func.of_int 1

let scale =
  {
    Mir_helper.id = Mir_id.Helper.of_int 0;
    name = "scale";
    version = 1;
    params = [ f64 ];
    results = [ f64 ];
    effects = Mir_helper.Effect.Pure;
    failures = [];
  }

let guard =
  {
    Mir_helper.id = Mir_id.Helper.of_int 1;
    name = "guard";
    version = 1;
    params = [ i64 ];
    results = [];
    effects = Mir_helper.Effect.Pure;
    failures = [ Mir_failure.I64_division_overflow ];
  }

let models =
  [
    {
      Mir_helper_model.name = "scale";
      version = 1;
      run =
        (fun _ -> function
          | [ Mir_datum.Bits b ] ->
              Mir_helper_model.Returns
                [
                  Mir_datum.f64
                    (Sys.opaque_identity (Int64.float_of_bits b *. 2.) +. 1.);
                ]
          | _ -> assert false);
    };
    {
      Mir_helper_model.name = "guard";
      version = 1;
      run =
        (fun _ -> function
          | [ Mir_datum.Bits n ] ->
              if Int64.compare n 0L < 0 then
                Mir_helper_model.Fails (Mir_failure.I64_division_overflow, [])
              else Mir_helper_model.Returns []
          | _ -> assert false);
    };
  ]

let signature = function
  | Mir_op.Callee.Helper id when Mir_id.Helper.equal id scale.Mir_helper.id ->
      Some { Mir_typing.Signature.params = [ f64 ]; results = [ f64 ] }
  | Mir_op.Callee.Helper _ ->
      Some { Mir_typing.Signature.params = [ i64 ]; results = [] }
  | Mir_op.Callee.Func _ ->
      Some { Mir_typing.Signature.params = [ i64 ]; results = [] }

(* check(n): fails with a zero divisor when n = 0 *)
let check =
  let bld = B.create () in
  let e = B.new_block bld [ i64 ] in
  let bad = B.new_block bld [] and ok = B.new_block bld [] in
  let n = List.hd (B.param e) in
  let z =
    B.emit bld e
      (Mir_op.Icmp
         (Mir_op.Icmp.Eq, n, B.emit bld e (Mir_op.Const (Mir_const.i64 0L))))
  in
  B.branch e z (bad, []) (ok, []);
  B.fail bad Mir_failure.I64_division_by_zero [];
  B.return ok [];
  B.func bld ~id:fn1 ~name:"check" ~entry:e ~results:[]

(* main(x, n): out <- scale(x) + x; guard(n); check(n) *)
let main =
  let bld = B.create () in
  let e = B.new_block bld [ f64; i64 ] in
  let x, n = match B.param e with [ x; n ] -> (x, n) | _ -> assert false in
  let call c args =
    Result.get_ok (B.op bld e ~signature (Mir_op.Call (c, args)))
  in
  let y = List.hd (call (Mir_op.Callee.Helper scale.Mir_helper.id) [ x ]) in
  let z = B.emit bld e (Mir_op.Fbinary (Mir_op.Fbinary.Add, y, x)) in
  let addr = B.emit bld e (Mir_op.Addr (Mir_id.View.of_int 0)) in
  B.emit_unit bld e
    (Mir_op.Store
       ( { Mir_op.Access.width = Mir_width.W64; addr; align = 8L },
         B.emit bld e (Mir_op.Bitcast (i64, z)) ));
  ignore (call (Mir_op.Callee.Helper guard.Mir_helper.id) [ n ]);
  ignore (call (Mir_op.Callee.Func fn1) [ n ]);
  B.return e [];
  B.func bld ~id:fn0 ~name:"main" ~entry:e ~results:[]

let program =
  B.program
    ~regions:
      [
        {
          Mir_region.id = Mir_id.Region.of_int 0;
          size = 8L;
          align = 8L;
          init = Mir_region.Bound;
        };
      ]
    ~views:
      [
        {
          Mir_view.id = Mir_id.View.of_int 0;
          region = Mir_id.Region.of_int 0;
          offset = 0L;
          size = 8L;
          perm = Mir_view.Read_write;
          role = Mir_view.Output;
          source = None;
        };
      ]
    ~helpers:[ scale; guard ] [ main; check ] ~main:fn0

let out_bytes memory binding =
  let key =
    Option.get (Mir_interp.Binding.instance binding (Mir_id.Region.of_int 0))
  in
  match
    Mir_memory.load memory
      (Mir_memory.pointer memory key ~lo:0L ~hi:8L)
      ~bytes:8L ~align:8L
  with
  | Ok b -> Fmt.str "%h" (Int64.float_of_bits b)
  | Error _ -> "undefined"

(* A selected or allocated run's status, its stored record, and the output. *)
let staged ~record memory binding (o : Mir_interp.Outcome.t) =
  match o with
  | Mir_interp.Outcome.Success vs -> (
      match List.rev vs with
      | Mir_datum.Bits 0L :: _ -> "ok " ^ out_bytes memory binding
      | Mir_datum.Bits _ :: _ -> (
          match
            Machine_source_test.Mir_source.record_row memory
              (Option.get (Mir_interp.Binding.instance binding record))
              ~sites:[||]
          with
          | Mir_observation.Status.Failure r ->
              Fmt.str "%a" Mir_failure.pp r.Mir_observation.Row.failure
          | _ -> "bad record")
      | _ -> "bad status")
  | o -> Fmt.str "%a" Mir_interp.Outcome.pp o

let routes ?edit (x, n) =
  let args = [ Mir_datum.f64 x; Mir_datum.Bits n ] in
  let g = Result.get_ok (Err.payload (Mir_verify.generic program)) in
  let memory = Mir_memory.create () in
  let binding =
    Result.get_ok (Mir_interp.instantiate program memory ~bound:(fun _ -> None))
  in
  let generic =
    match
      (Mir_interp.run ~models g memory binding ~args).Mir_interp.outcome
    with
    | Mir_interp.Outcome.Success _ -> "ok " ^ out_bytes memory binding
    | Mir_interp.Outcome.Failure r ->
        Fmt.str "%a" Mir_failure.pp r.Mir_observation.Row.failure
    | o -> Fmt.str "%a" Mir_interp.Outcome.pp o
  in
  let res = Result.get_ok (Err.payload (A64_select.program g)) in
  let sel = A64_stage.Sel.Verified.selected res.A64_select.selected in
  let record =
    (Option.get
       (Mir_program.find_view sel.A64_stage.Sel.program res.A64_select.record))
      .Mir_view.region
  in
  let memory = Mir_memory.create () in
  let binding =
    Result.get_ok
      (Mir_interp.instantiate sel.A64_stage.Sel.program memory ~bound:(fun _ ->
           None))
  in
  let selected =
    staged ~record memory binding
      (A64_stage.Interp.run ~models res.A64_select.selected memory binding ~args)
        .A64_stage.Interp.outcome
  in
  let allocated =
    match H.allocate ?edit res with
    | Error e -> "rejected: " ^ e
    | Ok phys ->
        let memory = Mir_memory.create () in
        let binding =
          Result.get_ok
            (Mir_interp.instantiate (H.regions_program phys) memory
               ~bound:(fun _ -> None))
        in
        staged ~record memory binding
          (H.P.run ~models phys memory binding ~args).H.P.outcome
  in
  Fmt.pr "generic %s | selected %s | allocated %s@." generic selected allocated

let%expect_test "calls: helpers, a fallible function, propagated records" =
  List.iter routes [ (1.5, 3L); (0.1, 0L); (-2., -4L) ];
  [%expect
    {|
    generic ok 0x1.6p+2 | selected ok 0x1.6p+2 | allocated ok 0x1.6p+2
    generic i64_division_by_zero | selected i64_division_by_zero | allocated i64_division_by_zero
    generic i64_division_overflow | selected i64_division_overflow | allocated i64_division_overflow |}]

(* The allocation edited to keep [x] (the f64 parameter, used after the first
   call) in [keep] across that call instead of reloading it from its slot. *)
let keep_across_call keep (p : (_, _) Mir_phys.Program.t) =
  let x = { Mir_value.id = Mir_id.Value.of_int 0; ty = f64 } in
  let main =
    List.find
      (fun (f : (_, _) Mir_phys.Func.t) ->
        Mir_id.Func.equal f.Mir_phys.Func.id fn0)
      p.Mir_phys.Program.funcs
  in
  let slot =
    List.find_map
      (function
        | Mir_phys.Instr.Move { dst = Loc.Slot _ as s; value; _ }
          when Mir_value.equal value x ->
            Some s
        | _ -> None)
      (List.concat_map
         (fun (b : (_, _) Mir_phys.Block.t) -> b.Mir_phys.Block.body)
         main.Mir_phys.Func.blocks)
    |> Option.get
  in
  let seen_call = ref false in
  let edit_body body =
    List.concat_map
      (fun i ->
        match i with
        | Mir_phys.Instr.Exec
            {
              instr = { Mir_instr.op = Mir_sel.Op.Machine (A64_op.Bl _); _ };
              _;
            }
          when not !seen_call ->
            seen_call := true;
            [
              Mir_phys.Instr.Move { dst = Loc.Reg keep; src = slot; value = x };
              i;
            ]
        | Mir_phys.Instr.Move ({ src = Loc.Slot _; value; _ } as m)
          when !seen_call && Mir_value.equal value x ->
            [ Mir_phys.Instr.Move { m with src = Loc.Reg keep } ]
        | i -> [ i ])
      body
  in
  {
    p with
    Mir_phys.Program.funcs =
      List.map
        (fun (f : (_, _) Mir_phys.Func.t) ->
          if Mir_id.Func.equal f.Mir_phys.Func.id fn0 then
            {
              f with
              Mir_phys.Func.blocks =
                List.map
                  (fun (b : (_, _) Mir_phys.Block.t) ->
                    {
                      b with
                      Mir_phys.Block.body = edit_body b.Mir_phys.Block.body;
                    })
                  f.Mir_phys.Func.blocks;
            }
          else f)
        p.Mir_phys.Program.funcs;
  }

let%expect_test "a value across a call: preserved bits only" =
  (* d8's low half survives a call; d16 and x... are caller-saved *)
  routes ~edit:(keep_across_call (A64_reg.d 8)) (1.5, 3L);
  routes ~edit:(keep_across_call (A64_reg.d 16)) (1.5, 3L);
  [%expect
    {|
    generic ok 0x1.6p+2 | selected ok 0x1.6p+2 | allocated ok 0x1.6p+2
    generic ok 0x1.6p+2 | selected ok 0x1.6p+2 | allocated rejected: checker: fn0 bb0: d16 does not hold %0 |}]

let%expect_test "the physical interpreter alone catches the clobber" =
  (* bypassing the checker: the run reads undefined bits *)
  let g = Result.get_ok (Err.payload (Mir_verify.generic program)) in
  let res = Result.get_ok (Err.payload (A64_select.program g)) in
  let phys =
    keep_across_call (A64_reg.d 16) (H.A.allocate res.A64_select.selected)
  in
  let memory = Mir_memory.create () in
  let binding =
    Result.get_ok
      (Mir_interp.instantiate (H.regions_program phys) memory ~bound:(fun _ ->
           None))
  in
  Fmt.pr "%a@." Mir_interp.Outcome.pp
    (H.P.run ~models phys memory binding
       ~args:[ Mir_datum.f64 1.5; Mir_datum.Bits 3L ])
      .H.P.outcome;
  [%expect {| defect uninitialized at fn0 bb0 |}]

(* the callee [check] copies its argument into x19 *)
let writes_x19 (p : (_, _) Mir_phys.Program.t) =
  {
    p with
    Mir_phys.Program.funcs =
      List.map
        (fun (f : (_, _) Mir_phys.Func.t) ->
          if Mir_id.Func.equal f.Mir_phys.Func.id fn1 then
            let n, l = List.hd f.Mir_phys.Func.params in
            {
              f with
              Mir_phys.Func.blocks =
                List.map
                  (fun (b : (_, _) Mir_phys.Block.t) ->
                    if
                      Mir_id.Block.equal b.Mir_phys.Block.id
                        f.Mir_phys.Func.entry
                    then
                      {
                        b with
                        Mir_phys.Block.body =
                          Mir_phys.Instr.Move
                            { dst = Loc.Reg (A64_reg.x 19); src = l; value = n }
                          :: b.Mir_phys.Block.body;
                      }
                    else b)
                  f.Mir_phys.Func.blocks;
            }
          else f)
        p.Mir_phys.Program.funcs;
  }

(* Callee-saved registers seeded with patterns: a correct allocation leaves
   them as they were on every exit; a callee that writes x19 is caught. *)
let%expect_test "preserved state on every exit" =
  let g = Result.get_ok (Err.payload (Mir_verify.generic program)) in
  let res = Result.get_ok (Err.payload (A64_select.program g)) in
  let seed regs =
    List.iter
      (fun k ->
        H.P.seed regs (A64_reg.x k) ~lo:(Int64.of_int (0x1900 + k)) ~hi:0L)
      (A64_reg.range 19 28);
    List.iter
      (fun k ->
        H.P.seed regs (A64_reg.q k)
          ~lo:(Int64.of_int (0x800 + k))
          ~hi:(Int64.of_int (0x8000 + k)))
      (A64_reg.range 8 15)
  in
  let run phys =
    let memory = Mir_memory.create () in
    let binding =
      Result.get_ok
        (Mir_interp.instantiate (H.regions_program phys) memory ~bound:(fun _ ->
             None))
    in
    Fmt.pr "%a@." Mir_interp.Outcome.pp
      (H.P.run ~models ~seed phys memory binding
         ~args:[ Mir_datum.f64 1.5; Mir_datum.Bits 3L ])
        .H.P.outcome
  in
  let phys = H.A.allocate res.A64_select.selected in
  run phys;
  run (writes_x19 phys);
  [%expect {|
    success [0x0]
    defect preserved_state at fn1 bb2 |}]
