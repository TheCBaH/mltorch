open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures
module B = Ssa_ir.Ssa_builder
module F = Ssa_ir_test.Ssa_fixtures

(* C4 for AArch64: reference allocation (every value spilled), the checker,
   and physical interpretation agreeing with the selected route and the SSA
   oracle; then the allocation mutations. *)

let show s = Fmt.pr "%s@." s
let data_bind data = bind_data ~shape:(Loop_fixtures.shape_w 4) data
let row_buffer id n role = F.buffer id ~h:1L ~w:n Ssa_ir.Ssa_format.F32 role
let idx bld n = B.index bld (Int64.of_int n)

let store bld i x =
  B.store_f64 bld (F.buf 1) ~encode:Ssa_ir.Ssa_op.Encode.F32_round
    (F.at bld ~h:(idx bld 0) ~w:(idx bld i))
    x

(* swap and Fibonacci recurrences: back edges whose transfers form cycles *)
let recurrences n =
  F.build
    ~buffers:
      [
        row_buffer 0 4L Ssa_ir.Ssa_buffer.Input;
        row_buffer 1 4L Ssa_ir.Ssa_buffer.Output;
      ]
    (fun bld ->
      let (B.Cons (a, B.Cons (b, B.Nil))) =
        B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld n)
          ~init:(B.Cons (B.f64 bld 0., B.Cons (B.f64 bld 1., B.Nil)))
          (fun bld _ (B.Cons (a, B.Cons (b, B.Nil))) ->
            B.Cons (b, B.Cons (B.f64_binary bld Expr.Value.Add a b, B.Nil)))
      in
      store bld 0 a;
      store bld 1 b;
      let (B.Cons (x, B.Cons (y, B.Nil))) =
        B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld n)
          ~init:(B.Cons (B.f64 bld 1., B.Cons (B.f64 bld 2., B.Nil)))
          (fun _ _ (B.Cons (x, B.Cons (y, B.Nil))) ->
            B.Cons (y, B.Cons (x, B.Nil)))
      in
      store bld 2 x;
      store bld 3 y)

let inputs = [ (0, Ssa_ir.Ssa_memory.Floats [| 0.; 0.; 0.; 0. |]) ]

let%expect_test "allocated programs check and agree" =
  let zeros = data_bind [| 0.; 0.; 0.; 0. |] in
  show
    (Alloc_harness.plan Loop_programs.kernel
       ~bind:(data_bind [| -0.; 1.5; nan; 3. |]));
  show (Alloc_harness.plan Loop_programs.shifted_kernel ~bind:zeros);
  show (Alloc_harness.plan Loop_programs.overflow_kernel ~bind:zeros);
  List.iter
    (fun (m, k, n) ->
      let a = operand 3 (m * k) and b = operand 5 (k * n) in
      show
        (Alloc_harness.plan (matmul_kernel ~m ~k ~n)
           ~bind:(matmul_bind ~m ~k ~n ~a ~b)))
    [ (1, 1, 1); (3, 1, 4); (5, 7, 3) ];
  List.iter
    (fun n -> show (Alloc_harness.program (recurrences n) ~inputs))
    [ 0; 1; 5 ];
  [%expect
    {|
    ok
    coord_out_of_range(t0, W)
    index_overflow(mul)
    ok
    ok
    ok
    ok
    ok
    ok |}]

let%expect_test "allocation mutations are caught" =
  let open Machine_alloc.Mir_ref_alloc.Mutation in
  let m x = Some x in
  let a = operand 3 15 and b = operand 5 9 in
  let mm = matmul_kernel ~m:5 ~k:3 ~n:3
  and bind = matmul_bind ~m:5 ~k:3 ~n:3 ~a ~b in
  Fmt.pr "cycle scratch: %s@."
    (Alloc_harness.program ?mutation:(m Cycle_scratch) (recurrences 5) ~inputs);
  Fmt.pr "copy placement: %s@."
    (Alloc_harness.plan ?mutation:(m Copy_placement) mm ~bind);
  Fmt.pr "slot reuse: %s@."
    (Alloc_harness.plan ?mutation:(m Slot_reuse) mm ~bind);
  Fmt.pr "spill width: %s@."
    (Alloc_harness.plan ?mutation:(m Spill_width) mm ~bind);
  Fmt.pr "reload order: %s@."
    (Alloc_harness.plan ?mutation:(m Reload_order) mm ~bind);
  [%expect
    {|
    cycle scratch: rejected: checker: fn0 bb5: [slot13:8] does not hold %13
    copy placement: rejected: checker: fn0 bb8: [slot13:8] does not hold %11
    slot reuse: rejected: checker: fn0 bb1: [slot16:4] does not hold %16
    spill width: rejected: physical verifier: allocated fn0 bb1: target constraint: a location that does not fit its value
    reload order: rejected: checker: fn0 bb1: [slot129:8] does not hold %129 |}]

open Machine_ir
module Loc = Mir_phys.Loc

(* Edits of a correct allocation, each one defect. *)
let map_blocks f (p : (_, _) Mir_phys.Program.t) =
  {
    p with
    Mir_phys.Program.funcs =
      List.map
        (fun (fn : (_, _) Mir_phys.Func.t) ->
          { fn with Mir_phys.Func.blocks = List.map f fn.Mir_phys.Func.blocks })
        p.Mir_phys.Program.funcs;
  }

let first_block pick f =
  let hit = ref false in
  map_blocks (fun b ->
      if (not !hit) && pick b then (
        hit := true;
        f b)
      else b)

let is_reload = function
  | Mir_phys.Instr.Move { dst = Loc.Reg _; src = Loc.Slot _; _ } -> true
  | _ -> false

let reloads (b : (_, _) Mir_phys.Block.t) =
  List.length (List.filter is_reload b.Mir_phys.Block.body)

(* the first reload of a block with at least two distinct reloaded registers *)
let missing_reload =
  first_block
    (fun b -> reloads b >= 2)
    (fun b ->
      let dropped = ref false in
      {
        b with
        Mir_phys.Block.body =
          List.filter
            (fun i ->
              if (not !dropped) && is_reload i then (
                dropped := true;
                false)
              else true)
            b.Mir_phys.Block.body;
      })

(* a reload into the register of an earlier reload of the same shape *)
let register_overlap =
  let shape (r : Mir_target.View.t) =
    (r.Mir_target.View.bank, r.Mir_target.View.bits)
  in
  let regs (b : (_, _) Mir_phys.Block.t) =
    List.filter_map
      (function
        | Mir_phys.Instr.Move { dst = Loc.Reg r; src = Loc.Slot _; _ } -> Some r
        | _ -> None)
      b.Mir_phys.Block.body
  in
  let pair b =
    let rs = regs b in
    List.find_map
      (fun r ->
        List.find_opt
          (fun q -> shape q = shape r && not (Mir_target.View.equal q r))
          rs
        |> Option.map (fun q -> (r, q)))
      rs
  in
  first_block
    (fun b -> Option.is_some (pair b))
    (fun b ->
      let r0, q = Option.get (pair b) in
      {
        b with
        Mir_phys.Block.body =
          List.map
            (function
              | Mir_phys.Instr.Move
                  ({ dst = Loc.Reg r; src = Loc.Slot _; _ } as m)
                when Mir_target.View.equal r q ->
                  Mir_phys.Instr.Move { m with dst = Loc.Reg r0 }
              | i -> i)
            b.Mir_phys.Block.body;
      })

(* two block parameters of one type with their entry claims exchanged *)
let live_in =
  let same (b : (_, _) Mir_phys.Block.t) =
    let rec find = function
      | ((p : Mir_value.t), _) :: rest -> (
          match
            List.find_opt
              (fun ((q : Mir_value.t), _) ->
                Mir_type.equal p.Mir_value.ty q.Mir_value.ty)
              rest
          with
          | Some (q, _) -> Some (p, q)
          | None -> find rest)
      | [] -> None
    in
    find b.Mir_phys.Block.entry
  in
  first_block
    (fun b -> Option.is_some (same b))
    (fun b ->
      let p, q = Option.get (same b) in
      let at v = List.assoc v b.Mir_phys.Block.entry in
      let lp = at p and lq = at q in
      {
        b with
        Mir_phys.Block.entry =
          List.map
            (fun (v, l) ->
              if Mir_value.equal v p then (v, lq)
              else if Mir_value.equal v q then (v, lp)
              else (v, l))
            b.Mir_phys.Block.entry;
      })

(* the moves of every edge transfer dropped from one split block *)
let edge_state =
  first_block
    (fun b ->
      match b.Mir_phys.Block.origin with
      | Mir_phys.Origin.Edge _ -> true
      | _ -> false)
    (fun b -> { b with Mir_phys.Block.body = [] })

let%expect_test "hand edits of a correct allocation are caught" =
  let a = operand 3 15 and b = operand 5 9 in
  let mm = matmul_kernel ~m:5 ~k:3 ~n:3
  and bind = matmul_bind ~m:5 ~k:3 ~n:3 ~a ~b in
  Fmt.pr "missing reload: %s@."
    (Alloc_harness.plan ~edit:missing_reload mm ~bind);
  Fmt.pr "register overlap: %s@."
    (Alloc_harness.plan ~edit:register_overlap mm ~bind);
  Fmt.pr "live-in claim: %s@."
    (Alloc_harness.program ~edit:live_in (recurrences 5) ~inputs);
  Fmt.pr "incoming edge state: %s@."
    (Alloc_harness.program ~edit:edge_state (recurrences 5) ~inputs);
  (* a tied form (movk, from a constant of several halfwords) with its result
     in another register *)
  let eps = Float.ldexp 1. (-27) in
  let mixed_inputs =
    [ (0, Ssa_ir.Ssa_memory.Floats [| 1. +. eps; 1. -. eps; -1. |]) ]
  in
  Fmt.pr "tie kept: %s@."
    (Alloc_harness.program Machine_aarch64_test.A64_select_test.mixed
       ~inputs:mixed_inputs);
  Fmt.pr "tie broken: %s@."
    (Alloc_harness.program ~mutation:Machine_alloc.Mir_ref_alloc.Mutation.Tie
       Machine_aarch64_test.A64_select_test.mixed ~inputs:mixed_inputs);
  [%expect
    {|
    missing reload: rejected: checker: fn0 bb1: x9 does not hold %129
    register overlap: rejected: checker: fn0 bb2: w9 does not hold %2
    live-in claim: rejected: checker: fn0 bb1: [slot4:8] does not hold %19
    incoming edge state: rejected: checker: fn0 bb3: [slot6:8] does not hold %6
    tie kept: ok
    tie broken: rejected: physical verifier: allocated fn0 bb36 i1278: target constraint: a tied result not in its use's register |}]
