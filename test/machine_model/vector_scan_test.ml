(* M11.3: planned vector kernels through linear scan on both targets, full and
   three-register pools: checked, run on the physical interpreter against the
   plan's oracle, with what the vector values cost. *)

open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures
open Machine_ir
open Machine_interp
module Src = Machine_source_test.Mir_source

let case kernel ~bind =
  Result.get_ok
    (Src.case_of_planned ~target:Ssa_ir.Ssa_target.neon128
       ~numerics:Ssa_ir.Ssa_numerics.Simd_fp32_ordered
       (Fusion_plan.default kernel)
       ~bind)

let data n = Array.init n (fun i -> (float_of_int (i * 7 mod 11) -. 5.) /. 4.)

let kernels () =
  let x = Loop_fixtures.load_t0 in
  let shape = Loop_fixtures.shape_w 37 in
  [
    ( "x * x + x, w=37",
      case
        (Loop_fixtures.pixel_kernel ~shape Expr.Value.(add (mul x x) x))
        ~bind:(bind_data ~shape (data 37)) );
  ]
  @ List.map
      (fun (m, k, n) ->
        ( Fmt.str "matmul %dx%dx%d" m k n,
          case (matmul_kernel ~m ~k ~n)
            ~bind:
              (matmul_bind ~m ~k ~n
                 ~a:(operand 3 (m * k))
                 ~b:(operand 5 (k * n))) ))
      [ (2, 3, 16); (5, 7, 33) ]

module Run
    (T : Mir_sel_interp.SEMANTICS)
    (R : Machine_alloc.Mir_linear_scan.POOL)
    (X : sig
      val name : string

      val select :
        Mir_verify.Generic.t ->
        (Mir_sel.Make(T).Verified.t * Mir_id.View.t, string) result
    end) =
struct
  module Ls = Machine_alloc.Mir_linear_scan.Make (T) (R)
  module V = Mir_phys_verify.Make (T)
  module C = Machine_check.Mir_checker.Make (T)
  module P = Mir_phys_interp.Make (T)
  module S = Mir_sel.Make (T)

  let run ?mutation (c : Src.Case.t) =
    match X.select c.Src.Case.lowered.Machine_lower.Mir_lower.program with
    | Error e -> "refused: " ^ e
    | Ok (sel, record) -> (
        let phys = Ls.allocate ?mutation sel in
        let stats = Machine_alloc.Mir_alloc_stats.of_program phys in
        match Err.payload (V.verify phys) with
        | Error d -> Fmt.str "physical verifier: %a" Mir_diagnostic.pp d
        | Ok phys -> (
            match Err.payload (C.check sel phys) with
            | Error e ->
                Fmt.str "checker: %a" Machine_check.Mir_checker.pp_error e
            | Ok () ->
                let memory = Mir_memory.create () in
                let binding =
                  Result.get_ok
                    (Mir_interp.instantiate
                       (Machine_alloc_test.Alloc_harness.regions_program phys)
                       memory ~bound:c.Src.Case.bound)
                in
                let r =
                  P.run ~models:Mir_math_model.all phys memory binding ~args:[]
                in
                let program = (S.Verified.selected sel).S.program in
                let obs =
                  Src.selected_observation
                    ~layout:c.Src.Case.lowered.Machine_lower.Mir_lower.layout
                    ~sites:[||]
                    ~record:
                      (Option.get (Mir_program.find_view program record))
                        .Mir_view.region memory binding r.P.outcome r.P.events
                in
                Fmt.str "%s%s; %a; executed %a" (Src.status_name obs)
                  (match
                     Src.verdict ~expected:c.Src.Case.oracle
                       ~actual:{ Src.Route.name = X.name; observation = obs }
                   with
                  | None -> ""
                  | Some d -> " DISAGREE " ^ d)
                  Machine_alloc.Mir_alloc_stats.pp stats
                  Mir_phys_interp.Traffic.pp r.P.traffic))

  module Pr = Machine_alloc.Mir_pressure.Make (T)
  module Sch = Machine_alloc.Mir_schedule.Make (T)

  (* the production pipeline's pressure: sink scheduling, then linear scan *)
  let pressure (c : Src.Case.t) =
    match X.select c.Src.Case.lowered.Machine_lower.Mir_lower.program with
    | Error e -> "refused: " ^ e
    | Ok (sel, _) -> (
        match Sch.schedule Machine_alloc.Mir_schedule.Policy.Sink sel with
        | Error r -> Fmt.str "%a" Machine_alloc.Mir_schedule.Refusal.pp r
        | Ok s ->
            let r = Pr.report s (Ls.allocate s) in
            Fmt.str "%a; hot spills %a" Machine_alloc.Mir_pressure.pp r
              Machine_alloc.Mir_pressure.pp_depths
              (r.Machine_alloc.Mir_pressure.by_depth, r.innermost))

  let all () =
    List.iter
      (fun (name, c) -> Fmt.pr "%s %s: %s@." X.name name (run c))
      (kernels ())
end

let take n l = List.filteri (fun i _ -> i < n) l

module A64_select_ = struct
  let select g =
    match Err.payload (Machine_target_aarch64.A64_select.program g) with
    | Ok r ->
        Ok
          ( r.Machine_target_aarch64.A64_select.selected,
            r.Machine_target_aarch64.A64_select.record )
    | Error e ->
        Error (Fmt.str "%a" Machine_target_aarch64.A64_select.Refusal.pp e)
end

module X64_select_ = struct
  let select g =
    match Err.payload (Machine_target_x86_64.X64_select.program g) with
    | Ok r ->
        Ok
          ( r.Machine_target_x86_64.X64_select.selected,
            r.Machine_target_x86_64.X64_select.record )
    | Error e ->
        Error (Fmt.str "%a" Machine_target_x86_64.X64_select.Refusal.pp e)
end

module A64 =
  Run (Machine_target_aarch64.A64) (Machine_target_aarch64.A64_regs)
    (struct
      include A64_select_

      let name = "aarch64"
    end)

module X64 =
  Run (Machine_target_x86_64.X64) (Machine_target_x86_64.X64_regs)
    (struct
      include X64_select_

      let name = "x86_64"
    end)

module A64_small =
  Run
    (Machine_target_aarch64.A64)
    (struct
      include Machine_target_aarch64.A64_regs

      let allocatable bank = take 3 (allocatable bank)
    end)
    (struct
      include A64_select_

      let name = "aarch64, 3 registers"
    end)

module X64_small =
  Run
    (Machine_target_x86_64.X64)
    (struct
      include Machine_target_x86_64.X64_regs

      let allocatable bank = take 3 (allocatable bank)
    end)
    (struct
      include X64_select_

      let name = "x86_64, 3 registers"
    end)

let%expect_test "vector kernels through linear scan" =
  A64.all ();
  X64.all ();
  A64_small.all ();
  X64_small.all ();
  [%expect
    {|
    aarch64 x * x + x, w=37: ok; 49 instructions; 3 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 111 instructions, 3 register moves, 0 stores, 0 reloads, 0 rematerialized
    aarch64 matmul 2x3x16: ok; 65 instructions; 22 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 136 instructions, 22 register moves, 0 stores, 0 reloads, 0 rematerialized
    aarch64 matmul 5x7x33: ok; 62 instructions; 26 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 1821 instructions, 172 register moves, 0 stores, 0 reloads, 0 rematerialized
    x86_64 x * x + x, w=37: ok; 47 instructions; 4 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 117 instructions, 8 register moves, 0 stores, 0 reloads, 0 rematerialized
    x86_64 matmul 2x3x16: ok; 70 instructions; 30 register moves, 11 stores, 11 loads, 0 slot moves, 1 rematerialized; 11 slots (168 bytes); executed 157 instructions, 42 register moves, 23 stores, 23 reloads, 1 rematerialized
    x86_64 matmul 5x7x33: ok; 66 instructions; 29 register moves, 5 stores, 5 loads, 0 slot moves, 4 rematerialized; 4 slots (44 bytes); executed 2173 instructions, 192 register moves, 45 stores, 45 reloads, 30 rematerialized
    aarch64, 3 registers x * x + x, w=37: ok; 49 instructions; 8 register moves, 13 stores, 13 loads, 0 slot moves, 1 rematerialized; 5 slots (80 bytes); executed 111 instructions, 9 register moves, 26 stores, 26 reloads, 1 rematerialized
    aarch64, 3 registers matmul 2x3x16: ok; 65 instructions; 17 register moves, 57 stores, 53 loads, 0 slot moves, 3 rematerialized; 25 slots (336 bytes); executed 136 instructions, 27 register moves, 111 stores, 108 reloads, 3 rematerialized
    aarch64, 3 registers matmul 5x7x33: ok; 62 instructions; 18 register moves, 43 stores, 37 loads, 0 slot moves, 5 rematerialized; 19 slots (216 bytes); executed 1821 instructions, 505 register moves, 1095 stores, 1095 reloads, 26 rematerialized
    x86_64, 3 registers x * x + x, w=37: ok; 47 instructions; 9 register moves, 14 stores, 14 loads, 0 slot moves, 1 rematerialized; 6 slots (84 bytes); executed 117 instructions, 14 register moves, 31 stores, 31 reloads, 1 rematerialized
    x86_64, 3 registers matmul 2x3x16: ok; 70 instructions; 18 register moves, 62 stores, 61 loads, 0 slot moves, 4 rematerialized; 28 slots (380 bytes); executed 157 instructions, 34 register moves, 132 stores, 132 reloads, 4 rematerialized
    x86_64, 3 registers matmul 5x7x33: ok; 66 instructions; 19 register moves, 50 stores, 46 loads, 0 slot moves, 7 rematerialized; 21 slots (256 bytes); executed 2173 instructions, 437 register moves, 1495 stores, 1510 reloads, 36 rematerialized |}]

(* What a run executed, by origin role and mnemonic: every executed
   instruction once. *)
let%expect_test "executed instructions by role and operation" =
  let _, c = List.nth (kernels ()) 2 in
  match
    X64_select_.select c.Src.Case.lowered.Machine_lower.Mir_lower.program
  with
  | Error e -> print_endline e
  | Ok (sel, _) ->
      let phys =
        Result.get_ok (Err.payload (X64.V.verify (X64.Ls.allocate sel)))
      in
      let memory = Mir_memory.create () in
      let binding =
        Result.get_ok
          (Mir_interp.instantiate
             (Machine_alloc_test.Alloc_harness.regions_program phys)
             memory ~bound:c.Src.Case.bound)
      in
      let t =
        (X64.P.run ~models:Mir_math_model.all phys memory binding ~args:[])
          .X64.P.traffic
      in
      let module Tr = Mir_phys_interp.Traffic in
      Fmt.pr "%Ld instructions, %Ld profiled@." t.Tr.instructions
        (Tr.Ops.fold (fun _ n acc -> Int64.add n acc) t.Tr.ops 0L);
      Tr.Ops.iter (fun k n -> Fmt.pr "  %s: %Ld@." k n) t.Tr.ops;
      [%expect
        {|
        2173 instructions, 2173 profiled
          address leaq: 263
          compute addl: 125
          compute addps: 280
          compute addss: 35
          compute cmpl: 151
          compute event: 105
          compute mov.i32: 4
          compute movd.from_gpr: 1
          compute mulps: 280
          compute mulss: 35
          compute pshufd.splatps: 284
          decode load.l: 140
          decode movd.from_gpr: 140
          decode movups: 280
          encode movd.to_gpr: 5
          encode movups: 40
          encode store.l: 5 |}]

let%expect_test "a vector spilled to half its bytes is rejected" =
  let name, c = List.nth (kernels ()) 1 in
  List.iter
    (fun (label, f) -> Fmt.pr "%s %s: %s@." label name (f c))
    [
      ( "aarch64, 3 registers",
        A64_small.run
          ~mutation:Machine_alloc.Mir_linear_scan.Mutation.Half_spill );
      ( "x86_64, 3 registers",
        X64_small.run
          ~mutation:Machine_alloc.Mir_linear_scan.Mutation.Half_spill );
    ];
  [%expect
    {|
    aarch64, 3 registers matmul 2x3x16: physical verifier: allocated fn0 bb1: target constraint: a location that does not fit its value
    x86_64, 3 registers matmul 2x3x16: physical verifier: allocated fn0 bb1: target constraint: a location that does not fit its value |}]

(* The vector planner's row blocking: each blocked row holds its own
   accumulators, so the factor is bounded by registers, not lanes. *)
let%expect_test "row blocking under production allocation" =
  let m, k, n = (4, 8, 32) in
  List.iter
    (fun rows ->
      let target =
        Ssa_ir.Ssa_target.with_row_block rows Ssa_ir.Ssa_target.neon128
      in
      let c =
        Result.get_ok
          (Src.case_of_planned ~target
             ~numerics:Ssa_ir.Ssa_numerics.Simd_fp32_ordered
             (Fusion_plan.default (matmul_kernel ~m ~k ~n))
             ~bind:
               (matmul_bind ~m ~k ~n
                  ~a:(operand 3 (m * k))
                  ~b:(operand 5 (k * n))))
      in
      Fmt.pr "rows %d: aarch64 %s; %s@.  x86_64 %s; %s@." rows (A64.pressure c)
        (A64.run c) (X64.pressure c) (X64.run c))
    [ 1; 2; 4 ];
  [%expect
    {|
    rows 1: aarch64 peak fpr 13, gpr 11; hot stores none, loads none; 0 helper calls; frame unrealized; hot spills none; ok; 45 instructions; 18 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 1323 instructions, 106 register moves, 0 stores, 0 reloads, 0 rematerialized
      x86_64 peak fpr 16, gpr 12; hot stores fpr 2, gpr 1, loads fpr 2, gpr 1; 0 helper calls; frame unrealized; hot spills depth 2: 6; innermost 0; ok; 46 instructions; 21 register moves, 3 stores, 3 loads, 0 slot moves, 1 rematerialized; 3 slots (40 bytes); executed 1576 instructions, 182 register moves, 24 stores, 24 reloads, 8 rematerialized
    rows 2: aarch64 peak fpr 21, gpr 11; hot stores none, loads none; 0 helper calls; frame unrealized; hot spills none; ok; 65 instructions; 22 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 1169 instructions, 70 register moves, 0 stores, 0 reloads, 0 rematerialized
      x86_64 peak fpr 24, gpr 12; hot stores fpr 10, gpr 1, loads fpr 10, gpr 1; 0 helper calls; frame unrealized; hot spills depth 2: 10, depth 3: 12; innermost 12; ok; 70 instructions; 30 register moves, 11 stores, 11 loads, 0 slot moves, 1 rematerialized; 11 slots (168 bytes); executed 1422 instructions, 268 register moves, 212 stores, 212 reloads, 4 rematerialized
    rows 4: aarch64 peak fpr 37, gpr 11; hot stores fpr 7, loads fpr 7; 0 helper calls; frame unrealized; hot spills depth 2: 8, depth 3: 6; innermost 6; ok; 105 instructions; 40 register moves, 7 stores, 7 loads, 0 slot moves, 0 rematerialized; 7 slots (112 bytes); executed 1092 instructions, 167 register moves, 56 stores, 56 reloads, 0 rematerialized
      x86_64 peak fpr 40, gpr 12; hot stores fpr 35, gpr 1, loads fpr 34, gpr 1; 0 helper calls; frame unrealized; hot spills depth 2: 19, depth 3: 52; innermost 52; ok; 118 instructions; 56 register moves, 36 stores, 35 loads, 0 slot moves, 1 rematerialized; 27 slots (424 bytes); executed 1345 instructions, 257 register moves, 436 stores, 436 reloads, 2 rematerialized |}]
