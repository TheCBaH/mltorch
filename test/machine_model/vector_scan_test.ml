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
    aarch64 x * x + x, w=37: ok; 240 instructions; 17 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 567 instructions, 33 register moves, 0 stores, 0 reloads, 0 rematerialized
    aarch64 matmul 2x3x16: ok; 299 instructions; 30 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 635 instructions, 46 register moves, 0 stores, 0 reloads, 0 rematerialized
    aarch64 matmul 5x7x33: ok; 223 instructions; 22 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 8144 instructions, 412 register moves, 0 stores, 0 reloads, 0 rematerialized
    x86_64 x * x + x, w=37: ok; 236 instructions; 53 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 558 instructions, 126 register moves, 0 stores, 0 reloads, 0 rematerialized
    x86_64 matmul 2x3x16: ok; 296 instructions; 95 register moves, 12 stores, 12 loads, 0 slot moves, 3 rematerialized; 12 slots (192 bytes); executed 629 instructions, 207 register moves, 30 stores, 30 reloads, 5 rematerialized
    x86_64 matmul 5x7x33: ok; 222 instructions; 73 register moves, 5 stores, 5 loads, 0 slot moves, 3 rematerialized; 5 slots (68 bytes); executed 8070 instructions, 2657 register moves, 350 stores, 350 reloads, 50 rematerialized
    aarch64, 3 registers x * x + x, w=37: ok; 240 instructions; 23 register moves, 56 stores, 56 loads, 0 slot moves, 3 rematerialized; 11 slots (164 bytes); executed 567 instructions, 45 register moves, 113 stores, 112 reloads, 9 rematerialized
    aarch64, 3 registers matmul 2x3x16: ok; 299 instructions; 29 register moves, 105 stores, 103 loads, 0 slot moves, 8 rematerialized; 27 slots (376 bytes); executed 635 instructions, 57 register moves, 231 stores, 230 reloads, 12 rematerialized
    aarch64, 3 registers matmul 5x7x33: ok; 223 instructions; 18 register moves, 65 stores, 60 loads, 0 slot moves, 11 rematerialized; 18 slots (240 bytes); executed 8144 instructions, 740 register moves, 2570 stores, 2571 reloads, 182 rematerialized
    x86_64, 3 registers x * x + x, w=37: ok; 236 instructions; 59 register moves, 70 stores, 70 loads, 0 slot moves, 3 rematerialized; 12 slots (172 bytes); executed 558 instructions, 138 register moves, 153 stores, 152 reloads, 9 rematerialized
    x86_64, 3 registers matmul 2x3x16: ok; 296 instructions; 83 register moves, 118 stores, 116 loads, 0 slot moves, 7 rematerialized; 28 slots (380 bytes); executed 629 instructions, 183 register moves, 262 stores, 261 reloads, 12 rematerialized
    x86_64, 3 registers matmul 5x7x33: ok; 222 instructions; 62 register moves, 77 stores, 72 loads, 0 slot moves, 11 rematerialized; 20 slots (252 bytes); executed 8070 instructions, 2465 register moves, 3055 stores, 3056 reloads, 182 rematerialized |}]

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
    rows 1: aarch64 peak fpr 17, gpr 9; hot stores none, loads none; 0 helper calls; frame unrealized; hot spills none; ok; 163 instructions; 18 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 6041 instructions, 346 register moves, 0 stores, 0 reloads, 0 rematerialized
      x86_64 peak fpr 17, gpr 10; hot stores fpr 4, loads fpr 4; 0 helper calls; frame unrealized; hot spills depth 3: 8; innermost 8; ok; 163 instructions; 52 register moves, 4 stores, 4 loads, 0 slot moves, 1 rematerialized; 4 slots (64 bytes); executed 5994 instructions, 2014 register moves, 256 stores, 256 reloads, 64 rematerialized
    rows 2: aarch64 peak fpr 25, gpr 11; hot stores none, loads none; 0 helper calls; frame unrealized; hot spills none; ok; 299 instructions; 30 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 5772 instructions, 334 register moves, 0 stores, 0 reloads, 0 rematerialized
      x86_64 peak fpr 25, gpr 12; hot stores fpr 12, loads fpr 12; 0 helper calls; frame unrealized; hot spills depth 2: 6, depth 3: 18; innermost 18; ok; 296 instructions; 95 register moves, 12 stores, 12 loads, 0 slot moves, 3 rematerialized; 12 slots (192 bytes); executed 5681 instructions, 1934 register moves, 300 stores, 300 reloads, 40 rematerialized
    rows 4: aarch64 peak fpr 41, gpr 15; hot stores fpr 12, loads fpr 12; 0 helper calls; frame unrealized; hot spills depth 3: 24; innermost 24; ok; 571 instructions; 66 register moves, 12 stores, 12 loads, 0 slot moves, 0 rematerialized; 12 slots (192 bytes); executed 5639 instructions, 520 register moves, 192 stores, 192 reloads, 0 rematerialized
      x86_64 peak fpr 41, gpr 16; hot stores fpr 43, loads fpr 42; 0 helper calls; frame unrealized; hot spills depth 2: 27, depth 3: 58; innermost 58; ok; 562 instructions; 190 register moves, 43 stores, 42 loads, 0 slot moves, 9 rematerialized; 28 slots (448 bytes); executed 5526 instructions, 1856 register moves, 492 stores, 492 reloads, 42 rematerialized |}]
