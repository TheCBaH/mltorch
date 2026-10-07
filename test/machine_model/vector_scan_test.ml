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
                Fmt.str "%s%s; %a" (Src.status_name obs)
                  (match
                     Src.verdict ~expected:c.Src.Case.oracle
                       ~actual:{ Src.Route.name = X.name; observation = obs }
                   with
                  | None -> ""
                  | Some d -> " DISAGREE " ^ d)
                  Machine_alloc.Mir_alloc_stats.pp stats))

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
            Fmt.str "%a" Machine_alloc.Mir_pressure.pp
              (Pr.report s (Ls.allocate s)))

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
    aarch64 x * x + x, w=37: ok; 360 instructions; 19 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 matmul 2x3x16: ok; 395 instructions; 30 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    aarch64 matmul 5x7x33: ok; 319 instructions; 22 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
    x86_64 x * x + x, w=37: ok; 356 instructions; 117 register moves, 6 stores, 6 loads, 0 slot moves, 3 rematerialized; 3 slots (48 bytes)
    x86_64 matmul 2x3x16: ok; 392 instructions; 155 register moves, 31 stores, 31 loads, 0 slot moves, 7 rematerialized; 19 slots (268 bytes)
    x86_64 matmul 5x7x33: ok; 318 instructions; 120 register moves, 25 stores, 25 loads, 0 slot moves, 11 rematerialized; 11 slots (140 bytes)
    aarch64, 3 registers x * x + x, w=37: ok; 360 instructions; 22 register moves, 59 stores, 59 loads, 0 slot moves, 27 rematerialized; 11 slots (164 bytes)
    aarch64, 3 registers matmul 2x3x16: ok; 395 instructions; 31 register moves, 104 stores, 106 loads, 0 slot moves, 27 rematerialized; 27 slots (372 bytes)
    aarch64, 3 registers matmul 5x7x33: ok; 319 instructions; 19 register moves, 67 stores, 67 loads, 0 slot moves, 32 rematerialized; 19 slots (244 bytes)
    x86_64, 3 registers x * x + x, w=37: ok; 356 instructions; 119 register moves, 95 stores, 93 loads, 0 slot moves, 27 rematerialized; 12 slots (172 bytes)
    x86_64, 3 registers matmul 2x3x16: ok; 392 instructions; 134 register moves, 140 stores, 140 loads, 0 slot moves, 27 rematerialized; 28 slots (380 bytes)
    x86_64, 3 registers matmul 5x7x33: ok; 318 instructions; 113 register moves, 102 stores, 98 loads, 0 slot moves, 32 rematerialized; 20 slots (252 bytes) |}]

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
    rows 1: aarch64 peak fpr 17, gpr 9; hot stores none, loads none; 0 helper calls; frame unrealized; ok; 211 instructions; 18 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      x86_64 peak fpr 17, gpr 10; hot stores fpr 11, gpr 4, loads fpr 11, gpr 3; 0 helper calls; frame unrealized; ok; 211 instructions; 75 register moves, 17 stores, 17 loads, 0 slot moves, 6 rematerialized; 10 slots (136 bytes)
    rows 2: aarch64 peak fpr 25, gpr 11; hot stores none, loads none; 0 helper calls; frame unrealized; ok; 395 instructions; 30 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes)
      x86_64 peak fpr 25, gpr 12; hot stores fpr 21, gpr 8, loads fpr 21, gpr 7; 0 helper calls; frame unrealized; ok; 392 instructions; 155 register moves, 31 stores, 31 loads, 0 slot moves, 7 rematerialized; 19 slots (268 bytes)
    rows 4: aarch64 peak fpr 41, gpr 15; hot stores fpr 16, loads fpr 16; 0 helper calls; frame unrealized; ok; 763 instructions; 70 register moves, 16 stores, 16 loads, 0 slot moves, 0 rematerialized; 16 slots (256 bytes)
      x86_64 peak fpr 41, gpr 16; hot stores fpr 58, gpr 17, loads fpr 58, gpr 14; 0 helper calls; frame unrealized; ok; 754 instructions; 268 register moves, 77 stores, 75 loads, 0 slot moves, 10 rematerialized; 37 slots (532 bytes) |}]
