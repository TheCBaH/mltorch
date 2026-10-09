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
    aarch64 x * x + x, w=37: ok; 179 instructions; 1 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 399 instructions, 1 register moves, 0 stores, 0 reloads, 0 rematerialized
    aarch64 matmul 2x3x16: ok; 209 instructions; 14 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 422 instructions, 14 register moves, 0 stores, 0 reloads, 0 rematerialized
    aarch64 matmul 5x7x33: ok; 161 instructions; 14 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 5080 instructions, 92 register moves, 0 stores, 0 reloads, 0 rematerialized
    x86_64 x * x + x, w=37: ok; 181 instructions; 2 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 409 instructions, 6 register moves, 0 stores, 0 reloads, 0 rematerialized
    x86_64 matmul 2x3x16: ok; 212 instructions; 24 register moves, 11 stores, 11 loads, 0 slot moves, 0 rematerialized; 11 slots (176 bytes); executed 439 instructions, 38 register moves, 25 stores, 25 reloads, 0 rematerialized
    x86_64 matmul 5x7x33: ok; 157 instructions; 19 register moves, 4 stores, 4 loads, 0 slot moves, 2 rematerialized; 4 slots (52 bytes); executed 5355 instructions, 217 register moves, 40 stores, 40 reloads, 15 rematerialized
    aarch64, 3 registers x * x + x, w=37: ok; 179 instructions; 11 register moves, 49 stores, 49 loads, 0 slot moves, 1 rematerialized; 9 slots (144 bytes); executed 399 instructions, 21 register moves, 98 stores, 98 reloads, 2 rematerialized
    aarch64, 3 registers matmul 2x3x16: ok; 209 instructions; 28 register moves, 91 stores, 87 loads, 0 slot moves, 2 rematerialized; 28 slots (376 bytes); executed 422 instructions, 58 register moves, 195 stores, 192 reloads, 2 rematerialized
    aarch64, 3 registers matmul 5x7x33: ok; 161 instructions; 21 register moves, 68 stores, 59 loads, 0 slot moves, 5 rematerialized; 21 slots (256 bytes); executed 5080 instructions, 677 register moves, 2274 stores, 2245 reloads, 30 rematerialized
    x86_64, 3 registers x * x + x, w=37: ok; 181 instructions; 12 register moves, 50 stores, 50 loads, 0 slot moves, 1 rematerialized; 10 slots (148 bytes); executed 409 instructions, 26 register moves, 103 stores, 103 reloads, 2 rematerialized
    x86_64, 3 registers matmul 2x3x16: ok; 212 instructions; 30 register moves, 102 stores, 102 loads, 0 slot moves, 2 rematerialized; 29 slots (384 bytes); executed 439 instructions, 60 register moves, 224 stores, 227 reloads, 2 rematerialized
    x86_64, 3 registers matmul 5x7x33: ok; 157 instructions; 24 register moves, 72 stores, 67 loads, 0 slot moves, 7 rematerialized; 22 slots (260 bytes); executed 5355 instructions, 797 register moves, 2614 stores, 2700 reloads, 40 rematerialized |}]

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
        5355 instructions, 5355 profiled
          address addq: 145
          address imulq: 145
          address leaq: 40
          address movslq: 265
          compute addl: 125
          compute addps: 280
          compute addss: 35
          compute cmpl: 151
          compute cvt.sd: 5
          compute cvt.ss: 140
          compute cvtpd2ps: 560
          compute cvtps2pd: 80
          compute event: 105
          compute mov.i32: 4
          compute movd.from_gpr: 1
          compute movlhps: 280
          compute movq.low: 40
          compute movq.widen: 280
          compute mulps: 280
          compute mulss: 35
          compute pshufd: 40
          compute pshufd.splatps: 284
          decode cvt.sd: 140
          decode cvtps2pd: 560
          decode load.l: 140
          decode movd.from_gpr: 140
          decode movq.low: 280
          decode movups: 280
          decode pshufd: 280
          encode cvt.ss: 5
          encode cvtpd2ps: 80
          encode movd.to_gpr: 5
          encode movlhps: 40
          encode movq.widen: 40
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
    rows 1: aarch64 peak fpr 17, gpr 8; hot stores none, loads none; 0 helper calls; frame unrealized; hot spills none; ok; 120 instructions; 10 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 3913 instructions, 58 register moves, 0 stores, 0 reloads, 0 rematerialized
      x86_64 peak fpr 17, gpr 8; hot stores fpr 3, loads fpr 3; 0 helper calls; frame unrealized; hot spills depth 2: 6; innermost 0; ok; 120 instructions; 13 register moves, 3 stores, 3 loads, 0 slot moves, 0 rematerialized; 3 slots (48 bytes); executed 4197 instructions, 134 register moves, 24 stores, 24 reloads, 0 rematerialized
    rows 2: aarch64 peak fpr 25, gpr 9; hot stores none, loads none; 0 helper calls; frame unrealized; hot spills none; ok; 208 instructions; 14 register moves, 0 stores, 0 loads, 0 slot moves, 0 rematerialized; 0 slots (0 bytes); executed 3741 instructions, 46 register moves, 0 stores, 0 reloads, 0 rematerialized
      x86_64 peak fpr 25, gpr 9; hot stores fpr 11, loads fpr 11; 0 helper calls; frame unrealized; hot spills depth 2: 8, depth 3: 14; innermost 14; ok; 212 instructions; 24 register moves, 11 stores, 11 loads, 0 slot moves, 0 rematerialized; 11 slots (176 bytes); executed 3955 instructions, 280 register moves, 240 stores, 240 reloads, 0 rematerialized
    rows 4: aarch64 peak fpr 41, gpr 9; hot stores fpr 11, loads fpr 11; 0 helper calls; frame unrealized; hot spills depth 2: 8, depth 3: 14; innermost 14; ok; 380 instructions; 33 register moves, 11 stores, 11 loads, 0 slot moves, 0 rematerialized; 11 slots (176 bytes); executed 3591 instructions, 216 register moves, 120 stores, 120 reloads, 0 rematerialized
      x86_64 peak fpr 41, gpr 10; hot stores fpr 40, loads fpr 39; 0 helper calls; frame unrealized; hot spills depth 2: 25, depth 3: 54; innermost 54; ok; 392 instructions; 45 register moves, 40 stores, 39 loads, 0 slot moves, 0 rematerialized; 27 slots (432 bytes); executed 3826 instructions, 183 register moves, 458 stores, 458 reloads, 0 rematerialized |}]
