(* The kernels' register effects against the Coral runtime's calling
   convention: a kernel returns to the runtime, so any callee-saved register it
   writes must be one its prologue stores and its epilogue restores. *)

open Machine_rivet_riscv32
module K = Peak_kernels
module Insn = Rv32_kernel_insn
module Item = Rv32_kernel_item
module Gpr = Rv32_kernel_reg.Gpr
module Peak = Peak_baseline

let callee_saved = Gpr.[ s0; s1; s2; s3; s4; s5; s6; s7; s8; s9; s10; s11 ]

let insns case =
  match (Peak.kernel case).K.items () |> Err.payload with
  | Ok items ->
      List.filter_map (function Item.Insn i -> Some i | _ -> None) items
  | Error e -> Fmt.failwith "%a" K.pp_error e

let written_callee_saved insns =
  List.filter
    (fun r ->
      List.exists
        (fun i -> List.exists (Insn.Reg_ref.equal (Gpr r)) (Insn.writes i))
        insns)
    callee_saved

(* What the prologue stores below the stack pointer. *)
let stored insns =
  List.filter_map
    (function
      | Insn.Sw { src; base; _ } when Gpr.equal base Gpr.sp -> Some src
      | _ -> None)
    insns

let reloaded insns =
  List.filter_map
    (function
      | Insn.Lw { dst; base; _ } when Gpr.equal base Gpr.sp -> Some dst
      | _ -> None)
    insns

let names rs = String.concat " " (List.map (Fmt.str "%a" Gpr.pp) rs)

let%expect_test "a callee-saved register a kernel writes is stored and restored"
    =
  List.iter
    (fun case ->
      let insns = insns case in
      let written = written_callee_saved insns in
      let stored = stored insns and reloaded = reloaded insns in
      let covered rs =
        List.for_all (fun r -> List.exists (Gpr.equal r) rs) written
      in
      if not (covered stored && covered reloaded) then
        Fmt.pr "%s writes [%s], stores [%s], restores [%s]@." (Peak.name case)
          (names written) (names stored) (names reloaded)
      else Fmt.pr "%s: [%s]@." (Peak.name case) (names written))
    Peak.cases;
  [%expect
    {|
    i8add: []
    i8mul: []
    i8macc: []
    i8wmacc: []
    i16add: []
    i16mul: []
    i16macc: []
    i16wmacc: []
    i32add: []
    i32mul: []
    i32macc: []
    f32add: []
    f32mul: []
    f32macc: []
    bf16wmacc: []
    f32macc_m1: []
    gemm_i8: [s0 s1]
    gemm_i16: [s0 s1]
    gemm_i32: [s0 s1]
    gemm_f32: [s0 s1]
    gemm_bf16: [s0 s1]
    gemm_i8_w48: [s0 s1]
    gemm_i8_w128: [s0 s1 s2 s3]
    gemm_i16_w24: [s0 s1]
    gemm_i16_w64: [s0 s1 s2 s3]
    gemm_i32_w24: [s0 s1]
    gemm_i32_w64: [s0 s1 s2 s3]
    gemm_f32_w24: [s0 s1]
    gemm_f32_w64: [s0 s1 s2 s3]
    gemm_bf16_w24: [s0 s1]
    gemm_bf16_w64: [s0 s1 s2 s3] |}]

let%expect_test "a VLMAX request is visible as a write to its register" =
  let vlmax =
    Insn.Vsetvli
      {
        avl = Rv32_kernel_vtype.Avl.Vlmax { vl = Gpr.t0 };
        vtype =
          {
            Rv32_kernel_vtype.sew = E8;
            lmul = M4;
            tail = Agnostic;
            mask = Agnostic;
          };
      }
  in
  Fmt.pr "%a@." Fmt.(list ~sep:comma Insn.Reg_ref.pp) (Insn.writes vlmax);
  Fmt.pr "elementwise writes t0: %b@."
    (List.exists
       (fun i -> List.exists (Insn.Reg_ref.equal (Gpr Gpr.t0)) (Insn.writes i))
       (insns (List.hd Peak.cases)));
  Fmt.pr "gemm_i16 (A in t0-t2) never sets vl into t0: %b@."
    (List.for_all
       (function
         | Insn.Vsetvli _ as i ->
             not (List.exists (Insn.Reg_ref.equal (Gpr Gpr.t0)) (Insn.writes i))
         | _ -> true)
       (insns (Peak.Gemm "i16")));
  [%expect
    {|
    t0
    elementwise writes t0: true
    gemm_i16 (A in t0-t2) never sets vl into t0: true |}]
