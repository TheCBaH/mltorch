open Machine_ir
module Src = Machine_source_test.Mir_source
module Art = Machine_model.Mir_artifact
module Gnu = Asm_core.Gnu_module

let ( let* ) = Result.bind

(* A pointwise kernel over four values: binary64, or binary32 where the target has no f64 (RV32). *)
let case ?(binary32 = false) () =
  let shape = Loop_ir_test.Loop_fixtures.shape_w 4 in
  let plan = Fusion_plan.default Loop_ir_test.Loop_programs.kernel in
  let bind =
    Ssa_bridge_test.Ssa_fixtures.bind_data ~shape [| -0.; 1.5; nan; 3. |]
  in
  match
    if binary32 then
      Src.case_of_planned ~target:Ssa_ir.Ssa_target.scalar
        ~numerics:Ssa_ir.Ssa_numerics.Fp32_ordered plan ~bind
    else Src.case_of_plan plan ~bind
  with
  | Ok c -> c
  | Error e -> failwith e

let planning g =
  Option.to_result ~none:"no planning summary"
    (Mir_verify.Generic.program g).Mir_program.planning

(* The text's size and digest identify it; the head shows its shape. *)
let show ~type_char ~instruction m =
  let text = Fmt.str "%a" (Gnu.pp { Gnu.type_char } ~instruction) m in
  let lines = String.split_on_char '\n' text in
  Fmt.pr "%d lines, md5 %s@." (List.length lines)
    (Digest.to_hex (Digest.string text));
  List.iteri
    (fun i l ->
      if i < 6 then
        Fmt.pr "  %s@."
          (String.concat "<TAB>" (String.split_on_char (Char.chr 9) l)))
    lines

let published ~select ~realize ~publish g =
  let* planning = planning g in
  let* v = select g in
  let* real = realize v in
  publish ~planning v real

let%expect_test "x86-64 typed module and manifest" =
  let open Machine_target_x86_64 in
  let module A = Machine_alloc.Mir_ref_alloc.Make (X64) (X64_regs) in
  let module Fr = Machine_alloc.Mir_frame.Make (X64) (X64_frame) in
  let module Pub = Machine_model.Mir_artifact.Make (X64) in
  let g = (case ()).Src.Case.lowered.Machine_lower.Mir_lower.program in
  (match
     published
       ~select:(fun g ->
         Result.map
           (fun r -> r.X64_select.selected)
           (Result.map_error
              (Fmt.str "%a" X64_select.Refusal.pp)
              (Err.payload (X64_select.program ~sites:[||] g))))
       ~realize:(fun v ->
         Result.map_error
           (Fmt.str "%a" Machine_alloc.Mir_frame.Refusal.pp)
           (Fr.realize (A.allocate v)))
       ~publish:(fun ~planning v real -> Pub.publish ~planning v real)
       g
   with
  | Error e -> Fmt.pr "%s@." e
  | Ok artifact -> (
      match
        Err.payload
          (Machine_rivet_x86_64_module.Rivet_x64_module.of_artifact artifact)
      with
      | Error r ->
          Fmt.pr "%a@." Machine_rivet_x86_64_module.Rivet_x64_refusal.pp r
      | Ok m ->
          show ~type_char:'@' ~instruction:X86_64_encode.Instruction.pp_gnu m;
          let man =
            Machine_rivet_x86_64_module.Rivet_x64_manifest.make
              ~runtime:
                Machine_rivet_x86_64_module.Rivet_x64_runtime.Dependency_free
              ~binding:
                Machine_rivet_x86_64_module.Rivet_x64_module.Image_resident
              artifact
          in
          Fmt.pr "code_digest %s@."
            man.Machine_rivet_x86_64_module.Rivet_x64_manifest.code_digest));
  [%expect
    {|
    110 lines, md5 ed66ea5c12db7a63158923d557e43550
      <TAB>.section<TAB>.text,"ax",@progbits
      <TAB>.balign<TAB>16
      <TAB>.globl<TAB>kernel
      <TAB>.type<TAB>kernel,@function
      kernel:
      <TAB>.cfi_startproc
    code_digest ed66ea5c12db7a63158923d557e43550 |}]

let%expect_test "AArch64 typed module and manifest" =
  let open Machine_target_aarch64 in
  let module A = Machine_alloc.Mir_ref_alloc.Make (A64) (A64_regs) in
  let module Fr = Machine_alloc.Mir_frame.Make (A64) (A64_frame) in
  let module Pub = Machine_model.Mir_artifact.Make (A64) in
  let g = (case ()).Src.Case.lowered.Machine_lower.Mir_lower.program in
  (match
     published
       ~select:(fun g ->
         Result.map
           (fun r -> r.A64_select.selected)
           (Result.map_error
              (Fmt.str "%a" A64_select.Refusal.pp)
              (Err.payload (A64_select.program ~sites:[||] g))))
       ~realize:(fun v ->
         Result.map_error
           (Fmt.str "%a" Machine_alloc.Mir_frame.Refusal.pp)
           (Fr.realize (A.allocate v)))
       ~publish:(fun ~planning v real -> Pub.publish ~planning v real)
       g
   with
  | Error e -> Fmt.pr "%s@." e
  | Ok artifact -> (
      match
        Err.payload
          (Machine_rivet_aarch64_module.Rivet_a64_module.of_artifact artifact)
      with
      | Error r ->
          Fmt.pr "%a@." Machine_rivet_aarch64_module.Rivet_a64_refusal.pp r
      | Ok m ->
          show ~type_char:'%' ~instruction:Aarch64_encode.Instruction.pp_gnu m;
          let man =
            Machine_rivet_aarch64_module.Rivet_a64_manifest.make
              ~runtime:
                Machine_rivet_aarch64_module.Rivet_a64_runtime.Dependency_free
              ~binding:
                Machine_rivet_aarch64_module.Rivet_a64_module.Image_resident
              artifact
          in
          Fmt.pr "code_digest %s@."
            man.Machine_rivet_aarch64_module.Rivet_a64_manifest.code_digest));
  [%expect
    {|
    118 lines, md5 548fbdd5e9297e022a8528396e0f2b1c
      <TAB>.section<TAB>.text,"ax",%progbits
      <TAB>.balign<TAB>4
      <TAB>.globl<TAB>kernel
      <TAB>.type<TAB>kernel,%function
      kernel:
      <TAB>.cfi_startproc
    code_digest 548fbdd5e9297e022a8528396e0f2b1c |}]

let%expect_test "RV32 typed module" =
  let open Machine_target_riscv32 in
  let module A = Machine_alloc.Mir_ref_alloc.Make (Rv32) (Rv32_regs) in
  let module Fr = Machine_alloc.Mir_frame.Make (Rv32) (Rv32_frame) in
  let module Pub = Machine_model.Mir_artifact.Make (Rv32) in
  let g =
    (case ~binary32:true ()).Src.Case.lowered.Machine_lower.Mir_lower.program
  in
  (match
     published
       ~select:(fun g ->
         Result.map
           (fun r -> r.Rv32_select.selected)
           (Result.map_error
              (Fmt.str "%a" Rv32_select.Refusal.pp)
              (Err.payload
                 (Rv32_select.program ~sites:[||]
                    ~unlisted:Mir_failure.Unlisted.Unreachable g))))
       ~realize:(fun v ->
         Result.map_error
           (Fmt.str "%a" Machine_alloc.Mir_frame.Refusal.pp)
           (Fr.realize (A.allocate v)))
       ~publish:(fun ~planning v real -> Pub.publish ~planning v real)
       g
   with
  | Error e -> Fmt.pr "%s@." e
  | Ok artifact -> (
      match
        Err.payload
          (Machine_rivet_riscv32_mir_module.Rv32_mir_module.of_artifact artifact)
      with
      | Error r ->
          Fmt.pr "%a@."
            Machine_rivet_riscv32_mir_module.Rv32_mir_form.Refusal.pp r
      | Ok m -> show ~type_char:'@' ~instruction:Riscv32_encode.Instruction.pp m
      ));
  [%expect
    {|
    81 lines, md5 292627d1148d071d5157aa27f39174e3
      <TAB>.section<TAB>.text,"ax",@progbits
      <TAB>.balign<TAB>4
      <TAB>.globl<TAB>kernel
      kernel:
      .Lkernel_b0:
      <TAB>addi x2, x2, -64 |}]
