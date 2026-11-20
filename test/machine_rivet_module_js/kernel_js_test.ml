(* The RV32 kernel DSL and the Coral kernel emitters are pure: the same expect
   blocks run natively and under node ([js]). A 32-bit constant check written
   with a literal that js_of_ocaml's 32-bit int truncates would refuse here. *)

module Insn = Machine_rivet_riscv32_module.Rv32_kernel_insn
module Reg = Machine_rivet_riscv32_module.Rv32_kernel_reg
module C = Contraction
module Gnu = Asm_core.Gnu_module

let descriptor numerics ~m ~n ~k =
  Result.get_ok
    (Err.payload
       (C.make ~batch:1L ~m ~n ~k
          ~a:{ C.Operand.batch = 0L; outer = k; k = 1L }
          ~b:{ C.Operand.batch = 0L; outer = 1L; k = n }
          ~c:{ C.Output.batch = 0L; row = n; col = 1L }
          ~format:C.Element.F32 ~seed:C.Seed.Zero ~numerics))

let show m =
  let text =
    Fmt.str "%a"
      (Gnu.pp { Gnu.type_char = '@' } ~instruction:Riscv32_encode.Instruction.pp)
      m
  in
  let lines = String.split_on_char '\n' text in
  Fmt.pr "%d lines, md5 %s@." (List.length lines)
    (Digest.to_hex (Digest.string text))

let%expect_test "load_imm across the signed and unsigned 32-bit range" =
  List.iter
    (fun v ->
      match Err.payload (Insn.load_imm Reg.Gpr.t0 v) with
      | Ok is -> Fmt.pr "%d: %d insns@." v (List.length is)
      | Error _ -> Fmt.pr "%d: refused@." v)
    [ 0; 2047; 2048; 4096; 0x12345; 0x7FFF_F000; 0x7FFF_FFFF; -0x8000_0000 ];
  [%expect
    {|
    0: 1 insns
    2047: 1 insns
    2048: 2 insns
    4096: 1 insns
    74565: 2 insns
    2147479552: 1 insns
    2147483647: 2 insns
    -2147483648: 1 insns |}]

let%expect_test "Rvv_contraction module" =
  (match
     Coralnpu_kernels.Rvv_contraction.module_
       (descriptor C.Numerics.Ordered ~m:5L ~n:40L ~k:7L)
   with
  | Ok m -> show m
  | Error r -> Fmt.pr "%s@." (Coralnpu_kernels.Kernel_refusal.to_string r));
  [%expect {| 167 lines, md5 f84289dfeaa7b64f157e7126d8edd00f |}]

let%expect_test "Zvt_contraction module" =
  (match
     Coralnpu_kernels_zvt.Zvt_contraction.module_
       (descriptor C.Numerics.Ordered ~m:5L ~n:40L ~k:7L)
   with
  | Ok m -> show m
  | Error r -> Fmt.pr "%s@." (Coralnpu_kernels.Kernel_refusal.to_string r));
  [%expect {| 109 lines, md5 c88c9cbc5698e873af66635bb3f93787 |}]

let%expect_test "Coral_transfer, Coral_estimate and Coralnpu link and run" =
  let d = descriptor C.Numerics.Ordered ~m:5L ~n:40L ~k:7L in
  let budget = Coral_transfer.Transfer_units.Bytes.of_int 65536 in
  let plan = Coral_transfer.Transfer_model.choose ~budget d in
  Fmt.pr "%a@." Coral_transfer.Transfer_plan.pp plan;
  Fmt.pr "%.1f@." (Coral_transfer.Transfer_model.predict d plan);
  ignore Coral_estimate.Coral_rates.vme;
  (match Jsont_bytesrw.decode_string Jsont.json {|{"name":"x","a":1}|} with
  | Ok j -> (
      match Err.payload (Coral_manifest.configuration_id j) with
      | Ok id -> Fmt.pr "%s@." id
      | Error _ -> Fmt.pr "error@.")
  | Error e -> Fmt.pr "%s@." e);
  [%expect
    {|
    A in place, B in place
    1255.4
    coralnpu:x:2163e01d4c8038446b0b |}]
