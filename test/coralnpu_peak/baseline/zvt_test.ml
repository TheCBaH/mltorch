(* The Zvt instructions through the kernel lowering and coralview's Rivet
   extension, against Coral's own words: the matrix.elf release, the VME cocotb
   tests and their encoding helpers. *)

open Machine_rivet_riscv32
module K = Peak_kernels
module Insn = Rv32_kernel_insn
module Item = Rv32_kernel_item
module Lower = Rv32_kernel_lower
module Gpr = Rv32_kernel_reg.Gpr
module Mtreg = Rv32_kernel_reg.Mtreg
module Vreg = Rv32_kernel_reg.Vreg

let lower ?(profile = K.rv32_profile) ~extensions i =
  match
    Err.payload (Rv32_kernel.finalize ~name:"zvt" [ Item.text; Item.insn i ])
  with
  | Error e -> Fmt.failwith "%a" Rv32_kernel.pp_error e
  | Ok k -> Err.payload (Lower.module_ ~extensions ~profile k)

(* The one word a single-instruction kernel lowers to. *)
let word m =
  match m.Asm_core.Normalized_ast.items with
  | [
   _;
   Directive
     { directive = Data { width = 4; values = [ Asm_core.Expr.Const n ] }; _ };
  ] ->
      Option.get (Foundation.Bigint.to_int64_opt n)
  | _ -> failwith "not one word"

let zvt = match K.zvt with Ok x -> x | Error e -> failwith e

let%expect_test "each Zvt form encodes to Coral's word" =
  let mt4 = Result.get_ok (Err.payload (Mtreg.of_index 4)) in
  List.iter
    (fun (i, expected) ->
      match lower ~extensions:[ zvt ] i with
      | Error e -> Fmt.pr "%a: %a@." Insn.pp i Lower.pp_error e
      | Ok m ->
          let w = word m in
          Fmt.pr "%08Lx %s %a@." w
            (if w = expected then "=" else Fmt.str "<> %08Lx" expected)
            Insn.pp i)
    [
      ( Insn.Msetmtype { dst = Gpr.zero; mtype = Gpr.a3; vtype = Gpr.a1 },
        0x82b6f057L );
      (Insn.Msettn { dst = Gpr.a1; avl = Gpr.a2 }, 0x840675d7L);
      (Insn.Vtzero { dst = Mtreg.mt0 }, 0x43e06057L);
      (Insn.Vtse { eew = E32; slice = Gpr.a2; base = Gpr.sp }, 0x52c17027L);
      ( Insn.Vtmm { op = Vtmms; acc = Mtreg.mt0; a = Vreg.v8; b = Vreg.v16 },
        0xf28800f7L );
      ( Insn.Vtmm { op = Vtmmu; acc = mt4; a = Vreg.v8; b = Vreg.v16 },
        0xf2880477L );
      ( Insn.Vtmm { op = Vtfmm; acc = Mtreg.mt0; a = Vreg.v8; b = Vreg.v16 },
        0xf2881077L );
      ( Insn.Vtmm { op = Vtfmm_alt; acc = Mtreg.mt0; a = Vreg.v8; b = Vreg.v16 },
        0xf28810f7L );
    ];
  [%expect
    {|
    82b6f057 = msetmtype zero, a3, a1
    840675d7 = msettn a1, a2
    43e06057 = vtzero mt0
    52c17027 = vtse32 a2, (sp)
    f28800f7 = vtmms.tvv mt0, v8, v16
    f2880477 = vtmmu.tvv mt4, v8, v16
    f2881077 = vtfmm.tvv mt0, v8, v16
    f28810f7 = vtfmm.alt.tvv mt0, v8, v16 |}]

let%expect_test "without the extension, or its vector base, Zvt is refused" =
  let i = Insn.Vtzero { dst = Mtreg.mt0 } in
  let show r =
    match r with
    | Ok _ -> Fmt.pr "lowered@."
    | Error e -> Fmt.pr "%a@." Lower.pp_error e
  in
  show (lower ~extensions:[] i);
  show (lower ~profile:Rv32_profile.Rv32imf_ilp32f ~extensions:[ zvt ] i);
  [%expect
    {|
    item 1: needs the xcoralzvt extension, which was not given
    item 1: xcoralzvt needs zve32x, which rv32imf-ilp32f lacks |}]
