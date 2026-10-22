open Machine_rivet_riscv32
module I = Rv32_kernel_imm
module Insn = Rv32_kernel_insn
module R = Rv32_kernel_reg

let%expect_test "domain boundaries" =
  let edges (type a) (module M : I.S with type t = a) values =
    List.iter (fun n -> Fmt.pr "%d:%b " n (M.in_range (M.of_int n))) values;
    Fmt.pr "@."
  in
  edges (module I.Imm12) [ -2049; -2048; 2047; 2048 ];
  edges (module I.Shamt) [ -1; 0; 31; 32 ];
  edges (module I.Simm5) [ -17; -16; 15; 16 ];
  edges (module I.U20) [ -1; 0; 0xFFFFF; 0x100000 ];
  [%expect
    {|
    -2049:false -2048:true 2047:true 2048:false
    -1:false 0:true 31:true 32:false
    -17:false -16:true 15:true 16:false
    -1:false 0:true 1048575:true 1048576:false |}]

(* Whether Rivet encodes [li a0, v]: refusals arrive when the module is planned
   (an out-of-range addi fixup), not when the instruction is made. *)
let rivet_accepts_li v =
  let open Rv32_asm in
  match
    module_ "p" (fun () ->
        [
          text;
          global "e";
          label "e";
          insn "li" [ reg "a0"; imm v ];
          insn "ret" [];
        ])
  with
  | Error _ -> false
  | Ok m -> Result.is_ok (Err.payload (Rv32_image.plan ~entry:"e" [ m ]))

(* The Li domain is Rivet's, probed: it does not expand a wide constant itself. *)
let%expect_test "Li agrees with what Rivet encodes" =
  List.iter
    (fun v ->
      let ours = I.Li.in_range (I.Li.of_int v) and rivet = rivet_accepts_li v in
      Fmt.pr "%d: %b%s@." v ours (if ours = rivet then "" else " DISAGREES"))
    [
      -0x8000_0000;
      -4096;
      -2049;
      -2048;
      -1;
      0;
      2047;
      2048;
      0x1000;
      0x1001;
      0x7FFF_F000;
      0x8000_0000;
      0xFFFF_F000;
      0xFFFF_FFFF;
    ];
  [%expect
    {|
    -2147483648: true
    -4096: true
    -2049: false
    -2048: true
    -1: true
    0: true
    2047: true
    2048: false
    4096: true
    4097: false
    2147479552: true
    2147483648: false
    4294963200: false
    4294967295: false |}]

let pp_insns insns = String.concat "; " (List.map (Fmt.str "%a" Insn.pp) insns)

(* Rivet prints a0 as x10; the expansion is otherwise textually the same. *)
let rivet_expansion v =
  let words =
    Rv32_asm.load_imm "a0" v
    |> List.map (function
      | Asm_core.Normalized_ast.Instruction { insn; _ } ->
          Fmt.str "%a" Riscv32_encode.Instruction.pp insn
      | _ -> assert false)
    |> String.concat "; " |> String.split_on_char ' '
  in
  String.concat " "
    (List.map
       (fun w ->
         match w with
         | "x10" -> "a0"
         | "x10," -> "a0,"
         | "x10;" -> "a0;"
         | w -> w)
       words)

let%expect_test "load_imm expands as Rv32_asm.load_imm does" =
  List.iter
    (fun v ->
      match Err.payload (Insn.load_imm R.Gpr.a0 v) with
      | Error _ -> Fmt.pr "%d: refused@." v
      | Ok insns ->
          let ours = pp_insns insns in
          Fmt.pr "%d: %s%s@." v ours
            (if String.equal ours (rivet_expansion v) then ""
             else " DIFFERS from " ^ rivet_expansion v))
    [
      -0x8000_0000;
      -2049;
      -2048;
      0;
      2047;
      2048;
      4096;
      0x7FFF_FFFF;
      0x8000_0000;
      0xFFFF_FFFF;
      -2304;
    ];
  [%expect
    {|
    -2147483648: lui a0, 524288
    -2049: lui a0, 1048575; addi a0, a0, 2047
    -2048: li a0, -2048
    0: li a0, 0
    2047: li a0, 2047
    2048: lui a0, 1; addi a0, a0, -2048
    4096: lui a0, 1
    2147483647: lui a0, 524288; addi a0, a0, -1
    2147483648: lui a0, 524288
    4294967295: lui a0, 0; addi a0, a0, -1
    -2304: lui a0, 1048575; addi a0, a0, 1792 |}]

let%expect_test "load_imm refuses what is not a 32-bit word before splitting it"
    =
  List.iter
    (fun v ->
      match Err.payload (Insn.load_imm R.Gpr.a0 v) with
      | Ok _ -> Fmt.pr "%d: accepted@." v
      | Error (`Constant_range c) ->
          Fmt.pr "%d: %a@." v Insn.Constant_range.pp c)
    [ -0x8000_0001; 0x1_0000_0000; max_int ];
  [%expect
    {|
    -2147483649: -2147483649 is not a 32-bit constant
    4294967296: 4294967296 is not a 32-bit constant
    4611686018427387903: 4611686018427387903 is not a 32-bit constant |}]
