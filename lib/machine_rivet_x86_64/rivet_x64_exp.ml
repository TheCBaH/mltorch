(* The project-owned [exp] as typed Rivet instructions: the operation order
   of {!Machine_ir.Mir_exp.model}, step for step, so the result is the
   model's and the model is the host libm's. A leaf with the System V shape:
   the argument and the result in xmm0, the caller-saved registers rax, rcx,
   rdx, rsi, rdi, r8 and xmm1-xmm7 as scratch, the stack untouched. It needs
   FMA3, as the binary64 fused operations of the model do. *)

open Machine_ir
module F = Rivet_x64_form
module R = Rivet_x64_refusal
module Fam = X86_family_encode
module N = Asm_core.Normalized_ast
module D = Asm_core.Directive

let symbol = "exp"
let data = "mltorch_exp_data"
let origin = F.origin
let insn i = N.Instruction { insn = i; origin }
let dir directive = N.Directive { directive; origin }
let lbl name = N.Label { name; origin }
let const n = Asm_core.Expr.Const (Foundation.Bigint.of_int64 n)

(* The data block: the model's constants, then eight scalars the code needs,
   then the table. *)
let extras =
  [|
    1.0;
    0.5;
    Int64.float_of_bits 0x7fffffffffffffffL;
    Int64.float_of_bits Int64.min_int;
    0x1p1009;
    0x1p-1022;
  |]
  |> Array.map Int64.bits_of_float

let one = 7 * 8
let half = 8 * 8
let abs_mask = 9 * 8
let sign_mask = 10 * 8
let big = 11 * 8
let tiny = 12 * 8
let table_at = 13 * 8
let words = Array.concat [ Mir_exp.constants; extras; Mir_exp.table ]

let module_ =
  Err.Escape.with_escape @@ fun esc ->
  let env =
    {
      F.mutation = None;
      esc;
      reference = (fun _ -> None);
      table_slot = (fun _ -> None);
    }
  in
  let reg n = Fam.Operand.Reg (F.find env n) in
  let i mnemonic ops = insn (F.make env mnemonic ops) in
  let rip off =
    Fam.Operand.Mem
      {
        Fam.Mem.base = Some Fam.rip_reg;
        index = None;
        scale = 1;
        disp =
          Fam.Disp.Sym
            (Asm_core.Expr.Binary
               ( Asm_core.Expr.Add,
                 Asm_core.Expr.Symbol data,
                 const (Int64.of_int off) ));
      }
  in
  let at ?index base disp =
    F.mem_op ?index ~base:(F.find env base) ~disp:(Int64.of_int disp) ()
  in
  let target name = Fam.Operand.Sym (Asm_core.Expr.Symbol name) in
  let imm n = F.imm (Int64.of_int n) in
  let imm64 n = F.imm n in
  let x k = reg (Printf.sprintf "xmm%d" k) in
  let text =
    [
      dir
        (D.Section { name = ".text"; perms = Asm_core.Perms.rx; nobits = false });
      dir (D.Align { boundary = 16 });
      dir (D.Global { name = symbol });
      dir (D.Sym_type { name = symbol; kind = D.Function });
      lbl symbol;
      (* rax = the bits of x, ecx = its exponent field *)
      i "movq" [ x 0; reg "rax" ];
      i "movq" [ reg "rax"; reg "rcx" ];
      i "shrq" [ imm 52; reg "rcx" ];
      i "andl" [ imm 0x7ff; reg "ecx" ];
      i "movl" [ reg "ecx"; reg "edx" ];
      i "subl" [ imm 0x3c9; reg "edx" ];
      i "cmpl" [ imm 0x3e; reg "edx" ];
      i "jbe" [ target ".Lexp_main" ];
      (* outside [2^-54, 512) *)
      i "cmpl" [ imm 0x3c9; reg "ecx" ];
      i "jb" [ target ".Lexp_one_plus" ];
      i "cmpl" [ imm 0x408; reg "ecx" ];
      i "je" [ target ".Lexp_main" ];
      i "movabsq" [ imm64 0xfff0000000000000L; reg "rdx" ];
      i "cmpq" [ reg "rdx"; reg "rax" ];
      i "je" [ target ".Lexp_zero" ];
      i "cmpl" [ imm 0x7ff; reg "ecx" ];
      i "je" [ target ".Lexp_one_plus" ];
      i "testq" [ reg "rax"; reg "rax" ];
      i "js" [ target ".Lexp_zero" ];
      i "movabsq" [ imm64 0x7ff0000000000000L; reg "rax" ];
      i "movq" [ reg "rax"; x 0 ];
      i "ret" [];
      lbl ".Lexp_zero";
      i "xorps" [ x 0; x 0 ];
      i "ret" [];
      lbl ".Lexp_one_plus";
      i "addsd" [ rip one; x 0 ];
      i "ret" [];
      lbl ".Lexp_main";
      i "leaq" [ rip 0; reg "r8" ];
      (* z = x * N/ln2; kd = z rounded half away from zero, ki = kd *)
      i "movsd" [ at "r8" 0; x 1 ];
      i "mulsd" [ x 0; x 1 ];
      i "roundsd" [ imm 11; x 1; x 2 ];
      i "movaps" [ x 1; x 3 ];
      i "subsd" [ x 2; x 3 ];
      i "movsd" [ at "r8" abs_mask; x 5 ];
      i "andpd" [ x 5; x 3 ];
      i "movsd" [ at "r8" half; x 4 ];
      i "cmpsd" [ imm 2; x 3; x 4 ];
      i "movsd" [ at "r8" sign_mask; x 5 ];
      i "andpd" [ x 5; x 1 ];
      i "movsd" [ at "r8" one; x 5 ];
      i "orpd" [ x 5; x 1 ];
      i "andpd" [ x 1; x 4 ];
      i "addsd" [ x 4; x 2 ];
      i "cvttsd2si" [ x 2; reg "rsi" ];
      (* r = x - kd * ln2 / N, in two fused steps *)
      i "movaps" [ x 0; x 3 ];
      i "movsd" [ at "r8" 8; x 4 ];
      i "vfmadd231sd" [ x 2; x 4; x 3 ];
      i "movsd" [ at "r8" 16; x 4 ];
      i "vfmadd231sd" [ x 2; x 4; x 3 ];
      (* the table entry and the scale bits *)
      i "movq" [ reg "rsi"; reg "rdi" ];
      i "andl" [ imm 127; reg "edi" ];
      i "shlq" [ imm 4; reg "rdi" ];
      i "movsd" [ at ~index:(F.find env "rdi", 1) "r8" table_at; x 5 ];
      i "movq"
        [ at ~index:(F.find env "rdi", 1) "r8" (table_at + 8); reg "rax" ];
      i "movq" [ reg "rsi"; reg "rdx" ];
      i "shlq" [ imm 45; reg "rdx" ];
      i "addq" [ reg "rdx"; reg "rax" ];
      (* tmp = (r + tail) + r2 * (c2 + r * c3) + r2 * r2 * (c4 + r * c5) *)
      i "movsd" [ at "r8" 24; x 4 ];
      i "movsd" [ at "r8" 32; x 6 ];
      i "vfmadd231sd" [ x 3; x 6; x 4 ];
      i "movaps" [ x 3; x 2 ];
      i "mulsd" [ x 3; x 2 ];
      i "movsd" [ at "r8" 40; x 6 ];
      i "movsd" [ at "r8" 48; x 7 ];
      i "vfmadd231sd" [ x 3; x 7; x 6 ];
      i "addsd" [ x 3; x 5 ];
      i "vfmadd231sd" [ x 2; x 4; x 5 ];
      i "movaps" [ x 2; x 7 ];
      i "mulsd" [ x 2; x 7 ];
      i "vfmadd231sd" [ x 6; x 7; x 5 ];
      i "cmpl" [ imm 0x408; reg "ecx" ];
      i "je" [ target ".Lexp_large" ];
      (* scale + scale * tmp *)
      i "movq" [ reg "rax"; x 0 ];
      i "movaps" [ x 0; x 1 ];
      i "vfmadd231sd" [ x 5; x 1; x 0 ];
      i "ret" [];
      lbl ".Lexp_large";
      i "testl" [ imm 0x80000000; reg "esi" ];
      i "jne" [ target ".Lexp_negative" ];
      i "movabsq" [ imm64 (Int64.shift_left 1009L 52); reg "rdx" ];
      i "subq" [ reg "rdx"; reg "rax" ];
      i "movq" [ reg "rax"; x 0 ];
      i "movaps" [ x 0; x 1 ];
      i "vfmadd231sd" [ x 5; x 1; x 0 ];
      i "mulsd" [ at "r8" big; x 0 ];
      i "ret" [];
      lbl ".Lexp_negative";
      i "movabsq" [ imm64 (Int64.shift_left 1022L 52); reg "rdx" ];
      i "addq" [ reg "rdx"; reg "rax" ];
      i "movq" [ reg "rax"; x 0 ];
      i "mulsd" [ x 0; x 5 ];
      i "movaps" [ x 0; x 1 ];
      i "addsd" [ x 5; x 1 ];
      i "movsd" [ at "r8" one; x 2 ];
      i "ucomisd" [ x 2; x 1 ];
      i "jae" [ target ".Lexp_scaled" ];
      (* y < 1: round to the subnormal precision before scaling *)
      i "movaps" [ x 0; x 3 ];
      i "subsd" [ x 1; x 3 ];
      i "addsd" [ x 5; x 3 ];
      i "movaps" [ x 2; x 4 ];
      i "addsd" [ x 1; x 4 ];
      i "movaps" [ x 2; x 6 ];
      i "subsd" [ x 4; x 6 ];
      i "addsd" [ x 1; x 6 ];
      i "addsd" [ x 3; x 6 ];
      i "addsd" [ x 6; x 4 ];
      i "subsd" [ x 2; x 4 ];
      i "movaps" [ x 4; x 1 ];
      lbl ".Lexp_scaled";
      i "mulsd" [ at "r8" tiny; x 1 ];
      i "movaps" [ x 1; x 0 ];
      i "ret" [];
    ]
  in
  let data_items =
    [
      dir
        (D.Section
           { name = ".rodata"; perms = Asm_core.Perms.ro; nobits = false });
      dir (D.Align { boundary = 16 });
      dir (D.Global { name = data });
      lbl data;
      dir (D.Data { width = 8; values = Array.to_list (Array.map const words) });
    ]
  in
  {
    N.unit_name = "exp";
    items =
      text @ data_items
      @ [ dir (D.Declared_section { name = ".note.GNU-stack" }) ];
  }
