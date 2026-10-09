(* The project-owned [exp] as typed Rivet instructions: the operation order
   of {!Machine_ir.Mir_exp.model}, step for step, so the result is the
   model's and the model is the host libm's. A leaf with the AAPCS64 shape:
   the argument and the result in d0, x1-x11 and d1-d7 as scratch (all of
   them caller-saved and none the table register), the stack untouched. The
   reduction rounds half away from zero with the exact trick the instruction
   set lacks a direct form for: the sum with the largest double below one half
   of the argument's sign, truncated. *)

open Machine_ir
module F = Rivet_a64_form
module A = Aarch64
module N = Asm_core.Normalized_ast
module D = Asm_core.Directive

let symbol = "exp"
let data = "mltorch_exp_data"
let origin = F.origin
let insn i = N.Instruction { insn = i; origin }
let dir directive = N.Directive { directive; origin }
let lbl name = N.Label { name; origin }
let const n = Asm_core.Expr.Const (Foundation.Bigint.of_int64 n)

(* The data block: the model's constants, then four scalars the code needs,
   then the table. *)
let extras =
  [| 1.0; Int64.float_of_bits 0x3fdfffffffffffffL; 0x1p1009; 0x1p-1022 |]
  |> Array.map Int64.bits_of_float

let one = 7 * 8
let round_bias = 8 * 8
let big = 9 * 8
let tiny = 10 * 8
let table_at = 11 * 8
let words = Array.concat [ Mir_exp.constants; extras; Mir_exp.table ]

let module_ =
  Err.Escape.with_escape @@ fun _esc ->
  let i op ops = insn (F.ins op ops) in
  let x n = A.Operand.Reg { A.Reg.num = n; width = 64; is_sp = false } in
  let w n = A.Operand.Reg { A.Reg.num = n; width = 32; is_sp = false } in
  let d n = A.Operand.Freg { A.Freg.num = n; double = true } in
  let xzr = x 31 in
  let imm n = F.imm (Int64.of_int n) in
  let target name = A.Operand.Sym (Asm_core.Expr.Symbol name) in
  let at ?(base = 9) off =
    A.Operand.Mem
      {
        A.Mem.base = { A.Reg.num = base; width = 64; is_sp = false };
        offset = A.Disp.Const (Int64.of_int off);
        writeback = false;
        pre = true;
      }
  in
  let shift16 k = A.Operand.Shift { A.Shift.kind = "lsl"; amount = k } in
  let b_cond c l = insn (F.ins (A.Opcode.Bcond c) [ target l ]) in
  let cond n =
    match A.Cond.of_name n with Some c -> c | None -> invalid_arg n
  in
  let text =
    [
      dir
        (D.Section { name = ".text"; perms = Asm_core.Perms.rx; nobits = false });
      dir (D.Align { boundary = 4 });
      dir (D.Global { name = symbol });
      dir (D.Sym_type { name = symbol; kind = D.Function });
      lbl symbol;
      (* x1 = the bits of x, w2 = its exponent field *)
      i A.Opcode.Fmov [ x 1; d 0 ];
      i A.Opcode.Ubfx [ x 2; x 1; imm 52; imm 11 ];
      i A.Opcode.Sub [ w 3; w 2; imm 0x3c9 ];
      i A.Opcode.Cmp [ w 3; imm 0x3e ];
      b_cond (cond "ls") ".Lexp_main";
      (* outside [2^-54, 512) *)
      i A.Opcode.Cmp [ w 2; imm 0x3c9 ];
      b_cond (cond "lo") ".Lexp_one_plus";
      i A.Opcode.Cmp [ w 2; imm 0x408 ];
      b_cond (cond "eq") ".Lexp_main";
      i A.Opcode.Movz [ x 4; imm 0xfff0; shift16 48 ];
      i A.Opcode.Cmp [ x 1; x 4 ];
      b_cond (cond "eq") ".Lexp_zero";
      i A.Opcode.Cmp [ w 2; imm 0x7ff ];
      b_cond (cond "eq") ".Lexp_one_plus";
      i A.Opcode.Cmp [ x 1; imm 0 ];
      b_cond (cond "lt") ".Lexp_zero";
      i A.Opcode.Movz [ x 4; imm 0x7ff0; shift16 48 ];
      i A.Opcode.Fmov [ d 0; x 4 ];
      i A.Opcode.Ret [];
      lbl ".Lexp_zero";
      i A.Opcode.Fmov [ d 0; xzr ];
      i A.Opcode.Ret [];
      lbl ".Lexp_one_plus";
      i A.Opcode.Adrp [ x 9; target data ];
      i A.Opcode.Add
        [
          x 9;
          x 9;
          A.Operand.Sym
            (Asm_core.Expr.Modifier ("lo12", Asm_core.Expr.Symbol data));
        ];
      i A.Opcode.Ldr [ d 1; at one ];
      i A.Opcode.Fadd [ d 0; d 0; d 1 ];
      i A.Opcode.Ret [];
      lbl ".Lexp_main";
      i A.Opcode.Adrp [ x 9; target data ];
      i A.Opcode.Add
        [
          x 9;
          x 9;
          A.Operand.Sym
            (Asm_core.Expr.Modifier ("lo12", Asm_core.Expr.Symbol data));
        ];
      (* z = x * N/ln2; ki = z rounded half away from zero, kd = ki *)
      i A.Opcode.Ldr [ d 1; at 0 ];
      i A.Opcode.Fmul [ d 1; d 0; d 1 ];
      i A.Opcode.Fmov [ x 4; d 1 ];
      i A.Opcode.Ubfx [ x 4; x 4; imm 63; imm 1 ];
      i A.Opcode.Ubfiz [ x 4; x 4; imm 63; imm 1 ];
      i A.Opcode.Ldr [ x 5; at round_bias ];
      i A.Opcode.Orr [ x 4; x 4; x 5 ];
      i A.Opcode.Fmov [ d 2; x 4 ];
      i A.Opcode.Fadd [ d 2; d 1; d 2 ];
      i A.Opcode.Fcvtzs [ x 6; d 2 ];
      i A.Opcode.Scvtf [ d 2; x 6 ];
      (* r = x - kd * ln2 / N, in two fused steps *)
      i A.Opcode.Ldr [ d 4; at 8 ];
      i A.Opcode.Fmadd [ d 3; d 4; d 2; d 0 ];
      i A.Opcode.Ldr [ d 4; at 16 ];
      i A.Opcode.Fmadd [ d 3; d 4; d 2; d 3 ];
      (* the table entry and the scale bits *)
      i A.Opcode.Ubfiz [ x 7; x 6; imm 4; imm 7 ];
      i A.Opcode.Add [ x 8; x 9; x 7 ];
      i A.Opcode.Ldr [ d 5; at ~base:8 table_at ];
      i A.Opcode.Ldr [ x 10; at ~base:8 (table_at + 8) ];
      i A.Opcode.Ubfiz [ x 11; x 6; imm 45; imm 19 ];
      i A.Opcode.Add [ x 10; x 10; x 11 ];
      (* tmp = (r + tail) + r2 * (c2 + r * c3) + r2 * r2 * (c4 + r * c5) *)
      i A.Opcode.Ldr [ d 4; at 24 ];
      i A.Opcode.Ldr [ d 1; at 32 ];
      i A.Opcode.Fmadd [ d 4; d 1; d 3; d 4 ];
      i A.Opcode.Fmul [ d 2; d 3; d 3 ];
      i A.Opcode.Ldr [ d 1; at 40 ];
      i A.Opcode.Ldr [ d 6; at 48 ];
      i A.Opcode.Fmadd [ d 1; d 6; d 3; d 1 ];
      i A.Opcode.Fadd [ d 5; d 3; d 5 ];
      i A.Opcode.Fmadd [ d 5; d 4; d 2; d 5 ];
      i A.Opcode.Fmul [ d 6; d 2; d 2 ];
      i A.Opcode.Fmadd [ d 5; d 6; d 1; d 5 ];
      i A.Opcode.Cmp [ w 2; imm 0x408 ];
      b_cond (cond "eq") ".Lexp_large";
      (* scale + scale * tmp *)
      i A.Opcode.Fmov [ d 0; x 10 ];
      i A.Opcode.Fmadd [ d 0; d 5; d 0; d 0 ];
      i A.Opcode.Ret [];
      lbl ".Lexp_large";
      i A.Opcode.Cmp [ x 6; imm 0 ];
      b_cond (cond "lt") ".Lexp_negative";
      i A.Opcode.Movz [ x 4; imm 0x3f10; shift16 48 ];
      i A.Opcode.Sub [ x 10; x 10; x 4 ];
      i A.Opcode.Fmov [ d 0; x 10 ];
      i A.Opcode.Fmadd [ d 0; d 5; d 0; d 0 ];
      i A.Opcode.Ldr [ d 1; at big ];
      i A.Opcode.Fmul [ d 0; d 0; d 1 ];
      i A.Opcode.Ret [];
      lbl ".Lexp_negative";
      i A.Opcode.Movz [ x 4; imm 0x3fe0; shift16 48 ];
      i A.Opcode.Add [ x 10; x 10; x 4 ];
      i A.Opcode.Fmov [ d 0; x 10 ];
      i A.Opcode.Fmul [ d 5; d 5; d 0 ];
      i A.Opcode.Fadd [ d 1; d 0; d 5 ];
      i A.Opcode.Ldr [ d 2; at one ];
      i A.Opcode.Fcmp [ d 1; d 2 ];
      b_cond (cond "pl") ".Lexp_scaled";
      (* y < 1: round to the subnormal precision before scaling *)
      i A.Opcode.Fsub [ d 3; d 0; d 1 ];
      i A.Opcode.Fadd [ d 3; d 3; d 5 ];
      i A.Opcode.Fadd [ d 4; d 2; d 1 ];
      i A.Opcode.Fsub [ d 6; d 2; d 4 ];
      i A.Opcode.Fadd [ d 6; d 6; d 1 ];
      i A.Opcode.Fadd [ d 6; d 6; d 3 ];
      i A.Opcode.Fadd [ d 4; d 4; d 6 ];
      i A.Opcode.Fsub [ d 1; d 4; d 2 ];
      lbl ".Lexp_scaled";
      i A.Opcode.Ldr [ d 3; at tiny ];
      i A.Opcode.Fmul [ d 0; d 1; d 3 ];
      i A.Opcode.Ret [];
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
