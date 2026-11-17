(* The contraction helpers an artifact calls, as typed Rivet instructions: one
   leaf function per descriptor, the AAPCS64 shape [(a, b, c)] in x0-x2, the
   stack untouched, x3-x16 and s0-s2 as scratch (all caller-saved). The loop
   nest is the reference's: batch, row and column outermost, [k] ascending
   innermost with one accumulator seeded from zero or from C, each product
   (fmul) and each sum (fadd) rounded to binary32 and never fused, so the bits
   are {!Contraction.eval}'s under [Ordered]. Strides are walked as pointers
   advanced by constant byte steps; a stride past [max_stride] elements is
   refused rather than wrapped. Integer and bfloat16 operands, and any other
   numerics, are typed refusals. *)

module F = Rivet_a64_form
module A = Aarch64_encode
module N = Asm_core.Normalized_ast
module D = Asm_core.Directive
module C = Contraction
module R = Rivet_a64_refusal

let origin = F.origin
let insn i = N.Instruction { insn = i; origin }
let dir directive = N.Directive { directive; origin }
let lbl name = N.Label { name; origin }

(* The largest stride in elements: below it every byte step is an [int64]. *)
let max_stride = 0x100_0000_0000L

(* The scratch registers by role. *)
module Reg = struct
  let a_arg = 0
  let b_arg = 1
  let c_arg = 2
  let batch_left = 3
  let row_left = 4
  let col_left = 5
  let k_left = 6
  let a_batch = 7
  let b_batch = 8
  let c_batch = 9
  let a_row = 10
  let c_row = 11
  let b_col = 12
  let c_at = 13
  let a_at = 14
  let b_at = 15
  let step = 16
end

(* A stride in elements as bytes. *)
let bytes esc stride =
  if Int64.compare stride max_stride > 0 then
    Err.Escape.throw esc (R.Contraction_stride stride)
  else Int64.mul stride 4L

let function_items ?mutation esc (d : C.t) =
  (match d.C.format with
  | C.Element.F32 -> ()
  | (C.Element.Bf16 | C.Element.I8 _) as e ->
      Err.Escape.throw esc (R.Contraction_format e));
  (match d.C.numerics with
  | C.Numerics.Ordered -> ()
  | (C.Numerics.Fused_sequential | C.Numerics.Paired) as n ->
      Err.Escape.throw esc (R.Contraction_numerics n));
  let by = bytes esc in
  let sa_b = by d.C.a.C.Operand.batch
  and sa_o = by d.C.a.C.Operand.outer
  and sa_k = by d.C.a.C.Operand.k
  and sb_b = by d.C.b.C.Operand.batch
  and sb_o = by d.C.b.C.Operand.outer
  and sb_k = by d.C.b.C.Operand.k
  and sc_b = by d.C.c.C.Output.batch
  and sc_r = by d.C.c.C.Output.row
  and sc_c = by d.C.c.C.Output.col in
  let name = C.symbol d in
  let reg n = { A.Reg.num = n; width = 64; is_sp = false } in
  let x n = A.Operand.Reg (reg n) in
  let w n = A.Operand.Reg { A.Reg.num = n; width = 32; is_sp = false } in
  let s n = A.Operand.Freg { A.Freg.num = n; double = false } in
  let i op ops = insn (F.ins op ops) in
  let at n =
    A.Operand.Mem
      {
        A.Mem.base = reg n;
        offset = A.Disp.Const 0L;
        writeback = false;
        pre = true;
      }
  in
  let target l = A.Operand.Sym (Asm_core.Expr.Symbol l) in
  let load rd v = List.map insn (F.materialize (reg rd) v) in
  (* [rd] advanced by [step] bytes *)
  let advance rd step =
    if Int64.equal step 0L then []
    else if Int64.compare step 4096L < 0 then
      [ i A.Opcode.Add [ x rd; x rd; F.imm step ] ]
    else load Reg.step step @ [ i A.Opcode.Add [ x rd; x rd; x Reg.step ] ]
  in
  let next rd label =
    [
      i A.Opcode.Sub [ x rd; x rd; F.imm 1L ];
      i A.Opcode.Cbnz [ x rd; target label ];
    ]
  in
  let l suffix = Fmt.str ".L%s_%s" name suffix in
  let lb = l "b" and li = l "i" and lj = l "j" and lk = l "k" in
  let open Reg in
  let seed =
    match d.C.seed with
    | C.Seed.Tensor when mutation <> Some F.Mutation.Contraction_seed_dropped ->
        [ i A.Opcode.Ldr [ s 0; at c_at ] ]
    | C.Seed.Tensor | C.Seed.Zero -> [ i A.Opcode.Fmov [ s 0; w 31 ] ]
  in
  let inner =
    if Int64.equal d.C.k 0L then []
    else
      load k_left d.C.k
      @ [
          i A.Opcode.Mov [ x a_at; x a_row ];
          i A.Opcode.Mov [ x b_at; x b_col ];
          lbl lk;
          i A.Opcode.Ldr [ s 1; at a_at ];
          i A.Opcode.Ldr [ s 2; at b_at ];
        ]
      @ (if mutation = Some F.Mutation.Contraction_fused then
           [ i A.Opcode.Fmadd [ s 0; s 1; s 2; s 0 ] ]
         else
           [
             i A.Opcode.Fmul [ s 1; s 1; s 2 ];
             i A.Opcode.Fadd [ s 0; s 0; s 1 ];
           ])
      @ [] @ advance a_at sa_k @ advance b_at sb_k @ next k_left lk
  in
  [
    dir
      (D.Section { name = ".text"; perms = Asm_core.Perms.rx; nobits = false });
    dir (D.Align { boundary = 4 });
    dir (D.Global { name });
    dir (D.Sym_type { name; kind = D.Function });
    lbl name;
  ]
  @ load batch_left d.C.batch
  @ [
      i A.Opcode.Mov [ x a_batch; x a_arg ];
      i A.Opcode.Mov [ x b_batch; x b_arg ];
      i A.Opcode.Mov [ x c_batch; x c_arg ];
      lbl lb;
    ]
  @ load row_left d.C.m
  @ [
      i A.Opcode.Mov [ x a_row; x a_batch ];
      i A.Opcode.Mov [ x c_row; x c_batch ];
      lbl li;
    ]
  @ load col_left d.C.n
  @ [
      i A.Opcode.Mov [ x b_col; x b_batch ];
      i A.Opcode.Mov [ x c_at; x c_row ];
      lbl lj;
    ]
  @ seed @ inner
  @ [ i A.Opcode.Str [ s 0; at c_at ] ]
  @ advance b_col sb_o @ advance c_at sc_c @ next col_left lj
  @ advance a_row sa_o @ advance c_row sc_r @ next row_left li
  @ advance a_batch sa_b @ advance b_batch sb_b @ advance c_batch sc_b
  @ next batch_left lb
  @ [
      i A.Opcode.Ret [];
      dir
        (D.Sym_size
           {
             name;
             size =
               Asm_core.Expr.Binary
                 ( Asm_core.Expr.Sub,
                   Asm_core.Expr.Current_location,
                   Asm_core.Expr.Symbol name );
           });
    ]

let module_ ?mutation descriptors =
  Err.Escape.with_escape @@ fun esc ->
  {
    N.unit_name = "contraction";
    items =
      List.concat_map (function_items ?mutation esc) descriptors
      @ [ dir (D.Declared_section { name = ".note.GNU-stack" }) ];
  }

module Art = Machine_model.Mir_artifact

let of_artifact artifact =
  List.filter_map
    (fun (s : Art.Symbol.t) ->
      match s.Art.Symbol.kind with
      | Art.Symbol.Generated_function d -> Some d
      | Art.Symbol.Data _ | Art.Symbol.External_function _
      | Art.Symbol.Function _ | Art.Symbol.Generated_attention _ ->
          None)
    (Art.symbols artifact)
