(* Typed RV32 instructions and directives for hand-built modules, made through
   the encoder's surface constructor (a mnemonic and typed operands, never
   assembly text). *)

module N = Asm_core.Normalized_ast
module D = Asm_core.Directive
module E = Riscv32_encode

let origin = Foundation.Origin.synthesized ~pass:"machine_rivet_riscv32" ()

exception Refused of string

let reg name =
  match E.Reg.find name with
  | Some r -> E.Operand.Reg r
  | None -> raise (Refused ("register " ^ name))

let imm n = E.Operand.Imm (Foundation.Bigint.of_int n)
let const n = Asm_core.Expr.Const (Foundation.Bigint.of_int n)
let sym name = E.Operand.Sym (Asm_core.Expr.Symbol name)

let sym_at name off =
  E.Operand.Sym
    (Asm_core.Expr.Binary
       (Asm_core.Expr.Add, Asm_core.Expr.Symbol name, const off))

(* [off(base)]. *)
let mem base off =
  match reg base with
  | E.Operand.Reg b -> E.Operand.Mem { E.Mem.offset = const off; base = b }
  | _ -> assert false

let insn mnemonic ops =
  match E.make_surface_instruction ~mnemonic ~origin ops with
  | Error _ -> raise (Refused mnemonic)
  | Ok s -> (
      match E.simplify_instruction E.default_state s with
      | Ok i -> N.Instruction { insn = i; origin }
      | Error _ -> raise (Refused mnemonic))

let dir directive = N.Directive { directive; origin }
let label name = N.Label { name; origin }
let global name = dir (D.Global { name })
let align boundary = dir (D.Align { boundary })
let section name perms nobits = dir (D.Section { name; perms; nobits })
let text = section ".text" Asm_core.Perms.rx false
let data = section ".data" Asm_core.Perms.rw false
let bss = section ".bss" Asm_core.Perms.rw true
let zero length = dir (D.Zero { length })
let words values = dir (D.Data { width = 4; values = List.map const values })

(* A module of [items]; a refused form is the [Error]. *)
let module_ unit_name items =
  match items () with
  | items -> Ok { N.unit_name; items }
  | exception Refused what -> Error what

(* [rd = v] for any 32-bit [v]. Rivet's [li] covers a signed 12-bit constant or
   one whose low 12 bits are zero; the rest is [lui] of the high part, adjusted
   for the sign of the low part, then [addi]. *)
let load_imm rd v =
  if v >= -2048 && v < 2048 then [ insn "li" [ reg rd; imm v ] ]
  else
    let lo = (v land 0xFFF lxor 0x800) - 0x800 in
    let hi = ((v - lo) asr 12) land 0xFFFFF in
    insn "lui" [ reg rd; imm hi ]
    :: (if lo = 0 then [] else [ insn "addi" [ reg rd; reg rd; imm lo ] ])
