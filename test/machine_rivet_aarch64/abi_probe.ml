(* An AAPCS64 probe around a kernel, built as typed Rivet instructions. It
   saves what it is about to disturb, seeds the callee-saved registers
   (x19-x29, d8-d15) and FPCR from the caller's words, calls the kernel's entry,
   records the registers, FPCR and stack pointer it comes back with, and
   restores the caller's state. A kernel that keeps the ABI hands back every
   seed and the FPCR it was given; one that does not shows which.

   The words, 8 bytes each:
   in  0-9    x19-x28           out 20-29  x19-x28 after the call
       10-17  d8-d15                30-37  d8-d15 after
       18     FPCR                  38     FPCR after
       19     x29                   39     the kernel's x0
                                    40     x29 after
                                    41     sp after
                                    42     sp before *)

module O = Aarch64.Operand
module I = Aarch64.Instruction
module Op = Aarch64.Opcode

let words = 43
let origin = Foundation.Origin.synthesized ~pass:"abi_probe" ()
let x n = { Aarch64.Reg.num = n; width = 64; is_sp = false }
let sp = { Aarch64.Reg.num = 31; width = 64; is_sp = true }
let reg n = O.Reg (x n)
let freg n = O.Freg { Aarch64.Freg.num = n; double = true }
let imm n = O.Imm (Foundation.Bigint.of_int n)

let mem base off =
  O.Mem
    {
      Aarch64.Mem.base;
      offset = Aarch64.Disp.Const (Int64.of_int off);
      writeback = false;
      pre = true;
    }

let ins op ops = { I.op; ops }
let str r base off = ins Op.Str [ reg r; mem base off ]
let ldr r base off = ins Op.Ldr [ reg r; mem base off ]
let strd d base off = ins Op.Str [ freg d; mem base off ]
let ldrd d base off = ins Op.Ldr [ freg d; mem base off ]
let frame = 192
let io_slot = 160
let fpcr_slot = 168
let sp_slot = 176

let module_ ~kernel =
  let open Asm_core in
  let instr i = Normalized_ast.Instruction { insn = i; origin } in
  let dir directive = Normalized_ast.Directive { directive; origin } in
  let callee = [ 19; 20; 21; 22; 23; 24; 25; 26; 27; 28 ] in
  let fp = [ 8; 9; 10; 11; 12; 13; 14; 15 ] in
  let save =
    ins Op.Sub [ O.Reg sp; O.Reg sp; imm frame ]
    :: str 29 sp 0 :: str 30 sp 8
    :: List.mapi (fun i r -> str r sp (16 + (8 * i))) callee
    @ List.mapi (fun i d -> strd d sp (96 + (8 * i))) fp
    @ [
        str 0 sp io_slot;
        ins Op.Mrs [ reg 9; O.Sym (Expr.Symbol "fpcr") ];
        str 9 sp fpcr_slot;
        ins Op.Mov [ reg 9; O.Reg sp ];
        str 9 sp sp_slot;
      ]
  in
  let seed =
    List.mapi (fun i r -> ldr r (x 0) (8 * i)) callee
    @ List.mapi (fun i d -> ldrd d (x 0) (80 + (8 * i))) fp
    @ [
        ldr 29 (x 0) 152;
        ldr 9 (x 0) 144;
        ins Op.Msr [ O.Sym (Expr.Symbol "fpcr"); reg 9 ];
        ins Op.Bl [ O.Sym (Expr.Symbol kernel) ];
      ]
  in
  let record =
    [ ins Op.Ldr [ reg 9; mem sp io_slot ] ]
    @ List.mapi (fun i r -> str r (x 9) (160 + (8 * i))) callee
    @ List.mapi (fun i d -> strd d (x 9) (240 + (8 * i))) fp
    @ [
        ins Op.Mrs [ reg 10; O.Sym (Expr.Symbol "fpcr") ];
        str 10 (x 9) 304;
        str 0 (x 9) 312;
        str 29 (x 9) 320;
        ins Op.Mov [ reg 10; O.Reg sp ];
        str 10 (x 9) 328;
        ins Op.Ldr [ reg 11; mem sp sp_slot ];
        str 11 (x 9) 336;
      ]
  in
  let restore =
    [
      ins Op.Ldr [ reg 10; mem sp fpcr_slot ];
      ins Op.Msr [ O.Sym (Expr.Symbol "fpcr"); reg 10 ];
    ]
    @ List.mapi (fun i d -> ldrd d sp (96 + (8 * i))) fp
    @ List.mapi (fun i r -> ldr r sp (16 + (8 * i))) callee
    @ [
        ins Op.Ldr [ reg 29; mem sp 0 ];
        ins Op.Ldr [ reg 30; mem sp 8 ];
        ins Op.Add [ O.Reg sp; O.Reg sp; imm frame ];
        ins Op.Ret [];
      ]
  in
  {
    Normalized_ast.unit_name = "abi_probe";
    items =
      [
        dir
          (Directive.Section
             { name = ".text"; perms = Perms.rx; nobits = false });
        dir (Directive.Align { boundary = 4 });
        dir (Directive.Global { name = "abi_probe" });
        dir
          (Directive.Sym_type { name = "abi_probe"; kind = Directive.Function });
        Normalized_ast.Label { name = "abi_probe"; origin };
      ]
      @ List.map instr (save @ seed @ record @ restore)
      @ [
          dir
            (Directive.Sym_size
               {
                 name = "abi_probe";
                 size =
                   Expr.Binary
                     (Expr.Sub, Expr.Current_location, Expr.Symbol "abi_probe");
               });
        ];
  }

let entry = "abi_probe"

(* FPCR with round toward zero, flush to zero and default NaN: every control the
   kernel's own arithmetic would notice, and none that traps. *)
let altered_fpcr = 0x03C0_0000L
let seed k = Int64.add 0x5EED_0000_0000_0000L (Int64.of_int (0x1111 * (k + 1)))

let input ~fpcr =
  let io = Bigarray.Array1.create Bigarray.char Bigarray.c_layout (8 * words) in
  Bigarray.Array1.fill io '\000';
  let set w v =
    for b = 0 to 7 do
      io.{(8 * w) + b} <-
        Char.chr
          (Int64.to_int
             (Int64.logand (Int64.shift_right_logical v (8 * b)) 0xFFL))
    done
  in
  for w = 0 to 17 do
    set w (seed w)
  done;
  set 18 fpcr;
  set 19 (seed 19);
  io

let word io w =
  let v = ref 0L in
  for b = 7 downto 0 do
    v :=
      Int64.logor (Int64.shift_left !v 8)
        (Int64.of_int (Char.code io.{(8 * w) + b}))
  done;
  !v

(* What the kernel failed to keep, as names. *)
let violations io ~fpcr =
  let bad = ref [] in
  let check name ok = if not ok then bad := name :: !bad in
  List.iteri
    (fun i r ->
      check (Printf.sprintf "x%d" r) (Int64.equal (word io (20 + i)) (seed i)))
    [ 19; 20; 21; 22; 23; 24; 25; 26; 27; 28 ];
  List.iteri
    (fun i d ->
      check (Printf.sprintf "d%d" d)
        (Int64.equal (word io (30 + i)) (seed (10 + i))))
    [ 8; 9; 10; 11; 12; 13; 14; 15 ];
  check "x29" (Int64.equal (word io 40) (seed 19));
  check "fpcr" (Int64.equal (word io 38) fpcr);
  check "sp" (Int64.equal (word io 41) (word io 42));
  List.rev !bad
