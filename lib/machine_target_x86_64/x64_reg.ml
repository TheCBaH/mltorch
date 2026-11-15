(* x86-64 register units and views. A 64-bit GPR has its 32-bit view (a
   32-bit write zero-extends into the whole register; 8- and 16-bit views are
   not admitted); an XMM register is 128 bits with its scalar S and D views
   (legacy-SSE scalar arithmetic preserves the bits above them). RFLAGS is
   kept as six bits — CF 1, PF 2, AF 4, ZF 8, SF 16, OF 32 — and MXCSR as a
   control register. *)

open Machine_ir
module V = Mir_target.View

(* A register's number within its bank: rax and xmm0 are each number 0. *)
module Num = Mir_target.Reg_num

(* Sequences of register numbers: a range written as literals, a pool, and the
   register the k-th scratch operand is placed on from a base. *)
module Num_seq = struct
  let range lo hi = List.init (hi - lo + 1) (fun i -> Num.of_int (lo + i))
  let of_ints = List.map Num.of_int

  let at base slot =
    Num.of_int (Num.to_int base + Mir_target.Scratch_slot.to_int slot)
end

let names64 =
  [|
    "rax";
    "rcx";
    "rdx";
    "rbx";
    "rsp";
    "rbp";
    "rsi";
    "rdi";
    "r8";
    "r9";
    "r10";
    "r11";
    "r12";
    "r13";
    "r14";
    "r15";
  |]

let names32 =
  [|
    "eax";
    "ecx";
    "edx";
    "ebx";
    "esp";
    "ebp";
    "esi";
    "edi";
    "r8d";
    "r9d";
    "r10d";
    "r11d";
    "r12d";
    "r13d";
    "r14d";
    "r15d";
  |]

let gpr (bits : Mir_target.Bits.t) k =
  let k = Num.to_int k in
  {
    V.name = (if (bits :> int) = 64 then names64.(k) else names32.(k));
    bank = Mir_target.Bank.Gpr;
    unit = Mir_id.Unit.of_int k;
    span = Mir_target.Span.One;
    lo = Mir_target.Bits.of_int 0;
    bits;
  }

let q = gpr (Mir_target.Bits.of_int 64)
let d = gpr (Mir_target.Bits.of_int 32)

let xmm_view (bits : Mir_target.Bits.t) k =
  let k = Num.to_int k in
  {
    V.name =
      Printf.sprintf "xmm%d%s" k
        (match (bits :> int) with 128 -> "" | 64 -> ".d" | _ -> ".s");
    bank = Mir_target.Bank.Fpr;
    unit = Mir_id.Unit.of_int (16 + k);
    span = Mir_target.Span.One;
    lo = Mir_target.Bits.of_int 0;
    bits;
  }

let xmm = xmm_view (Mir_target.Bits.of_int 128)
let sd = xmm_view (Mir_target.Bits.of_int 64)
let ss = xmm_view (Mir_target.Bits.of_int 32)

(* The register number a unit names, when it is one of the sixteen registers
   of that file; XMM units follow the sixteen general registers. *)
let num_of_unit ~(first : Mir_id.Unit.t) u =
  let k = Mir_id.Unit.to_int u - Mir_id.Unit.to_int first in
  if k >= 0 && k <= 15 then Some (Num.of_int k) else None

let gpr_num_of_unit = num_of_unit ~first:(Mir_id.Unit.of_int 0)
let xmm_num_of_unit = num_of_unit ~first:(Mir_id.Unit.of_int 16)

let rflags =
  {
    V.name = "rflags";
    bank = Mir_target.Bank.Flags;
    unit = Mir_id.Unit.of_int 32;
    span = Mir_target.Span.One;
    lo = Mir_target.Bits.of_int 0;
    bits = Mir_target.Bits.of_int 6;
  }

let mxcsr =
  {
    V.name = "mxcsr";
    bank = Mir_target.Bank.Control;
    unit = Mir_id.Unit.of_int 33;
    span = Mir_target.Span.One;
    lo = Mir_target.Bits.of_int 0;
    bits = Mir_target.Bits.of_int 32;
  }

let rax = Num.of_int 0
and rcx = Num.of_int 1
and rdx = Num.of_int 2
and rbx = Num.of_int 3
and rsp = Num.of_int 4
and rbp = Num.of_int 5
and rsi = Num.of_int 6
and rdi = Num.of_int 7

let r8 = Num.of_int 8
and r9 = Num.of_int 9
and r10 = Num.of_int 10
and r11 = Num.of_int 11

(* rsp the stack pointer, rbp the frame pointer, r10 and r11 frame and
   control scratch: none allocatable. *)
let reserved = List.map q [ rsp; rbp; r10; r11 ]

let abi =
  {
    Mir_target.Abi.int_args = List.map q [ rdi; rsi; rdx; rcx; r8; r9 ];
    fp_args = List.map sd (Num_seq.range 0 7);
    int_results = List.map q [ rax; rdx ];
    fp_results = List.map sd (Num_seq.of_ints [ 0; 1 ]);
    (* rbx, rbp and r12-r15 whole; MXCSR's control bits as the caller set
       them; every XMM register is caller-saved *)
    preserved = List.map q ([ rbx; rbp ] @ Num_seq.range 12 15) @ [ mxcsr ];
    reserved;
    stack_align = 16L;
  }
