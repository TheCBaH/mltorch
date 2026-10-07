(* x86-64 register units and views. A 64-bit GPR has its 32-bit view (a
   32-bit write zero-extends into the whole register; 8- and 16-bit views are
   not admitted); an XMM register is 128 bits with its scalar S and D views
   (legacy-SSE scalar arithmetic preserves the bits above them). RFLAGS is
   kept as six bits — CF 1, PF 2, AF 4, ZF 8, SF 16, OF 32 — and MXCSR as a
   control register. *)

open Machine_ir
module V = Mir_target.View

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

let gpr bits k =
  {
    V.name = (if bits = 64 then names64.(k) else names32.(k));
    bank = Mir_target.Bank.Gpr;
    unit = Mir_id.Unit.of_int k;
    lo = 0;
    bits;
  }

let q = gpr 64
let d = gpr 32

let xmm_view bits k =
  {
    V.name =
      Printf.sprintf "xmm%d%s" k
        (match bits with 128 -> "" | 64 -> ".d" | _ -> ".s");
    bank = Mir_target.Bank.Fpr;
    unit = Mir_id.Unit.of_int (16 + k);
    lo = 0;
    bits;
  }

let xmm = xmm_view 128
let sd = xmm_view 64
let ss = xmm_view 32

let rflags =
  {
    V.name = "rflags";
    bank = Mir_target.Bank.Flags;
    unit = Mir_id.Unit.of_int 32;
    lo = 0;
    bits = 6;
  }

let mxcsr =
  {
    V.name = "mxcsr";
    bank = Mir_target.Bank.Control;
    unit = Mir_id.Unit.of_int 33;
    lo = 0;
    bits = 32;
  }

let rax = 0
and rcx = 1
and rdx = 2
and rbx = 3
and rsp = 4
and rbp = 5
and rsi = 6
and rdi = 7

let range a b = List.init (b - a + 1) (fun i -> a + i)

(* rsp the stack pointer, rbp the frame pointer, r10 and r11 frame and
   control scratch: none allocatable. *)
let reserved = List.map q [ rsp; rbp; 10; 11 ]

let abi =
  {
    Mir_target.Abi.int_args = List.map q [ rdi; rsi; rdx; rcx; 8; 9 ];
    fp_args = List.map sd (range 0 7);
    int_results = List.map q [ rax; rdx ];
    fp_results = List.map sd [ 0; 1 ];
    (* rbx, rbp and r12-r15 whole; MXCSR's control bits as the caller set
       them; every XMM register is caller-saved *)
    preserved = List.map q ([ rbx; rbp ] @ range 12 15) @ [ mxcsr ];
    reserved;
    stack_align = 16L;
  }
