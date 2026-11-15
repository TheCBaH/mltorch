(* AArch64 register units, their views and the AAPCS64 contract. A general
   register unit has W (low 32) and X (64) views; a SIMD&FP unit has S (32),
   D (64) and Q (128) views. A W write zeroes bits 63:32; a scalar S or D write
   zeroes the rest of the 128-bit register (DDI 0487, "Registers in AArch64
   state"). *)

open Machine_ir
module V = Mir_target.View

(* A register's number within its bank: x5, w5, d5 and q5 are each number 5.
   It also counts a sequence from its first member. *)
module Num =
  Core.Tagged_int.Make
    (struct
      let prefix = "r"
    end)
    ()

(* Sequences of register numbers: a range or list written as literals, and
   the sum of two numbers where one is an offset. *)
module Num_seq = struct
  let zero = Num.of_int 0
  let range lo hi = List.init (hi - lo + 1) (fun i -> Num.of_int (lo + i))
  let of_ints = List.map Num.of_int
end

let gpr_unit k = Mir_id.Unit.of_int (Num.to_int k)
let fpr_unit k = Mir_id.Unit.of_int (32 + Num.to_int k)
let nzcv_unit = Mir_id.Unit.of_int 64

let x k =
  {
    V.name = Printf.sprintf "x%d" (Num.to_int k);
    bank = Mir_target.Bank.Gpr;
    unit = gpr_unit k;
    span = Mir_target.Span.One;
    lo = Mir_target.Bits.of_int 0;
    bits = Mir_target.Bits.of_int 64;
  }

let w k =
  {
    V.name = Printf.sprintf "w%d" (Num.to_int k);
    bank = Mir_target.Bank.Gpr;
    unit = gpr_unit k;
    span = Mir_target.Span.One;
    lo = Mir_target.Bits.of_int 0;
    bits = Mir_target.Bits.of_int 32;
  }

let q k =
  {
    V.name = Printf.sprintf "q%d" (Num.to_int k);
    bank = Mir_target.Bank.Fpr;
    unit = fpr_unit k;
    span = Mir_target.Span.One;
    lo = Mir_target.Bits.of_int 0;
    bits = Mir_target.Bits.of_int 128;
  }

let d k =
  {
    V.name = Printf.sprintf "d%d" (Num.to_int k);
    bank = Mir_target.Bank.Fpr;
    unit = fpr_unit k;
    span = Mir_target.Span.One;
    lo = Mir_target.Bits.of_int 0;
    bits = Mir_target.Bits.of_int 64;
  }

let s k =
  {
    V.name = Printf.sprintf "s%d" (Num.to_int k);
    bank = Mir_target.Bank.Fpr;
    unit = fpr_unit k;
    span = Mir_target.Span.One;
    lo = Mir_target.Bits.of_int 0;
    bits = Mir_target.Bits.of_int 32;
  }

let nzcv =
  {
    V.name = "nzcv";
    bank = Mir_target.Bank.Flags;
    unit = nzcv_unit;
    span = Mir_target.Span.One;
    lo = Mir_target.Bits.of_int 0;
    bits = Mir_target.Bits.of_int 4;
  }

let sp =
  {
    V.name = "sp";
    bank = Mir_target.Bank.Gpr;
    unit = Mir_id.Unit.of_int 65;
    span = Mir_target.Span.One;
    lo = Mir_target.Bits.of_int 0;
    bits = Mir_target.Bits.of_int 64;
  }

(* FPCR: rounding mode, flush-to-zero, default NaN and trap enables. *)
let fpcr =
  {
    V.name = "fpcr";
    bank = Mir_target.Bank.Control;
    unit = Mir_id.Unit.of_int 66;
    span = Mir_target.Span.One;
    lo = Mir_target.Bits.of_int 0;
    bits = Mir_target.Bits.of_int 64;
  }

(* x16/x17 are the intra-procedure-call scratch a linker veneer may clobber,
   x18 the platform register, x29 the frame pointer, x30 the link register;
   none is allocatable. *)
let reserved = List.map x (Num_seq.of_ints [ 16; 17; 18; 29; 30 ]) @ [ sp ]

let abi =
  {
    Mir_target.Abi.int_args = List.map x (Num_seq.range 0 7);
    fp_args = List.map d (Num_seq.range 0 7);
    int_results = [ x Num_seq.zero ];
    fp_results = [ d Num_seq.zero ];
    (* x19-x28 and the frame pointer whole; v8-v15 only their low 64 bits;
       FPCR as the caller set it *)
    preserved =
      List.map x (Num_seq.range 19 28)
      @ List.map d (Num_seq.range 8 15)
      @ [ x (Num.of_int 29); fpcr ];
    reserved;
    stack_align = 16L;
  }

let allocatable_gpr =
  List.filter
    (fun k -> not (List.mem k (Num_seq.of_ints [ 16; 17; 18; 29; 30 ])))
    (Num_seq.range 0 30)

let gpr64 =
  { Mir_target.Class.name = "gpr64"; views = List.map x allocatable_gpr }

let gpr32 =
  { Mir_target.Class.name = "gpr32"; views = List.map w allocatable_gpr }

let fpr64 =
  { Mir_target.Class.name = "fpr64"; views = List.map d (Num_seq.range 0 31) }

let fpr32 =
  { Mir_target.Class.name = "fpr32"; views = List.map s (Num_seq.range 0 31) }

(* The class a virtual value of this type occupies. *)
let class_of (ty : Mir_type.t) =
  match ty with
  | Mir_type.Int Mir_width.W64 | Mir_type.Ptr -> Some gpr64
  | Mir_type.Int (Mir_width.W8 | Mir_width.W16 | Mir_width.W32) | Mir_type.Pred
    ->
      Some gpr32
  | Mir_type.F64 -> Some fpr64
  | Mir_type.F32 -> Some fpr32
  | Mir_type.Flags | Mir_type.Mask _ | Mir_type.Order | Mir_type.Vec _ -> None
