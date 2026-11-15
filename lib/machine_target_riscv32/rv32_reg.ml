(* RV32 register units, their views and the ILP32F contract. Every general
   register x0-x31 and every floating-point register f0-f31 is one 32-bit unit
   with one view (FLEN is 32: there is no D extension). x0 reads zero and is
   never allocated. RISC-V has no condition register: a compare writes a
   general register and a branch compares registers itself, so the condition
   view below is never written by any form. Register numbering and the calling
   convention follow the RISC-V ELF psABI (ILP32F). *)

open Machine_ir
module V = Mir_target.View

(* A register's number within its bank: x5, f5 and v5 are each number 5. It
   also counts a sequence from its first member (the k-th argument register
   is a0 plus k), which is why two numbers add. *)
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
  let add a b = Num.of_int (Num.to_int a + Num.to_int b)
  let range lo hi = List.init (hi - lo + 1) (fun i -> Num.of_int (lo + i))
  let of_ints = List.map Num.of_int
end

let gpr_unit k = Mir_id.Unit.of_int (Num.to_int k)
let fpr_unit k = Mir_id.Unit.of_int (32 + Num.to_int k)

let gpr_names =
  [|
    "zero";
    "ra";
    "sp";
    "gp";
    "tp";
    "t0";
    "t1";
    "t2";
    "s0";
    "s1";
    "a0";
    "a1";
    "a2";
    "a3";
    "a4";
    "a5";
    "a6";
    "a7";
    "s2";
    "s3";
    "s4";
    "s5";
    "s6";
    "s7";
    "s8";
    "s9";
    "s10";
    "s11";
    "t3";
    "t4";
    "t5";
    "t6";
  |]

let fpr_names =
  [|
    "ft0";
    "ft1";
    "ft2";
    "ft3";
    "ft4";
    "ft5";
    "ft6";
    "ft7";
    "fs0";
    "fs1";
    "fa0";
    "fa1";
    "fa2";
    "fa3";
    "fa4";
    "fa5";
    "fa6";
    "fa7";
    "fs2";
    "fs3";
    "fs4";
    "fs5";
    "fs6";
    "fs7";
    "fs8";
    "fs9";
    "fs10";
    "fs11";
    "ft8";
    "ft9";
    "ft10";
    "ft11";
  |]

let x k =
  {
    V.name = gpr_names.(Num.to_int k);
    bank = Mir_target.Bank.Gpr;
    unit = gpr_unit k;
    span = Mir_target.Span.One;
    lo = Mir_target.Bits.of_int 0;
    bits = Mir_target.Bits.of_int 32;
  }

let f k =
  {
    V.name = fpr_names.(Num.to_int k);
    bank = Mir_target.Bank.Fpr;
    unit = fpr_unit k;
    span = Mir_target.Span.One;
    lo = Mir_target.Bits.of_int 0;
    bits = Mir_target.Bits.of_int 32;
  }

(* The ABI names of the registers the target refers to by role. *)
let zero = x (Num.of_int 0)
let ra = x (Num.of_int 1)
let sp = x (Num.of_int 2)
let gp = x (Num.of_int 3)
let tp = x (Num.of_int 4)
let a k = x (Num_seq.add (Num.of_int 10) k)
let fa k = f (Num_seq.add (Num.of_int 10) k)

(* t6 holds the address of a large frame offset; s11 the table of region
   addresses a table-bound image is entered with, which every callee keeps.
   Neither is allocatable. *)
let scratch = x (Num.of_int 31)
let table = x (Num.of_int 27)

(* A condition register no form writes: the selected stage needs one. *)
let flags =
  {
    V.name = "flags";
    bank = Mir_target.Bank.Flags;
    unit = Mir_id.Unit.of_int 64;
    span = Mir_target.Span.One;
    lo = Mir_target.Bits.of_int 0;
    bits = Mir_target.Bits.of_int 1;
  }

(* The vector registers v0-v31 (Zve32f), each one unit of VLEN bits, modelled
   at VLEN 128: a binary32 vector's four lanes, an int32 vector's, or a mask
   of four all-ones or all-zero lanes. An RVV mask operand is always v0, so the
   forms whose sequence needs one (a compare's bits before they become lanes,
   a slide's temporary) use v0 in passing: they clobber it and admit no
   operand there ([Rv32_op.clobbers], [Rv32_op.constraints]). Otherwise v0
   holds values like any other. Every vector register is caller-saved (the
   psABI preserves none). *)
let vector_unit k = Mir_id.Unit.of_int (65 + Num.to_int k)

let v k =
  {
    V.name = Printf.sprintf "v%d" (Num.to_int k);
    bank = Mir_target.Bank.Vector;
    unit = vector_unit k;
    span = Mir_target.Span.One;
    lo = Mir_target.Bits.of_int 0;
    bits = Mir_target.Bits.of_int 128;
  }

(* The register group of [span] based at register [k] (RVV requires the base
   to be a multiple of the group's size), named by its base, as the emitted
   instructions name it; the configuration says the size. *)
let group span k =
  let i = Num.to_int k in
  if i < 0 || i >= 32 || i mod Mir_target.Span.count span <> 0 then
    Err.fail (`Unaligned_group (span, k))
  else
    Err.return
      {
        V.name = Printf.sprintf "v%d" i;
        bank = Mir_target.Bank.Vector;
        unit = vector_unit k;
        span;
        lo = Mir_target.Bits.of_int 0;
        bits = Mir_target.Bits.of_int (128 * Mir_target.Span.count span);
      }

let pp_group_error ppf (`Unaligned_group (span, k)) =
  Fmt.pf ppf "no %s register group at v%d"
    (Mir_target.Span.name span)
    (Num.to_int k)

let vector_scratch = v Num_seq.zero
let reserved = [ zero; ra; sp; gp; tp; scratch; table ]

let abi =
  {
    Mir_target.Abi.int_args = List.map a (Num_seq.range 0 7);
    fp_args = List.map fa (Num_seq.range 0 7);
    int_results = [ a Num_seq.zero ];
    fp_results = [ fa Num_seq.zero ];
    (* s0-s10 and fs0-fs11; s11 is reserved for the table *)
    preserved =
      List.map x (Num_seq.of_ints [ 8; 9 ] @ Num_seq.range 18 26)
      @ List.map f (Num_seq.of_ints [ 8; 9 ] @ Num_seq.range 18 27);
    reserved;
    stack_align = 16L;
  }

(* The registers a value may occupy: every general register that is not
   reserved, every floating-point register. *)
let allocatable_gpr =
  List.filter
    (fun k -> not (List.exists (fun (v : V.t) -> V.equal v (x k)) reserved))
    (Num_seq.range 0 31)

let gpr32 =
  { Mir_target.Class.name = "gpr32"; views = List.map x allocatable_gpr }

let fpr32 =
  { Mir_target.Class.name = "fpr32"; views = List.map f (Num_seq.range 0 31) }
