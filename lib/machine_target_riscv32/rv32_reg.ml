(* RV32 register units, their views and the ILP32F contract. Every general
   register x0-x31 and every floating-point register f0-f31 is one 32-bit unit
   with one view (FLEN is 32: there is no D extension). x0 reads zero and is
   never allocated. RISC-V has no condition register: a compare writes a
   general register and a branch compares registers itself, so the condition
   view below is never written by any form. Register numbering and the calling
   convention follow the RISC-V ELF psABI (ILP32F). *)

open Machine_ir
module V = Mir_target.View

let gpr_unit k = Mir_id.Unit.of_int k
let fpr_unit k = Mir_id.Unit.of_int (32 + k)

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
    V.name = gpr_names.(k);
    bank = Mir_target.Bank.Gpr;
    unit = gpr_unit k;
    span = Mir_target.Span.One;
    lo = 0;
    bits = 32;
  }

let f k =
  {
    V.name = fpr_names.(k);
    bank = Mir_target.Bank.Fpr;
    unit = fpr_unit k;
    span = Mir_target.Span.One;
    lo = 0;
    bits = 32;
  }

(* The ABI names of the registers the target refers to by role. *)
let zero = x 0
let ra = x 1
let sp = x 2
let gp = x 3
let tp = x 4
let a k = x (10 + k)
let fa k = f (10 + k)

(* t6 holds the address of a large frame offset; s11 the table of region
   addresses a table-bound image is entered with, which every callee keeps.
   Neither is allocatable. *)
let scratch = x 31
let table = x 27

(* A condition register no form writes: the selected stage needs one. *)
let flags =
  {
    V.name = "flags";
    bank = Mir_target.Bank.Flags;
    unit = Mir_id.Unit.of_int 64;
    span = Mir_target.Span.One;
    lo = 0;
    bits = 1;
  }

(* The vector registers v0-v31 (Zve32f), each one unit of VLEN bits, modelled
   at VLEN 128: a binary32 vector's four lanes, an int32 vector's, or a mask
   of four all-ones or all-zero lanes. An RVV mask operand is always v0, so the
   forms whose sequence needs one (a compare's bits before they become lanes,
   a slide's temporary) use v0 in passing: they clobber it and admit no
   operand there ([Rv32_op.clobbers], [Rv32_op.constraints]). Otherwise v0
   holds values like any other. Every vector register is caller-saved (the
   psABI preserves none). *)
let vector_unit k = Mir_id.Unit.of_int (65 + k)

let v k =
  {
    V.name = Printf.sprintf "v%d" k;
    bank = Mir_target.Bank.Vector;
    unit = vector_unit k;
    span = Mir_target.Span.One;
    lo = 0;
    bits = 128;
  }

(* The register group of [span] based at register [k] (RVV requires the base
   to be a multiple of the group's size), named by its base, as the emitted
   instructions name it; the configuration says the size. *)
let group span k =
  if k < 0 || k >= 32 || k mod Mir_target.Span.count span <> 0 then
    Err.fail (`Unaligned_group (span, k))
  else
    Err.return
      {
        V.name = Printf.sprintf "v%d" k;
        bank = Mir_target.Bank.Vector;
        unit = vector_unit k;
        span;
        lo = 0;
        bits = 128 * Mir_target.Span.count span;
      }

let pp_group_error ppf (`Unaligned_group (span, k)) =
  Fmt.pf ppf "no %s register group at v%d" (Mir_target.Span.name span) k

let vector_scratch = v 0
let range a b = List.init (b - a + 1) (fun i -> a + i)
let reserved = [ zero; ra; sp; gp; tp; scratch; table ]

let abi =
  {
    Mir_target.Abi.int_args = List.map a (range 0 7);
    fp_args = List.map fa (range 0 7);
    int_results = [ a 0 ];
    fp_results = [ fa 0 ];
    (* s0-s10 and fs0-fs11; s11 is reserved for the table *)
    preserved =
      List.map x ([ 8; 9 ] @ range 18 26) @ List.map f ([ 8; 9 ] @ range 18 27);
    reserved;
    stack_align = 16L;
  }

(* The registers a value may occupy: every general register that is not
   reserved, every floating-point register. *)
let allocatable_gpr =
  List.filter
    (fun k -> not (List.exists (fun (v : V.t) -> V.equal v (x k)) reserved))
    (range 0 31)

let gpr32 =
  { Mir_target.Class.name = "gpr32"; views = List.map x allocatable_gpr }

let fpr32 = { Mir_target.Class.name = "fpr32"; views = List.map f (range 0 31) }
