(* The interface every target implements, kept in the pure common layer so
   target libraries depend on it and nothing here depends on a target.

   Hardware capability, assembler capability and interpreter capability are
   three separate properties: a form can be interpretable (its semantics are
   modelled) before any assembler accepts it, and a host may lack a feature an
   assembler knows. *)

(* An instruction-set feature. Closed across targets, alphabetical. *)
module Feature = struct
  type t =
    | Avx
    | Avx2
    | Fma  (** x86-64 FMA3 *)
    | Fp  (** AArch64 scalar floating point *)
    | Neon  (** AArch64 Advanced SIMD *)
    | Sse2  (** the x86-64 baseline *)
    | Sse41

  let name = function
    | Avx -> "avx"
    | Avx2 -> "avx2"
    | Fma -> "fma"
    | Fp -> "fp"
    | Neon -> "neon"
    | Sse2 -> "sse2"
    | Sse41 -> "sse4.1"
end

(* A register bank: storage of one kind. Views of one bank overlap; banks do
   not. *)
module Bank = struct
  type t = Control | Flags | Fpr | Gpr

  let name = function
    | Control -> "control"
    | Flags -> "flags"
    | Fpr -> "fpr"
    | Gpr -> "gpr"
end

(* A physical register view: [bits] bits of a register unit starting at bit
   [lo]. W0 and X0 are two views of one unit; S0, D0 and Q0 of another. *)
module View = struct
  type t = {
    name : string;
    bank : Bank.t;
    unit : Mir_id.Unit.t;
    lo : int;
    bits : int;
  }

  let equal (a : t) b = a = b

  (* Whether two views share any bit. *)
  let overlap a b =
    Mir_id.Unit.equal a.unit b.unit
    && a.lo < b.lo + b.bits
    && b.lo < a.lo + a.bits
end

(* How writing a destination view affects the rest of its unit. *)
module Write = struct
  type t =
    | Merge  (** bits outside the view keep their value *)
    | Undefined_upper
        (** bits above the view hold something the model does not track (a
            whole-register bitwise or copy form): nothing may rely on them *)
    | Zero_upper  (** every bit above the view becomes zero *)
end

(* A register-allocation constraint on an instruction's operands, by position
   among its value uses or results. *)
module Constraint = struct
  type t =
    | Early_clobber of int
        (** result [k] is written before every use is read: it may share no
            location with a use *)
    | Fixed_result of { result : int; view : View.t }
    | Fixed_use of { use : int; view : View.t }
    | Tied of { result : int; use : int }
        (** a destructive form: result [k] occupies use [j]'s location; the
            virtual values stay distinct *)
end

(* A register class: the views an operand of a given type may occupy. *)
module Class = struct
  type t = { name : string; views : View.t list }
end

(* The calling convention: argument and result views in order, the views a
   call preserves (with their preserved bit range), the stack alignment at a
   call and the reserved views no allocation may use. *)
module Abi = struct
  type t = {
    int_args : View.t list;
    fp_args : View.t list;
    int_results : View.t list;
    fp_results : View.t list;
    preserved : View.t list;
        (** the preserved bits of each callee-saved unit *)
    reserved : View.t list;
    stack_align : int64;
  }
end

(* A symbolic address a form encodes, which a native realization resolves by
   relocation: part of a view's address, or a call's target. *)
module Reference = struct
  module Form = struct
    type t =
      | Page  (** the 4 KiB page of the address (AArch64 ADRP) *)
      | Page_offset  (** its low 12 bits (AArch64 ADD :lo12:) *)
      | Pc_relative  (** a 32-bit displacement from the next instruction *)

    let name = function
      | Page -> "page"
      | Page_offset -> "page_offset"
      | Pc_relative -> "pc_relative"
  end

  type t = Call of Mir_op.Callee.t | View of Mir_id.View.t * Form.t
end

(* The specification a selected form's semantics cite. *)
module Source = struct
  type t = { document : string; revision : string }
end
