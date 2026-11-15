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
    | Zve32f  (** RISC-V embedded vectors, binary32 elements *)

  let name = function
    | Avx -> "avx"
    | Avx2 -> "avx2"
    | Fma -> "fma"
    | Fp -> "fp"
    | Neon -> "neon"
    | Sse2 -> "sse2"
    | Sse41 -> "sse4.1"
    | Zve32f -> "zve32f"
end

(* A width in bits: a register view's, a value's shape, a unit's. *)
module Bits =
  Core.Tagged_int.Make
    (struct
      let prefix = "b"
    end)
    ()

(* A register's number within its bank: x5, f5 and v5 are each number 5. The
   allocation contract names a pool, a view and a position in a sequence by
   it; a target's own register constructors take the same type. *)
module Reg_num =
  Core.Tagged_int.Make
    (struct
      let prefix = "r"
    end)
    ()

(* Which scratch operand register of an instruction: the reference
   allocator reloads its k-th distinct operand into the k-th. A target places
   the sequence at a register number of its own. *)
module Scratch_slot =
  Core.Tagged_int.Make
    (struct
      let prefix = "scratch"
    end)
    ()

(* A register bank: storage of one kind. Views of one bank overlap; banks do
   not. *)
module Bank = struct
  type t = Control | Flags | Fpr | Gpr | Vector

  let name = function
    | Control -> "control"
    | Flags -> "flags"
    | Fpr -> "fpr"
    | Gpr -> "gpr"
    | Vector -> "vector"
end

(* How many consecutive units of its bank a view covers: RVV's integer
   register-group multipliers. Numeric order, not alphabetical: it is the
   order of [vtype.vlmul]'s encodings 0-3 and of increasing size. *)
module Span = struct
  type t = One | Two | Four | Eight

  (* How many consecutive units the group covers. *)
  module Count =
    Core.Tagged_int.Make
      (struct
        let prefix = "units"
      end)
      ()

  let count = function
    | One -> Count.of_int 1
    | Two -> Count.of_int 2
    | Four -> Count.of_int 4
    | Eight -> Count.of_int 8

  let name = function One -> "m1" | Two -> "m2" | Four -> "m4" | Eight -> "m8"

  (* The span a value [bits] wide needs of units [unit] wide: one for
     anything not wider than a unit. *)
  let of_width ~(unit : Bits.t) (bits : Bits.t) =
    match (bits :> int) / (unit :> int) with
    | 0 | 1 -> One
    | 2 -> Two
    | 4 -> Four
    | 8 -> Eight
    | _ -> invalid_arg "Mir_target.Span.of_width: no such register group"

  (* The register numbers a group of this span based at [base] covers, lowest
     first, and whether [base] may start one (a multiple of the span). *)
  let members t (base : Reg_num.t) =
    List.init (count t :> int) (fun i -> Reg_num.of_int ((base :> int) + i))

  let aligned t (base : Reg_num.t) = (base :> int) mod (count t :> int) = 0

  (* The lanes of a vector of [lanes] lanes that one member of the group
     holds. *)
  let share t (lanes : Mir_type.Lanes.t) =
    Mir_type.Lanes.of_int ((lanes :> int) / (count t :> int))

  (* The width of a group of this span of units each [unit] wide. *)
  let bits ~(unit : Bits.t) t = Bits.of_int ((unit :> int) * (count t :> int))

  (* Whether a group of this span based at unit [base] has [u] among its
     members, and whether it does with [u] above [base]. *)
  let covers t ~(base : Mir_id.Unit.t) (u : Mir_id.Unit.t) =
    let b = (base :> int) and u = (u :> int) in
    b <= u && u < b + (count t :> int)

  let covers_above t ~(base : Mir_id.Unit.t) (u : Mir_id.Unit.t) =
    (base :> int) < (u :> int) && covers t ~base u
end

(* A physical register view: [bits] bits of a register unit starting at bit
   [lo]. W0 and X0 are two views of one unit; S0, D0 and Q0 of another. A view
   whose [span] is not [One] is a register group: [unit] is its lowest member,
   the members are the [Span.count] consecutive units from it, and it covers
   each of them whole ([lo = 0], [bits] the members' bits). Which base
   registers may start a group is the target's rule, checked where it builds
   the view. *)
module View = struct
  type t = {
    name : string;
    bank : Bank.t;
    unit : Mir_id.Unit.t;
    span : Span.t;
    lo : Bits.t;
    bits : Bits.t;
  }

  let equal (a : t) b = a = b

  (* The member units, lowest first. *)
  let units v =
    let rec go u k =
      if k = 0 then [] else u :: go (Mir_id.Unit.succ u) (k - 1)
    in
    go v.unit (Span.count v.span :> int)

  (* Whether two views share any bit: a common unit, and, between two
     single-unit views, intersecting bit ranges (a group covers its members
     whole). *)
  let overlap a b =
    match (a.span, b.span) with
    | Span.One, Span.One ->
        Mir_id.Unit.equal a.unit b.unit
        && (a.lo :> int) < (b.lo :> int) + (b.bits :> int)
        && (b.lo :> int) < (a.lo :> int) + (a.bits :> int)
    | _ ->
        let bu = units b in
        List.exists (fun u -> List.exists (Mir_id.Unit.equal u) bu) (units a)

  (* The same register: base, span and bit range alike. What a tied operand
     must be, which a shared unit alone does not make (v4 is not the v4m4
     group). *)
  let same_register a b =
    Mir_id.Unit.equal a.unit b.unit
    && a.span = b.span && a.lo = b.lo && a.bits = b.bits
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

(* The register shape of each type on a 64-bit target whose vectors share the
   floating-point bank (AArch64, x86-64): a pointer or i64 a 64-bit general
   register, a narrower integer or predicate its 32-bit view, a float its
   precision's view, a vector a register-wide slice or its half, a mask a whole
   register of four binary32 or two binary64 lanes. *)
let lp64_shape ~(flags_bits : Bits.t) (ty : Mir_type.t) =
  let b = Bits.of_int in
  match ty with
  | Mir_type.Int Mir_width.W64 | Mir_type.Ptr -> Some (Bank.Gpr, b 64)
  | Mir_type.Int (Mir_width.W8 | Mir_width.W16 | Mir_width.W32) | Mir_type.Pred
    ->
      Some (Bank.Gpr, b 32)
  | Mir_type.F64 -> Some (Bank.Fpr, b 64)
  | Mir_type.F32 -> Some (Bank.Fpr, b 32)
  | Mir_type.Flags -> Some (Bank.Flags, flags_bits)
  | Mir_type.Vec (e, n) -> (
      match
        Int64.to_int (Mir_type.Elem.bytes e) * 8 * Mir_type.Lanes.to_int n
      with
      | (64 | 128) as bits -> Some (Bank.Fpr, b bits)
      | _ -> None)
  | Mir_type.Mask n -> (
      match Mir_type.Lanes.to_int n with
      | 2 | 4 -> Some (Bank.Fpr, b 128)
      | _ -> None)
  | Mir_type.Order -> None

(* A register-allocation constraint on an instruction's operands, by position
   among its value uses or results. *)
module Constraint = struct
  (* A position among an instruction's value uses, and one among its results:
     two numberings, so neither passes for the other. *)
  module Use_position =
    Core.Tagged_int.Make
      (struct
        let prefix = "use"
      end)
      ()

  module Result_position =
    Core.Tagged_int.Make
      (struct
        let prefix = "result"
      end)
      ()

  (* Whether a position names an element of the instruction's uses or
     results, and that element. *)
  let has_use uses (p : Use_position.t) =
    (p :> int) >= 0 && (p :> int) < List.length uses

  let has_result results (p : Result_position.t) =
    (p :> int) >= 0 && (p :> int) < List.length results

  let nth_use uses (p : Use_position.t) = List.nth uses (p :> int)
  let nth_result results (p : Result_position.t) = List.nth results (p :> int)

  type t =
    | Early_clobber of Result_position.t
        (** result [k] is written before every use is read: it may share no
            location with a use *)
    | Fixed_result of { result : Result_position.t; view : View.t }
    | Fixed_use of { use : Use_position.t; view : View.t }
    | Scratch of View.t
        (** the form's own sequence overwrites the view while it reads its uses
            and writes its results: no use or result may share it. The view is
            among the form's clobbers too. *)
    | Tied of { result : Result_position.t; use : Use_position.t }
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
