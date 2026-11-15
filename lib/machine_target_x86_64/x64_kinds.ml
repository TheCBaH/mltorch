(* The operand kinds of the x86-64 selected forms: sizes, conditions, ALU and
   shift operations, packed formats and addresses. [X64_op] includes this. *)

open Machine_ir

module Sz = struct
  type t = L | Q

  let bits = function
    | L -> Mir_target.Bits.of_int 32
    | Q -> Mir_target.Bits.of_int 64

  let name = function L -> "l" | Q -> "q"
end

module Fsz = struct
  type t = D | S

  let name = function D -> "sd" | S -> "ss"
end

(* A memory access size and the register file it moves. *)
module Msz = struct
  type t = B | D | H | L | Q | S

  let bytes = function B -> 1L | H -> 2L | L | S -> 4L | D | Q -> 8L

  let name = function
    | B -> "b"
    | D -> "sd"
    | H -> "w"
    | L -> "l"
    | Q -> "q"
    | S -> "ss"
end

(* Our compact RFLAGS: CF 1, PF 2, AF 4, ZF 8, SF 16, OF 32. *)
module Flag = struct
  let cf = 1L
  let pf = 2L
  let af = 4L
  let zf = 8L
  let sf = 16L
  let of_ = 32L
  let all = 63L
end

module Cond = struct
  type t = A | Ae | B | Be | E | G | Ge | L | Le | Ne | Np | P

  let reads = function
    | A | Be -> Int64.logor Flag.cf Flag.zf
    | Ae | B -> Flag.cf
    | E | Ne -> Flag.zf
    | G | Le -> Int64.logor Flag.zf (Int64.logor Flag.sf Flag.of_)
    | Ge | L -> Int64.logor Flag.sf Flag.of_
    | Np | P -> Flag.pf

  let holds t f =
    let bit b = not (Int64.equal (Int64.logand f b) 0L) in
    match t with
    | A -> (not (bit Flag.cf)) && not (bit Flag.zf)
    | Ae -> not (bit Flag.cf)
    | B -> bit Flag.cf
    | Be -> bit Flag.cf || bit Flag.zf
    | E -> bit Flag.zf
    | G -> (not (bit Flag.zf)) && bit Flag.sf = bit Flag.of_
    | Ge -> bit Flag.sf = bit Flag.of_
    | L -> bit Flag.sf <> bit Flag.of_
    | Le -> bit Flag.zf || bit Flag.sf <> bit Flag.of_
    | Ne -> not (bit Flag.zf)
    | Np -> not (bit Flag.pf)
    | P -> bit Flag.pf

  let name = function
    | A -> "a"
    | Ae -> "ae"
    | B -> "b"
    | Be -> "be"
    | E -> "e"
    | G -> "g"
    | Ge -> "ge"
    | L -> "l"
    | Le -> "le"
    | Ne -> "ne"
    | Np -> "np"
    | P -> "p"
end

module Alu = struct
  type t = Add | And | Or | Sub | Xor

  let name = function
    | Add -> "add"
    | And -> "and"
    | Or -> "or"
    | Sub -> "sub"
    | Xor -> "xor"
end

module Shift = struct
  type t = Sar | Shl | Shr

  let name = function Sar -> "sar" | Shl -> "shl" | Shr -> "shr"
end

module Fop = struct
  (* [Max] is MAXSD/MAXSS: the source operand when either is a NaN or both
     are zero, not IEEE maximum *)
  type t = Add | Div | Max | Mul | Sub

  let name = function
    | Add -> "add"
    | Div -> "div"
    | Max -> "max"
    | Mul -> "mul"
    | Sub -> "sub"
end

module Flogic = struct
  type t = And | Andn | Or | Xor

  let name = function
    | And -> "andp"
    | Andn -> "andnp"
    | Or -> "orp"
    | Xor -> "xorp"
end

(* CMPSD/CMPSS and CMPPS/CMPPD predicates admitted: 0 (equal, ordered), 1
   (less than, ordered), 2 (less or equal, ordered) and 3 (unordered). *)
module Cmp_pred = struct
  type t = Eq | Le | Lt | Unord

  let imm = function Eq -> 0L | Lt -> 1L | Le -> 2L | Unord -> 3L
  let name = function Eq -> "eq" | Le -> "le" | Lt -> "lt" | Unord -> "unord"
end

(* A packed format: four binary32 lanes (PS) or two binary64 lanes (PD) of an
   XMM register. *)
module Pk = struct
  type t = Pd | Ps

  let fsz = function Pd -> Fsz.D | Ps -> Fsz.S

  let lanes = function
    | Pd -> Mir_type.Lanes.of_int 2
    | Ps -> Mir_type.Lanes.of_int 4

  let ty t =
    Mir_type.Vec
      ( (match t with Pd -> Mir_type.Elem.F64 | Ps -> Mir_type.Elem.F32),
        lanes t )

  (* the mask a packed compare of these lanes fills: every bit of a lane set or
     clear *)
  let mask_ty t = Mir_type.Mask (lanes t)
  let name = function Pd -> "pd" | Ps -> "ps"
end

(* Two binary32 lanes in an XMM register's low half. *)
let half_ty = Mir_type.Vec (Mir_type.Elem.F32, Mir_type.Lanes.of_int 2)

type v = Mir_value.t

(* [base + index * scale + disp]: scale 1, 2, 4 or 8; disp a signed 32-bit
   value *)
module Addr = struct
  type t = { base : v; index : (v * int64) option; disp : int64 }

  let uses a = a.base :: Option.to_list (Option.map fst a.index)
end
