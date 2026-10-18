(* The binary32 operations scalar and vector forms share, by kind. *)

(* binary32 compares into a general register: 1 when the ordered relation
   holds, 0 otherwise or when either operand is a NaN. *)
module Fcmp = struct
  type t = Eq | Le | Lt

  let name = function Eq -> "feq.s" | Le -> "fle.s" | Lt -> "flt.s"
  let vname = function Eq -> "vmfeq.vv" | Le -> "vmfle.vv" | Lt -> "vmflt.vv"
end

(* binary32 arithmetic. [Max] and [Min] are IEEE 754-2019 maximumNumber and
   minimumNumber: a NaN operand gives the other operand. *)
module Fop = struct
  type t = Add | Div | Max | Min | Mul | Sub

  let name = function
    | Add -> "fadd.s"
    | Div -> "fdiv.s"
    | Max -> "fmax.s"
    | Min -> "fmin.s"
    | Mul -> "fmul.s"
    | Sub -> "fsub.s"

  let vname = function
    | Add -> "vfadd.vv"
    | Div -> "vfdiv.vv"
    | Max -> "vfmax.vv"
    | Min -> "vfmin.vv"
    | Mul -> "vfmul.vv"
    | Sub -> "vfsub.vv"
end

(* Sign injection: the first operand's magnitude with the second's sign, its
   complement, or the two signs' exclusive or. *)
module Sgnj = struct
  type t = J | Jn | Jx

  let name = function J -> "fsgnj.s" | Jn -> "fsgnjn.s" | Jx -> "fsgnjx.s"

  let vname = function
    | J -> "vfsgnj.vv"
    | Jn -> "vfsgnjn.vv"
    | Jx -> "vfsgnjx.vv"
end

(* Lanewise logic on 32-bit lanes (a mask is lanes of all ones or zeros). *)
module Logic = struct
  type t = And | Or | Xor

  let name = function And -> "vand.vv" | Or -> "vor.vv" | Xor -> "vxor.vv"
end

(* Lane shifts by an immediate. *)
module Shift = struct
  type t = Sll | Sra | Srl

  let name = function Sll -> "vsll.vi" | Sra -> "vsra.vi" | Srl -> "vsrl.vi"
end
