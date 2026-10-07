(* The generic opcodes: primitive machine computation over virtual values.
   Every consumer matches this type exhaustively and alphabetically, so a new
   opcode is a compile error in the verifier, the printer and the interpreter
   until each handles it. Uses, results and effects are derived here; no pass
   supplies its own smaller list. *)

module Fbinary = struct
  (* [Max] is IEEE 754-2019 maximum: a NaN operand gives NaN, and +0 > -0. *)
  type t = Add | Div | Max | Mul | Sub

  let name = function
    | Add -> "add"
    | Div -> "div"
    | Max -> "max"
    | Mul -> "mul"
    | Sub -> "sub"
end

module Fcmp = struct
  (* [Eq], [Le] and [Lt] are ordered: false when either operand is NaN.
     [Unordered] is true exactly when one is. *)
  type t = Eq | Le | Lt | Unordered

  let name = function
    | Eq -> "oeq"
    | Le -> "ole"
    | Lt -> "olt"
    | Unordered -> "uno"
end

module Fconvert = struct
  (* Numeric conversions, each one rounding to nearest-even where inexact. *)
  type t = F32_to_f64 | F64_to_f32 | S64_to_f32 | S64_to_f64

  let name = function
    | F32_to_f64 -> "fext.f32.f64"
    | F64_to_f32 -> "fround.f64.f32"
    | S64_to_f32 -> "scvt.i64.f32"
    | S64_to_f64 -> "scvt.i64.f64"
end

module Funary = struct
  type t = Neg | Sqrt | Trunc

  let name = function Neg -> "neg" | Sqrt -> "sqrt" | Trunc -> "trunc"
end

module Iarith = struct
  (* Modular two's-complement arithmetic at the operands' width. A shift count
     must be a constant below the width. *)
  type t = Add | And | Mul | Or | Shl | Shr_s | Shr_u | Sub | Xor

  let name = function
    | Add -> "add"
    | And -> "and"
    | Mul -> "mul"
    | Or -> "or"
    | Shl -> "shl"
    | Shr_s -> "ashr"
    | Shr_u -> "lshr"
    | Sub -> "sub"
    | Xor -> "xor"

  let is_shift = function
    | Shl | Shr_s | Shr_u -> true
    | Add | And | Mul | Or | Sub | Xor -> false
end

module Icmp = struct
  type t = Eq | Ne | Sle | Slt | Ule | Ult

  let name = function
    | Eq -> "eq"
    | Ne -> "ne"
    | Sle -> "sle"
    | Slt -> "slt"
    | Ule -> "ule"
    | Ult -> "ult"
end

module Idiv = struct
  (* Signed, truncating toward zero. Defined only for a nonzero divisor and not
     the minimum divided by -1: guards before it implement the source failures,
     and executing it outside that domain is a defect. *)
  type t = Div_s | Rem_s

  let name = function Div_s -> "sdiv" | Rem_s -> "srem"
end

module Iext = struct
  type t = Sext | Zext

  let name = function Sext -> "sext" | Zext -> "zext"
end

module Pbinary = struct
  type t = And | Or | Xor

  let name = function And -> "and" | Or -> "or" | Xor -> "xor"
end

module Callee = struct
  type t = Func of Mir_id.Func.t | Helper of Mir_id.Helper.t

  let pp fmt = function
    | Func f -> Mir_id.Func.pp fmt f
    | Helper h -> Mir_id.Helper.pp fmt h
end

module Access = struct
  (* A raw little-endian memory access of [width] bytes at [addr], whose
     address must be a multiple of [align]. *)
  type t = { width : Mir_width.t; addr : Mir_value.t; align : int64 }
end

module Vaccess = struct
  (* [lanes] elements, lane [k] a raw little-endian [elem] at [addr + k *
     stride] bytes — [stride] the element's size is contiguous, 0 a broadcast
     of one element — each at an address that is a multiple of [align]. Every
     lane is accessed: a lane outside its object is a defect, never hidden. *)
  type t = {
    elem : Mir_type.Elem.t;
    lanes : Mir_type.Lanes.t;
    addr : Mir_value.t;
    stride : int64;
    align : int64;
  }
end

type t =
  | Addr of Mir_id.View.t
  | Bitcast of Mir_type.t * Mir_value.t
  | Call of Callee.t * Mir_value.t list
  | Const of Mir_const.t
  | Copy of Mir_value.t
  | Event of Mir_event.t * int64
  | Fbinary of Fbinary.t * Mir_value.t * Mir_value.t
  | Fcmp of Fcmp.t * Mir_value.t * Mir_value.t
  | Fconvert of Fconvert.t * Mir_value.t
  | Ffma of Mir_value.t * Mir_value.t * Mir_value.t
      (** [a * b + c] with one rounding, always *)
  | Fto_sint of Mir_value.t
      (** f64 to i64, truncating; defined only on finite values in
          [-2^63, 2^63) *)
  | Funary of Funary.t * Mir_value.t
  | Iarith of Iarith.t * Mir_value.t * Mir_value.t
  | Icmp of Icmp.t * Mir_value.t * Mir_value.t
  | Idiv of Idiv.t * Mir_value.t * Mir_value.t
  | Iext of Iext.t * Mir_width.t * Mir_value.t
  | Itrunc of Mir_width.t * Mir_value.t
  | Load of Access.t
  | Narrow of Mir_width.t * Mir_value.t
      (** signed narrowing, defined only when the value fits the narrower width:
          the realization of a proof (an in-domain index), so no guard precedes
          it. The interpreter treats a value that does not fit as a compiler
          defect; a native realization simply truncates. *)
  | Pbinary of Pbinary.t * Mir_value.t * Mir_value.t
  | Pnot of Mir_value.t
  | Ptr_add of Mir_value.t * Mir_value.t
      (** a pointer plus a signed i64 byte offset: never integer arithmetic *)
  | Select of Mir_value.t * Mir_value.t * Mir_value.t
  | Store of Access.t * Mir_value.t
  | Undef of Mir_id.View.t
      (** every byte of the view undefined again: a fresh object's lifetime
          begins there. Ordered as a write; a native realization emits nothing.
      *)
  | Vconcat of Mir_value.t list
      (** the vectors' lanes one after another, in list order *)
  | Vextract of Mir_type.Lane.t * Mir_value.t
  | Vinsert of Mir_type.Lane.t * Mir_value.t * Mir_value.t
      (** the vector with one lane replaced by the element *)
  | Vload of Vaccess.t
  | Vslice of Mir_type.Lane.t * Mir_type.Lanes.t * Mir_value.t
      (** [count] consecutive lanes from [first] *)
  | Vsplat of Mir_type.Lanes.t * Mir_value.t
  | Vstore of Vaccess.t * Mir_value.t  (** lanes written in lane order *)

(* The float and predicate operations ([Copy], [Fbinary], [Fcmp], the
   precision conversions, [Ffma], [Funary], [Pbinary], [Pnot], [Select]) also
   apply lane by lane to vectors and masks of one lane count: lane [k] of the
   result is the scalar operation on lane [k] of each operand, and a [Fcmp]
   of vectors is a mask. *)

(* How an operation interacts with order. [Partial] is pure but defined only on
   a restricted domain: it is never speculated above its guards. *)
module Effect = struct
  type t = Call | Event | Partial | Pure | Read | Write

  let ordered = function
    | Call | Event | Read | Write -> true
    | Partial | Pure -> false
end

let effect_class = function
  | Call _ -> Effect.Call
  | Event _ -> Effect.Event
  | Fto_sint _ | Idiv _ | Narrow _ -> Effect.Partial
  | Load _ | Vload _ -> Effect.Read
  | Store _ | Undef _ | Vstore _ -> Effect.Write
  | Addr _ | Bitcast _ | Const _ | Copy _ | Fbinary _ | Fcmp _ | Fconvert _
  | Ffma _ | Funary _ | Iarith _ | Icmp _ | Iext _ | Itrunc _ | Pbinary _
  | Pnot _ | Ptr_add _ | Select _ | Vconcat _ | Vextract _ | Vinsert _
  | Vslice _ | Vsplat _ ->
      Effect.Pure

let operands = function
  | Addr _ | Const _ | Event _ | Undef _ -> []
  | Bitcast (_, a)
  | Copy a
  | Fconvert (_, a)
  | Fto_sint a
  | Funary (_, a)
  | Iext (_, _, a)
  | Itrunc (_, a)
  | Narrow (_, a)
  | Pnot a
  | Vextract (_, a)
  | Vslice (_, _, a)
  | Vsplat (_, a) ->
      [ a ]
  | Fbinary (_, a, b)
  | Fcmp (_, a, b)
  | Iarith (_, a, b)
  | Icmp (_, a, b)
  | Idiv (_, a, b)
  | Pbinary (_, a, b)
  | Ptr_add (a, b)
  | Vinsert (_, a, b) ->
      [ a; b ]
  | Call (_, args) | Vconcat args -> args
  | Ffma (a, b, c) | Select (a, b, c) -> [ a; b; c ]
  | Load { Access.addr; _ } | Vload { Vaccess.addr; _ } -> [ addr ]
  | Store ({ Access.addr; _ }, v) | Vstore ({ Vaccess.addr; _ }, v) ->
      [ addr; v ]

let map_operands f = function
  | (Addr _ | Const _ | Event _ | Undef _) as op -> op
  | Bitcast (ty, a) -> Bitcast (ty, f a)
  | Call (c, args) -> Call (c, List.map f args)
  | Copy a -> Copy (f a)
  | Fbinary (o, a, b) ->
      let a = f a in
      Fbinary (o, a, f b)
  | Fcmp (o, a, b) ->
      let a = f a in
      Fcmp (o, a, f b)
  | Fconvert (c, a) -> Fconvert (c, f a)
  | Ffma (a, b, c) ->
      let a = f a in
      let b = f b in
      Ffma (a, b, f c)
  | Fto_sint a -> Fto_sint (f a)
  | Funary (o, a) -> Funary (o, f a)
  | Iarith (o, a, b) ->
      let a = f a in
      Iarith (o, a, f b)
  | Icmp (o, a, b) ->
      let a = f a in
      Icmp (o, a, f b)
  | Idiv (o, a, b) ->
      let a = f a in
      Idiv (o, a, f b)
  | Iext (k, w, a) -> Iext (k, w, f a)
  | Itrunc (w, a) -> Itrunc (w, f a)
  | Load acc -> Load { acc with Access.addr = f acc.Access.addr }
  | Narrow (w, a) -> Narrow (w, f a)
  | Pbinary (o, a, b) ->
      let a = f a in
      Pbinary (o, a, f b)
  | Pnot a -> Pnot (f a)
  | Ptr_add (a, b) ->
      let a = f a in
      Ptr_add (a, f b)
  | Select (p, a, b) ->
      let p = f p in
      let a = f a in
      Select (p, a, f b)
  | Store (acc, v) ->
      let addr = f acc.Access.addr in
      Store ({ acc with Access.addr }, f v)
  | Vconcat vs -> Vconcat (List.map f vs)
  | Vextract (l, a) -> Vextract (l, f a)
  | Vinsert (l, a, b) ->
      let a = f a in
      Vinsert (l, a, f b)
  | Vload acc -> Vload { acc with Vaccess.addr = f acc.Vaccess.addr }
  | Vslice (l, n, a) -> Vslice (l, n, f a)
  | Vsplat (n, a) -> Vsplat (n, f a)
  | Vstore (acc, v) ->
      let addr = f acc.Vaccess.addr in
      Vstore ({ acc with Vaccess.addr }, f v)

let name = function
  | Addr _ -> "addr"
  | Bitcast _ -> "bitcast"
  | Call _ -> "call"
  | Const _ -> "const"
  | Copy _ -> "copy"
  | Event _ -> "event"
  | Fbinary (o, _, _) -> "f" ^ Fbinary.name o
  | Fcmp (o, _, _) -> "fcmp." ^ Fcmp.name o
  | Fconvert (c, _) -> Fconvert.name c
  | Ffma _ -> "ffma"
  | Fto_sint _ -> "fcvt.f64.i64"
  | Funary (o, _) -> "f" ^ Funary.name o
  | Iarith (o, _, _) -> Iarith.name o
  | Icmp (o, _, _) -> "icmp." ^ Icmp.name o
  | Idiv (o, _, _) -> Idiv.name o
  | Iext (k, w, _) -> Iext.name k ^ "." ^ Mir_width.name w
  | Itrunc (w, _) -> "trunc." ^ Mir_width.name w
  | Load { Access.width; _ } -> "load." ^ Mir_width.name width
  | Narrow (w, _) -> "narrow." ^ Mir_width.name w
  | Pbinary (o, _, _) -> "p" ^ Pbinary.name o
  | Pnot _ -> "pnot"
  | Ptr_add _ -> "ptr.add"
  | Select _ -> "select"
  | Store ({ Access.width; _ }, _) -> "store." ^ Mir_width.name width
  | Undef _ -> "undef"
  | Vconcat _ -> "vconcat"
  | Vextract _ -> "vextract"
  | Vinsert _ -> "vinsert"
  | Vload { Vaccess.elem; _ } -> "vload." ^ Mir_type.Elem.name elem
  | Vslice _ -> "vslice"
  | Vsplat _ -> "vsplat"
  | Vstore ({ Vaccess.elem; _ }, _) -> "vstore." ^ Mir_type.Elem.name elem
