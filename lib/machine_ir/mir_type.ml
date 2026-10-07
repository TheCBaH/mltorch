(* The value types of Machine IR. A value type keeps the interpretation a
   verifier needs: an integer has a width but no signedness (operations choose
   it), a float has its working precision, a pointer is not an integer, and a
   predicate is a defined truth value whose stored form is an explicit
   operation's choice. [Order] is the compile-time sequencing state: it has no
   register, stack slot or runtime representation. *)

(* A vector's lane count: its own domain, never an extent or a register count.
   A logical vector may span several physical registers. *)
module Lanes =
  Core.Tagged_int.Make
    (struct
      let prefix = "x"
    end)
    ()

(* One lane's position, a different domain from the count. *)
module Lane =
  Core.Tagged_int.Make
    (struct
      let prefix = "lane"
    end)
    ()

(* The widest logical vector, as in the SSA IR. *)
let max_lanes = 64

(* A vector element: a float precision or an integer width. *)
module Elem = struct
  type t = F32 | F64 | Int of Mir_width.t

  let equal a b =
    match (a, b) with
    | F32, F32 | F64, F64 -> true
    | Int a, Int b -> Mir_width.equal a b
    | (F32 | F64 | Int _), _ -> false

  let bytes = function F32 -> 4L | F64 -> 8L | Int w -> Mir_width.bytes w
  let name = function F32 -> "f32" | F64 -> "f64" | Int w -> Mir_width.name w
end

type t =
  | F32
  | F64
  | Flags
      (** a selected program's condition state: defined bits per producing form,
          consumed in the block that defines it, never a block parameter, stack
          value or edge argument *)
  | Int of Mir_width.t
  | Mask of Lanes.t
  | Order
  | Pred
  | Ptr
  | Vec of Elem.t * Lanes.t

let i8 = Int Mir_width.W8
let i16 = Int Mir_width.W16
let i32 = Int Mir_width.W32
let i64 = Int Mir_width.W64

let equal a b =
  match (a, b) with
  | F32, F32 | F64, F64 | Flags, Flags | Order, Order | Pred, Pred | Ptr, Ptr ->
      true
  | Int a, Int b -> Mir_width.equal a b
  | Mask a, Mask b -> Lanes.equal a b
  | Vec (e, a), Vec (f, b) -> Elem.equal e f && Lanes.equal a b
  | (F32 | F64 | Flags | Int _ | Mask _ | Order | Pred | Ptr | Vec _), _ ->
      false

(* The scalar type of a vector element, and the element a scalar type is. *)
let of_elem = function Elem.F32 -> F32 | Elem.F64 -> F64 | Elem.Int w -> Int w

let elem = function
  | F32 -> Some Elem.F32
  | F64 -> Some Elem.F64
  | Int w -> Some (Elem.Int w)
  | Flags | Mask _ | Order | Pred | Ptr | Vec _ -> None

let lanes_in_range l =
  let n = Lanes.to_int l in
  n >= 1 && n <= max_lanes

let is_float = function F32 | F64 -> true | _ -> false
let is_int = function Int _ -> true | _ -> false

(* Whether a value of this type occupies machine state: everything but the
   order token. *)
let has_storage = function Order -> false | _ -> true

(* The bytes a value occupies in memory or a spill slot, for the scalar and
   pointer types; a predicate's stored form is an explicit choice, a vector's
   is its lanes. *)
let bytes = function
  | F32 -> Some 4L
  | F64 -> Some 8L
  | Int w -> Some (Mir_width.bytes w)
  | Ptr -> Some 8L
  | Vec (e, l) ->
      Some (Int64.mul (Elem.bytes e) (Int64.of_int (Lanes.to_int l)))
  | Flags | Mask _ | Order | Pred -> None

let pp fmt = function
  | F32 -> Fmt.string fmt "f32"
  | F64 -> Fmt.string fmt "f64"
  | Int w -> Mir_width.pp fmt w
  | Mask l -> Fmt.pf fmt "mask<%a>" Lanes.pp l
  | Flags -> Fmt.string fmt "flags"
  | Order -> Fmt.string fmt "order"
  | Pred -> Fmt.string fmt "pred"
  | Ptr -> Fmt.string fmt "ptr64"
  | Vec (e, l) -> Fmt.pf fmt "vec<%a,%s>" Lanes.pp l (Elem.name e)
