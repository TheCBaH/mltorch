(* A runtime datum: exact bits for every scalar (canonical integers, binary32
   in the low 32, binary64, predicates 0/1), a synthetic pointer, or the order
   state, which carries nothing. *)
type t =
  | Bits of int64
  | Flags of { bits : int64; defined : int64 }
      (** condition state: only the [defined] bits have a value *)
  | Order
  | Ptr of Mir_memory.Pointer.t

let of_const (c : Machine_ir.Mir_const.t) = Bits c.Machine_ir.Mir_const.bits
let f64 x = Bits (Int64.bits_of_float x)
let i64 x = Bits x
let i32 x = Bits (Machine_ir.Mir_width.normalize Machine_ir.Mir_width.W32 x)

let pp fmt = function
  | Bits b -> Fmt.pf fmt "0x%Lx" b
  | Flags { bits; defined } -> Fmt.pf fmt "flags 0x%Lx/0x%Lx" bits defined
  | Order -> Fmt.string fmt "order"
  | Ptr p -> Fmt.pf fmt "ptr+%Ld" p.Mir_memory.Pointer.offset
