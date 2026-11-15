(* A program point of a function's linear order: blocks in reverse postorder,
   every instruction at two points (its operands read at the even one, its
   results written at the odd one after it) and two more for the block's
   boundary. Points are only compared, ordered and stepped here, so nothing
   else can pass an allocator's register, slot or block number for one.

   A point is bounded by twice the function's instruction count plus the
   block count, far below the 32-bit [int] of js_of_ocaml; [never] is the
   largest [int], past every point. *)

module Id =
  Core.Tagged_int.Make
    (struct
      let prefix = "@"
    end)
    ()

type t = Id.t

let zero = Id.of_int 0
let never = Id.of_int max_int
let equal = Id.equal
let compare = Id.compare
let min a b = if compare a b <= 0 then a else b
let max a b = if compare a b >= 0 then a else b
let pp = Id.pp

(* Comparisons on points, to open where an expression holds nothing else. *)
module Infix = struct
  let ( = ) = equal
  let ( <> ) a b = not (equal a b)
  let ( < ) a b = compare a b < 0
  let ( <= ) a b = compare a b <= 0
  let ( > ) a b = compare a b > 0
  let ( >= ) a b = compare a b >= 0
end

(* The point just after [p]: where the results of an instruction read at [p]
   are written. *)
let succ = Id.succ

(* The point just before [p]. *)
let pred p = Id.of_int ((p : t :> int) - 1)

(* Two points on: the next instruction's read, from one's read. *)
let next p = Id.of_int ((p : t :> int) + 2)

(* Whether [p] is a write point (an odd one). *)
let is_write p = (p : t :> int) land 1 = 1

(* The read point at or before [p]: the point a split for a use at [p]
   happens at. *)
let read_at_or_before p = if is_write p then pred p else p

(* The next read point after the instruction [p] belongs to. *)
let after_instruction p = Id.of_int (((p : t :> int) lor 1) + 1)

(* The point past a block of [body] that starts at [from]: two per
   instruction and two for the boundary. *)
let block_end from body =
  Id.of_int ((from : t :> int) + (2 * List.length body) + 2)

(* Each instruction of a block of [body] starting at [from], with its read
   point: the first at [from + 2]. *)
let reads from body =
  List.mapi (fun j i -> (Id.of_int ((from : t :> int) + 2 + (2 * j)), i)) body

(* How far [to_] is after [from], to weight and compare, never to index. *)
let distance ~from ~to_ = float_of_int ((to_ : t :> int) - (from : t :> int))

(* One cell per point of a function, up to and including [upto]. *)
module Table = struct
  type 'a t = { cells : 'a array; absent : 'a }

  let make ~upto absent =
    { cells = Array.make ((upto : Id.t :> int) + 1) absent; absent }

  let fill t ~from ~to_ v =
    Array.fill t.cells
      (from : Id.t :> int)
      ((to_ : Id.t :> int) - (from : Id.t :> int))
      v

  (* The cell at [p], or the table's default outside it. *)
  let get t (p : Id.t) =
    if (p :> int) >= 0 && (p :> int) < Array.length t.cells then
      t.cells.((p :> int))
    else t.absent
end
