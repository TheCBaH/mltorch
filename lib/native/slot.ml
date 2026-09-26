(* See slot.mli. *)

type +'role t = int
type extent
type count
type offset

let extent n : extent t =
  if n < 0 then invalid_arg "Slot.extent: negative" else n

let of_dim (e : Dim.extent Dim.t) : extent t = (e :> int)
let one : count t = 1
let zero : offset t = 0
let count_of_extent (e : extent t) : count t = e

let trace_count ~(steps : extent t) ~(width : extent t) : count t =
  (steps + 1) * width

let advance (o : offset t) (c : count t) : offset t = o + c
let total (o : offset t) : count t = o

module Range = struct
  type nonrec t = { offset : offset t; count : count t }
end

let at { Range.offset; count } pos =
  if pos >= 0 && pos < count then Some (offset + pos) else None

let trace_at (offset : offset t) ~(width : extent t) ~row ~lane =
  offset + (row * width) + lane

let equal (a : 'role t) (b : 'role t) = Int.equal a b
let pp ppf (x : 'role t) = Format.pp_print_int ppf x
