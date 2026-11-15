(* See slot.mli. *)

type +'role t = int
type extent
type count
type offset
type position

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

let at { Range.offset; count } pos : position t option =
  if pos >= 0 && pos < count then Some (offset + pos) else None

let trace_at (offset : offset t) ~(steps : extent t) ~(width : extent t) ~row
    ~lane : (position t, [> `Lane | `Row ]) result =
  if row < 0 || row > steps then Error `Row
  else if lane < 0 || lane >= width then Error `Lane
  else Ok (offset + (row * width) + lane)

let equal (a : 'role t) (b : 'role t) = Int.equal a b
let pp ppf (x : 'role t) = Format.pp_print_int ppf x
let scan_state (w : extent t) = 2 * (w :> int)
let trace_rows ~(steps : extent t) : extent t = steps + 1
