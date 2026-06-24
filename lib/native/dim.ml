(* See dim.mli. Inside this module the manifest [ 'role t = int ] is visible, so
   ints may be used directly where a role type is expected; the signature hides
   that from callers. *)

type +'role t = int
type extent
type index
type count
type offset
type delta

let extent n : extent t =
  if n < 0 then invalid_arg "Dim.extent: negative" else n

let index n : index t = if n < 0 then invalid_arg "Dim.index: negative" else n
let delta n : delta t = n
let one_count : count t = 1
let zero_offset : offset t = 0
let ( *@ ) (acc : count t) (e : extent t) : count t = acc * e
let lin (acc : offset t) (e : extent t) (i : index t) : offset t = (acc * e) + i
let to_delta (i : index t) : delta t = i

let index_of ~(extent : extent t) (d : delta t) : index t option =
  if d >= 0 && d < extent then Some d else None

(* role-preserving: the larger/smaller of two same-role values is still that
   role (e.g. a broadcast picks the larger of two extents) *)
let max (a : 'role t) (b : 'role t) : 'role t = Stdlib.max a b
let min (a : 'role t) (b : 'role t) : 'role t = Stdlib.min a b
let to_int (x : 'role t) : int = x
let pp fmt (x : 'role t) = Format.pp_print_int fmt x
