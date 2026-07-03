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
  if n < 1 then invalid_arg "Dim.extent: must be >= 1" else n

let index n : index t = if n < 0 then invalid_arg "Dim.index: negative" else n

(* Recoverable error set owned by this module — for untrusted sizes (e.g. a raw
   model dim). The plain [extent]/[index] above stay total: they assert a
   trusted precondition (a negative there is a bug, not expected input). *)
type error = [ `Non_positive_extent of int ]

let pp_error ppf : error -> unit = function
  | `Non_positive_extent n -> Format.fprintf ppf "extent must be >= 1, got %d" n

let extent_checked n =
  if n < 1 then Core.fail (`Non_positive_extent n) else Core.return (extent n)

let delta n : delta t = n
let one_count : count t = 1
let zero_offset : offset t = 0
let ( *@ ) (acc : count t) (e : extent t) : count t = acc * e
let lin (acc : offset t) (e : extent t) (i : index t) : offset t = (acc * e) + i
let to_delta (i : index t) : delta t = i

let index_of ~(extent : extent t) (d : delta t) : index t option =
  if d >= 0 && d < extent then Some d else None

(* Role-preserving increment, no validation: for a loop counter already known
   to stay in range by construction (bounded above by a separate check, e.g.
   [Direct.sum]'s accumulator — see .ai/pt2_inference_perf.md), re-deriving
   the role via [index]/[extent]'s checked constructors on every step would
   re-pay a check the loop's own structure already guarantees passes. *)
let succ (x : 'role t) : 'role t = x + 1

(* role-preserving: same-role operands keep the role. [equal] compares two sizes;
   [one] is the unit extent a broadcast axis is tested against. *)
let equal (a : 'role t) (b : 'role t) : bool = Int.equal a b
let one : 'role t = 1
let to_int (x : 'role t) : int = x
let pp fmt (x : 'role t) = Format.pp_print_int fmt x

let extent_jsont : extent t Jsont.t =
  Jsont.map ~kind:"extent"
    ~dec:(fun n ->
      if n < 1 then
        Jsont.Error.msgf Jsont.Meta.none "extent: must be >= 1, got %d" n
      else extent n)
    ~enc:(fun x -> (x :> int))
    Jsont.int
