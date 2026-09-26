(* See dim.mli. Inside this module the manifest [ 'role t = int ] is visible, so
   ints may be used directly where a role type is expected; the signature hides
   that from callers. *)

type +'role t = int
type extent

(* [index]/[delta] are manifest aliases of the markers in [Role], which [Expr]
   re-exports, not fresh local tags. They have to be: [Symbolic]'s associated
   type becomes ['role Expr.Index.t], so the two role vocabularies must be the
   SAME types for the [SEMANTICS] instantiation to typecheck at all — a runtime
   conversion cannot bridge a phantom parameter. See role.mli.

   [extent] is here so an [Expr] intrinsic can take it; [count]/[offset]
   are storage-layout roles the expression language has no notion of. *)
type index = Role.Position.t
type delta = Role.Delta.t
type count
type offset
type fence

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
  if n < 1 then Err.fail (`Non_positive_extent n) else Err.return (extent n)

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
let fence n : fence t = if n < 0 then invalid_arg "Dim.fence: negative" else n
let fence_of_index (i : index t) : fence t = i
let fence_of_extent (e : extent t) : fence t = e
let fence_le (a : fence t) (b : fence t) = a <= b

let span (start : fence t) (stop : fence t) : extent t option =
  if start < stop then Some (stop - start) else None

module Product_witness = struct
  type nonrec t = { prefix : int64; factor : extent t; limit : int64 }

  let pp ppf { prefix; factor; limit } =
    Format.fprintf ppf "%Ld * %d does not fit below %Ld" prefix factor limit
end

let product_bounded ~limit (extents : extent t list) =
  let ceiling = Int64.pred limit in
  if Int64.compare ceiling (Int64.of_int max_int) > 0 then
    invalid_arg "Dim.product_bounded: limit exceeds max_int + 1";
  if Int64.compare ceiling 1L < 0 then
    invalid_arg "Dim.product_bounded: limit must exceed 1";
  let rec go prefix = function
    | [] -> Err.return (Int64.to_int prefix)
    | factor :: rest ->
        let f = Int64.of_int factor in
        if Int64.compare prefix (Int64.div ceiling f) > 0 then
          Err.fail
            (`Product_over_limit Product_witness.{ prefix; factor; limit })
        else go (Int64.mul prefix f) rest
  in
  go 1L extents

let advance ~(start : fence t) (i : index t) : index t = start + i
let fence_after (start : fence t) (e : extent t) : fence t = start + e

let local_in ~(start : fence t) ~(extent : extent t) (i : index t) :
    index t option =
  if i >= start && i - start < extent then Some (i - start) else None

let block_start (i : index t) (e : extent t) : fence t = i * e
let wrap (i : index t) (e : extent t) : index t = i mod e

let div_exact (a : extent t) ~(by : extent t) : extent t option =
  if a mod by = 0 then Some (a / by) else None

let divides ~(by : extent t) (a : extent t) = a mod by = 0
let unlin (o : offset t) (e : extent t) : offset t * index t = (o / e, o mod e)
let to_int64 (x : 'role t) : int64 = Int64.of_int x

module Delta = struct
  let add (a : delta t) (b : delta t) : delta t = a + b
  let neg (a : delta t) : delta t = -a
  let min (a : delta t) (b : delta t) : delta t = Stdlib.min a b
  let max (a : delta t) (b : delta t) : delta t = Stdlib.max a b
  let scale k (d : delta t) : delta t = k * d

  (* Floor toward negative infinity for a negative numerator, where [/]
     truncates toward zero. *)
  let floor_div_pos (n : delta t) ~(by : extent t) : delta t =
    if n >= 0 then n / by else -((-n + by - 1) / by)

  let ceil_div_pos (n : delta t) ~(by : extent t) : delta t =
    neg (floor_div_pos (neg n) ~by)

  let of_extent (e : extent t) : delta t = e
  let clamp_low (x : delta t) : index t = if x < 0 then 0 else x
  let assume_index (x : delta t) : index t = index x
end
