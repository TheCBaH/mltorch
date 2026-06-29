(* Scalar dimensional types — `int` is too loose for the engine's sizes and
   positions. Four non-negative roles split into per-axis (extent/index) and
   flattened (count/offset) pairs, plus one signed role (delta). They share a
   `private int` representation tagged by a phantom role, so:
     - they never unify with each other (extent <> offset, count <> offset, …);
     - reading back to a raw int (for Bigarray / printing) is a free `:>` coercion;
     - construction is guarded (non-negative) — and count/offset are *derived*
       only, never built from a raw int by callers.
   See .ai/native_tensor_design.md §1a. *)

type +'role t = private int

(* phantom role tags (uninhabited) *)
type extent
type index
type count
type offset
type delta

(* checked constructors for the per-axis roles. An [extent] is a size, valid from
   1 (the engine has no empty tensors — lower-rank tensors embed with size-1 axes,
   never size-0); an [index] is a position, valid from 0. *)
val extent : int -> extent t (* raises [Invalid_argument] if < 1 *)
val index : int -> index t (* raises [Invalid_argument] if < 0 *)

(* Recoverable error set owned by this module, plus its printer — composes into
   a caller's row via [#Dim.error]. [extent]/[index] above assert a trusted
   precondition; [extent_checked] is the validated form for an untrusted size. *)
type error = [ `Non_positive_extent of int ]

val pp_error : Format.formatter -> error -> unit

(* Open row ([>]) so it unifies upward into a caller's wider union (see
   [Aten_shape.of_aten]); the closed [error] above is for [pp_error]/[#Dim.error]
   and for a boundary that pins its final set. *)
val extent_checked :
  int -> (extent t, [> `Non_positive_extent of int ]) Core.result

(* the one signed role: index differences and stencil offsets, which may be
   negative before being guarded back into an [index] *)
val delta : int -> delta t

(* count/offset are derived, not constructed from a raw int *)
val one_count : count t
val zero_offset : offset t
val ( *@ ) : count t -> extent t -> count t (* numel fold step *)

val lin :
  offset t -> extent t -> index t -> offset t (* o*ext + i, the Horner step *)

(* the bridge that confronts out-of-bounds: a signed [delta] becomes an in-range
   [index] only if it lands inside [extent] *)
val to_delta : index t -> delta t
val index_of : extent:extent t -> delta t -> index t option

(* role-preserving: same-role operands keep the role. [equal] compares two sizes;
   [one] is the unit extent a broadcast axis is tested against. *)
val equal : 'role t -> 'role t -> bool
val one : 'role t

val to_int :
  'role t -> int (* also available as the free coercion [(x :> int)] *)

val pp : Format.formatter -> 'role t -> unit
