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

(* checked constructors for the per-axis roles *)
val extent : int -> extent t (* raises [Invalid_argument] if < 0 *)
val index : int -> index t (* raises [Invalid_argument] if < 0 *)

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

(* role-preserving: max/min of two same-role values keeps the role (a broadcast
   picks the larger of two extents) *)
val max : 'role t -> 'role t -> 'role t
val min : 'role t -> 'role t -> 'role t

val to_int :
  'role t -> int (* also available as the free coercion [(x :> int)] *)

val pp : Format.formatter -> 'role t -> unit
