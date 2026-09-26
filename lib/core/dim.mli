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

(* Phantom role tags (uninhabited). [index]/[delta] are manifest aliases of
   [Role]'s markers, which [Expr] re-exports: [Symbolic]'s associated type is
   ['role Expr.Index.t], and a phantom parameter cannot be bridged by a
   conversion, so the two vocabularies must be the same types. [extent] is here
   too, so an [Expr] intrinsic can take a kernel or an input extent as the very
   type [native] holds. [count]/[offset] are storage-layout roles. See
   role.mli. *)
type extent
type index = Role.Position.t
type count
type offset
type delta = Role.Delta.t

(* A position with [0 <= f <= extent]: one past the last index is legal. A
   slice's resolved [start] and [stop] are fences (stop may equal the extent),
   which is why neither [index] nor [extent] fits them. *)
type fence

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
val extent_checked : int -> (extent t, [> `Non_positive_extent of int ]) Err.t

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

(* Role-preserving increment, no validation — see the .ml for why this
   exists alongside the checked constructors. *)
val succ : 'role t -> 'role t

(* role-preserving: same-role operands keep the role. [equal] compares two sizes;
   [one] is the unit extent a broadcast axis is tested against. *)
val equal : 'role t -> 'role t -> bool
val one : 'role t

val to_int :
  'role t -> int (* also available as the free coercion [(x :> int)] *)

val pp : Format.formatter -> 'role t -> unit

(* --- operations that keep a computation inside its domain ---

   Unwrapping to [int], computing and re-wrapping states nothing about what the
   result means and forfeits every check the type could make, so the arithmetic
   the engine actually does on these roles lives here. *)

(* [0 <= f]; the upper bound is the caller's extent, not known here. *)
val fence : int -> fence t (* raises [Invalid_argument] if < 0 *)
val fence_of_index : index t -> fence t
val fence_of_extent : extent t -> fence t

(* [a <= b]: the order on fences, so a range check needs no unwrapping. *)
val fence_le : fence t -> fence t -> bool

(* [stop - start] as an extent, [None] unless [start < stop]: the engine has no
   empty tensors, so an empty span is not representable. *)
val span : fence t -> fence t -> extent t option

(* Where a product of extents stopped fitting. [prefix] is the product of the
   factors accepted so far, [factor] the one that would have crossed [limit];
   neither is a wrapped value. *)
module Product_witness : sig
  type nonrec t = { prefix : int64; factor : extent t; limit : int64 }

  val pp : Format.formatter -> t -> unit
end

(* The product of [extents], exclusive of [limit]: the result is [< limit]. Each
   factor is divided into the ceiling BEFORE multiplying, so an intermediate
   never wraps — js_of_ocaml's [int] is 32 bits, and a check on a wrapped
   product is not a bound. The witness is the un-multiplied pair, never the
   wrapped product. Same convention as [Vec6.numel_bounded]. The empty product
   is [one].

   Raises [Invalid_argument] if [limit - 1] exceeds [max_int], since the result
   is an [int]: the caller names a ceiling this backend can represent. *)
val product_bounded :
  limit:int64 ->
  extent t list ->
  (extent t, [> `Product_over_limit of Product_witness.t ]) Err.t

(* Composing a window with the axis it sits in. [advance ~start i] is the
   position, in the whole axis, of the window-local position [i] of a window
   that begins at [start]. [fence_after start e] is where that window ends.
   [local_in ~start ~extent i] is the inverse: the window-local position of
   [i], or [None] when [i] lies before or after the window. *)
val advance : start:fence t -> index t -> index t
val fence_after : fence t -> extent t -> fence t
val local_in : start:fence t -> extent:extent t -> index t -> index t option

(* Where block [i] of equal blocks of extent [e] begins: [i * e]. The caller
   keeps [i * e] inside the axis it is splitting (the blocks tile an extent it
   already divided), so this does not bound the product. *)
val block_start : index t -> extent t -> fence t

(* [i] reduced modulo [e]: a position tiled onto a shorter axis. *)
val wrap : index t -> extent t -> index t

(* [a / by] when [by] divides [a] exactly. *)
val div_exact : extent t -> by:extent t -> extent t option
val divides : by:extent t -> extent t -> bool

(* The dual of [lin]: [unlin (lin o e i) e = (o, i)]. *)
val unlin : offset t -> extent t -> offset t * index t

(* The one named exit for arithmetic that is deliberately done in [Int64]
   (resize, pool). The other exit is the free coercion [(x :> int)], for storage
   indexing and wire encoders. *)
val to_int64 : 'role t -> int64

(* The [Semantics] carrier's arithmetic. Deltas are signed and bounded by the
   engine's extents; like the engine's [int] arithmetic these do not check for
   overflow — a delta near the backend's [int] range is a caller bug. *)
module Delta : sig
  val add : delta t -> delta t -> delta t
  val neg : delta t -> delta t
  val min : delta t -> delta t -> delta t
  val max : delta t -> delta t -> delta t

  (* [scale k d] is [k * d] for a dimensionless scalar [k], which may be negative — the
     expression language's [Scale of int * _] literal. A typed factor (a stride,
     a group count) goes through [Dim_arith]. *)
  val scale : int -> delta t -> delta t

  (* Floor division by an extent: [floor_div_pos (-1) ~by:2 = -1]. *)
  val floor_div_pos : delta t -> by:extent t -> delta t
  val ceil_div_pos : delta t -> by:extent t -> delta t
  val of_extent : extent t -> delta t

  (* The two ways back from a signed delta to a position. [clamp_low] is sound
     for every input (negative reads as 0); [assume_index] is the one unchecked
     claim that the delta is already in range, and raises [Invalid_argument] if
     it is negative. *)
  val clamp_low : delta t -> index t
  val assume_index : delta t -> index t
end
