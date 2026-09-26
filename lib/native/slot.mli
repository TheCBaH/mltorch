(* Numbers of a Region's flat scratch array. A slot offset indexes that array,
   not a tensor, so none of these is [Dim.offset] or [Dim.count]: conflating a
   slot with a tensor position is exactly the mistake the types are for.

   [extent] is how many slots one vector (or one trace row) holds, [count] how
   many slots a range holds in all, and [offset] where a range starts. Like
   [Dim], each is a [private int], so leaving the domain for an [Array] index is
   the free coercion [(x :> int)].

   The arithmetic here is plain [int], not [Int64]-checked: [Region_program.check]
   bounds the SUM of every local's slot count against [max_size] on [Int64]
   before a [Region_program.t] can exist, so by the time a value reaches this
   module the total is already proven to fit, on the 32-bit backends too. *)

type +'role t = private int
type extent
type count
type offset

val extent : int -> extent t (* raises [Invalid_argument] if negative *)

val of_dim : Dim.extent Dim.t -> extent t
(** A vector local that spans one tensor axis holds that axis's extent in slots.
    The one place a tensor extent becomes a slot extent. *)

val one : count t
val zero : offset t
val count_of_extent : extent t -> count t

val trace_count : steps:extent t -> width:extent t -> count t
(** A scan trace stores its initial row and one row per step, [width] lanes
    each: [(steps + 1) * width] slots. [Scan_limits]' hard ceilings bound both
    factors well below the 32-bit range before an [Expr.Scan.t] exists. *)

val advance : offset t -> count t -> offset t
(** The offset just past a range that starts at the first. *)

val total : offset t -> count t
(** The count of every slot laid out so far: the offset past the last range. *)

module Range : sig
  type nonrec t = { offset : offset t; count : count t }
end

val at : Range.t -> int -> int option
(** The array index of the [pos]-th slot of the range, [None] outside it. [pos]
    is the position an expression asked for, which may be anything. *)

val trace_at : offset t -> width:extent t -> row:int -> lane:int -> int
(** The array index of a trace's cell, row-major: [offset + row * width + lane].
    The caller has bounds-checked [row] and [lane] against the trace's own steps
    and width. *)

val equal : 'role t -> 'role t -> bool
val pp : Format.formatter -> 'role t -> unit
