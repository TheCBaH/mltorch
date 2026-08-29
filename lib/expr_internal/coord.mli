(* The language's six-field product over [Axis], parameterised by its component
   type: [int Coord.t] is a concrete output coordinate, [_ Index.t Coord.t] is
   the symbolic coordinate a [Load] carries.

   The record lives in its own module and is named [t], per the repository
   convention -- that is what keeps [Coord.h] from colliding with the [h] of
   every other six-field record. Conversion to and from [lib/native]'s [Vec6] is
   the adapter's business, not this library's. *)

type 'a t = { n : 'a; t : 'a; d : 'a; h : 'a; w : 'a; c : 'a }

val make : n:'a -> t:'a -> d:'a -> h:'a -> w:'a -> c:'a -> 'a t
val of_fn : (Axis.t -> 'a) -> 'a t
val get : 'a t -> Axis.t -> 'a
val set : 'a t -> Axis.t -> 'a -> 'a t
val map : ('a -> 'b) -> 'a t -> 'b t
val mapi : (Axis.t -> 'a -> 'b) -> 'a t -> 'b t
val fold : ('acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc
val foldi : (Axis.t -> 'acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc
val for_all : ('a -> bool) -> 'a t -> bool
val to_list : 'a t -> 'a list

val pp : (Format.formatter -> 'a -> unit) -> Format.formatter -> 'a t -> unit
(** Comma-separated in [Axis.all] order and deliberately unbracketed, so a
    [Load] prints as [t0[N,T,D,H,W,C]] exactly as it does today. Uses
    [Fmt.any ","], not [Fmt.comma], which is breakable. *)
