(* The number of dimensions of an ATen tensor. Not an extent, not a dim number
   and not a count of anything else: a rank names how many of the frame's axes
   a tensor's own shape occupies (right-aligned; see [Aten_shape]), and a dim
   number is judged against it. Non-negative; the frame's upper bound of 6 is
   [Aten_shape]'s to enforce, since an untrusted shape can carry more. *)

type t = private int

val of_int : int -> t (* raises [Invalid_argument] if < 0 *)
val of_list : 'a list -> t
val of_array : 'a array -> t
val to_int : t -> int
val equal : t -> t -> bool
val compare : t -> t -> int
val pp : Format.formatter -> t -> unit

val jsont : t Jsont.t
(** The wire form is the bare integer; decoding rejects a negative one. *)
