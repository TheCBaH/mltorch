(* An opaque symbol naming an external scalar-addressable source. The language
   knows nothing about tensor ownership, storage format, quantization or graph
   identity: [Eval.Env.load] asks the caller for the value at a coordinate, and
   that callback is the whole boundary. It is what lets this library stay
   independent of [native]'s [Tensor_sig.t].

   [to_int] is what keeps the [Tensor_id.t] <-> [Source.t] map a stateless
   bijection rather than a side table, so no semantics depends on allocation
   order beyond symbol equality. *)

type t

val create : int -> t
val to_int : t -> int
val compare : t -> t -> int
val equal : t -> t -> bool
val hash : t -> int

val pp : Format.formatter -> t -> unit
(** Renders [t<n>], matching [lib/native]'s [Tensor_id.pp], so a printed [Load]
    is byte-identical across the migration. *)

module Map : Map.S with type key = t
module Set : Set.S with type elt = t
