(* Which output of a multi-output node: the position in its [outputs] list.
   Distinct from a tensor id (which names the edge) and from an emitter ordinal
   in a Region group (which numbers the group's emitters). *)
include Core.Tagged_int.S

val zero : t
(** The first output: a value, for the ops whose second output is an index. *)

val one : t

val indexed : 'a list -> (t * 'a) list
(** [xs] paired with their ordinals, in order. *)
