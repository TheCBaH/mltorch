(** An index as an exact linear combination of opaque atoms plus a constant.
    Every coefficient and the constant are in [Loop_range.domain]; a combination
    that leaves it is [None]. The form is the index's value only: a caller that
    rebuilds an index from one proves the overflow of both sides itself. *)

type t = { terms : (Loop_index.t * int) list; const : int }
(** [terms] in order of first appearance, no zero coefficient, no atom twice. *)

val const : int -> t
val atom : Loop_index.t -> t
val add : t -> t -> t option
val sub : t -> t -> t option
val scale : int -> t -> t option

val of_index : Loop_index.t -> t option
(** [Add], [Scale] and [Const] unfolded; every other node is an atom. *)

val to_index : t -> Loop_index.t
val coefficient : t -> Loop_index.t -> int

val range : ?depth:int -> Loop_range.Env.t -> t -> Loop_range.t
(** The interval of the form's value, intersected with the relational fact that
    [m * (x - d * floor (x / d))] is in [m * [0, d - 1]]. *)
