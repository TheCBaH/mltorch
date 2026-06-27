(* A 6-component vector over the axes N T D H W C, tagged by a phantom [role] so
   that a [shape] (extents) and a [coord] (indices) share one representation yet
   never unify, and [get] hands back the right scalar dimensional type ([extent]
   for a shape, [index] for a coord). The only operations bridging the two roles
   ([numel], [offset], [in_bounds], [iter]) take both, so their arguments can't be
   swapped. See .ai/native_tensor_design.md §1b. *)

type ('role, 'comp) t
type shape_role
type coord_role
type shape = (shape_role, Dim.extent Dim.t) t (* extents, each >= 0 *)
type coord = (coord_role, Dim.index Dim.t) t (* indices, 0 <= a_i < extent_i *)

val shape : n:int -> t:int -> d:int -> h:int -> w:int -> c:int -> shape
val coord : n:int -> t:int -> d:int -> h:int -> w:int -> c:int -> coord
val origin : coord (* all zero *)
val get : ('role, 'comp) t -> Axis.t -> 'comp
val set : ('role, 'comp) t -> Axis.t -> 'comp -> ('role, 'comp) t

(* the role-bridging operations *)
val numel : shape -> Dim.count Dim.t
val offset : shape -> coord -> Dim.offset Dim.t (* dense NHWC linear index *)
val in_bounds : shape -> coord -> bool

val read_coord :
  shape -> coord -> coord (* broadcast: extent-1 axis -> index 0 *)

val iter : shape -> (coord -> unit) -> unit (* C innermost *)

val pp_shape :
  Format.formatter -> shape -> unit (* [H=2 W=2 C=3] — leading 1s trimmed *)

val pp_coord :
  Format.formatter -> coord -> unit (* (1,0,2) — leading 0s trimmed *)
