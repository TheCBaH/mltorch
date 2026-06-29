(* A 6-component vector over the axes N T D H W C. The single type parameter is
   the per-axis [Dim] role it carries, which is the only discriminator — there is
   no separate vector-role phantom: the role of the components IS the role of the
   vector. So the three per-axis dimensional roles give three non-unifiable vector
   types ([Dim.count]/[Dim.offset] are flattened whole-tensor scalars, not per-axis
   vectors, so they have no [Vec6] form):

     - [shape]  — extents (sizes, each >= 1)
     - [coord]  — indices (positions, 0 <= a_i < extent_i)
     - [deltas] — signed per-axis displacements (a stencil / shift)

   [get] hands back the right scalar dimensional type, and the operations bridging
   shape and coord ([numel], [offset], [in_bounds], [iter]) take both so their
   arguments can't be swapped. See .ai/native_tensor_design.md §1b. *)

type 'd t
type shape = Dim.extent Dim.t t
type coord = Dim.index Dim.t t
type deltas = Dim.delta Dim.t t

val shape : n:int -> t:int -> d:int -> h:int -> w:int -> c:int -> shape
val coord : n:int -> t:int -> d:int -> h:int -> w:int -> c:int -> coord
val deltas : n:int -> t:int -> d:int -> h:int -> w:int -> c:int -> deltas
val origin : coord (* all zero *)
val get : 'd t -> Axis.t -> 'd
val set : 'd t -> Axis.t -> 'd -> 'd t

(* the role-bridging operations *)
val numel : shape -> Dim.count Dim.t
val offset : shape -> coord -> Dim.offset Dim.t (* dense NHWC linear index *)
val in_bounds : shape -> coord -> bool
val iter : shape -> (coord -> unit) -> unit (* C innermost *)

val pp_shape :
  Format.formatter -> shape -> unit (* [H=2 W=2 C=3] — leading 1s trimmed *)

val pp_coord :
  Format.formatter -> coord -> unit (* (1,0,2) — leading 0s trimmed *)
