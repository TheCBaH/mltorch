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

(* A private record, not a fully abstract type: axis count is fixed (6) and
   every hot-path caller accesses by a statically-known axis, so exposing
   the fields for reading lets [Tensor.read_at]/[Schedule.coord_index] (the
   two innermost, highest-call-count paths in the engine — see
   .ai/pt2_inference_perf.md) read a component as a direct record
   projection instead of going through [get]'s [Axis.t] match + the (real,
   non-flambda-inlined) cross-module call that used to cost ~26% of total
   runtime on its own. [private] (same idiom as [Dim.t]) keeps construction
   and [{ v with ... }] updates funneled through [shape]/[coord]/[deltas]/
   [set] below — outside this module, a value can only be read, never built
   or mutated by writing record syntax directly. *)
type 'd t = private { n : 'd; t : 'd; d : 'd; h : 'd; w : 'd; c : 'd }
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

(* [in_bounds]+[offset] fused, without needing a [coord] — for [Tensor.read_at]
   specifically. Components are already-validated [Dim.index Dim.t] (the
   caller's producer already went through [Dim.index] — see direct.ml), so
   only the upper bound is checked here. Returns the dense linear offset, or
   [-1] (never a valid offset) if the point isn't fully inside [shape]. *)
val offset_of :
  shape ->
  n:Dim.index Dim.t ->
  t:Dim.index Dim.t ->
  d:Dim.index Dim.t ->
  h:Dim.index Dim.t ->
  w:Dim.index Dim.t ->
  c:Dim.index Dim.t ->
  int

val iter : shape -> (coord -> unit) -> unit (* C innermost *)

val pp_shape :
  Format.formatter -> shape -> unit (* [H=2 W=2 C=3] — leading 1s trimmed *)

val pp_coord :
  Format.formatter -> coord -> unit (* (1,0,2) — leading 0s trimmed *)

(* [n, t, d, h, w, c] — all 6 extents, no trimming. *)
val shape_jsont : shape Jsont.t
