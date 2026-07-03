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

(* Unchecked general-purpose constructor, for a payload that isn't a [Dim.t]
   role at all (so there's nothing for [shape]/[coord]/[deltas]'s checked
   constructors to check) — e.g. [Expr.Load]'s 6 per-axis sub-expressions.
   [shape]/[coord]/[deltas] are themselves built on this. *)
val make : n:'d -> t:'d -> d:'d -> h:'d -> w:'d -> c:'d -> 'd t

(* Elementwise transform / fold, in [Axis.all]'s N/T/D/H/W/C order. The [i]
   forms hand the axis to [f] as well — for logic that treats an axis
   specially (e.g. looking it up in another [Vec6.t] or an association
   list), avoiding a [get]-by-axis inside a [match]/closure. *)
val map : ('a -> 'b) -> 'a t -> 'b t
val mapi : (Axis.t -> 'a -> 'b) -> 'a t -> 'b t
val fold : ('acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc
val foldi : ('acc -> Axis.t -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc

(* Materializes an [Axis.t -> 'd] function into a ['d t] by calling it once
   per axis — for a caller holding a per-axis function (e.g. a [SEMANTICS]
   op's base output coordinate) that needs a [Vec6.t] to pipe through
   [set_h] etc. or pass to [SEMANTICS.load]. *)
val of_fn : (Axis.t -> 'd) -> 'd t
val get : 'd t -> Axis.t -> 'd
val set : 'd t -> Axis.t -> 'd -> 'd t

(* Per-axis setters, new value first / vec last, so [base |> set_h h' |>
   set_w w'] pipes: each partially applies to a ['d t -> 'd t] transformer.
   For op code overriding one or two axes of an existing coordinate. *)
val set_n : 'd -> 'd t -> 'd t
val set_t : 'd -> 'd t -> 'd t
val set_d : 'd -> 'd t -> 'd t
val set_h : 'd -> 'd t -> 'd t
val set_w : 'd -> 'd t -> 'd t
val set_c : 'd -> 'd t -> 'd t

(* Copies the source vec's value at [axis] into the same [axis] of the piped
   vec — one call and one axis dispatch, not [set v axis (get src axis)]'s
   two. The source vec is positional (not [~src]-labeled): with only one
   axis role there's nothing to disambiguate, and it keeps [copy_axis src
   axis] a "pre-filled args first, piped vec last" transformer like [set_h]
   etc. For op code overriding some of an existing vec's axes from another
   vec at the same axis (e.g. an all-zero base with a few axes copied over
   from a materialized output coordinate). *)
val copy_axis : 'd t -> Axis.t -> 'd t -> 'd t

(* General form: the source vec's value at axis [~src] into axis [~dst] of
   the piped vec — for the rarer case where the source and destination axes
   differ. The source vec stays positional (see [copy_axis]); [~src]/[~dst]
   label the two axis roles, which are what actually need disambiguating at
   a call site. *)
val copy : 'd t -> src:Axis.t -> dst:Axis.t -> 'd t -> 'd t

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
