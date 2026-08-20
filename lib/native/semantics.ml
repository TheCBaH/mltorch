(* Abstract domain for ops: both [t] (value) and ['role index] (index) are abstract,
   so one op functor runs as [Direct] (index=int, t=float), [Symbolic] (index=index_expr,
   t=expr), or a future [Footprint] (index=interval). See .ai/native_compute_design.md §1.

   Index phantom roles: [position] is a known ≥ 0 (what [load] accepts);
   [delta] is a signed affine. Windowed arithmetic is built in [delta] and
   converted to [position] only via [clamp_low] (sound: ≥ 0) or [assume_index]
   (one encapsulated unchecked claim, in Window_axis.window).

   Aliased to [Dim.index]/[Dim.delta] (see dim.mli), not fresh local marker
   types: a semantics whose index representation IS a [Dim.t] (e.g. [Direct]
   — see .ai/pt2_inference_perf.md) can then use [position index = Dim.index
   Dim.t] directly, with zero coercion needed at [Tensor.read_at]/
   [Vec6.offset_of], the innermost per-element call sites. A semantics that
   doesn't use [Dim.t] at all (e.g. [Symbolic], whose index is an
   [Expr.Index.t] AST node) is unaffected — [position]/[delta] stay purely
   phantom there either way. *)
type position = Dim.index
type delta = Dim.delta

module type SEMANTICS = sig
  type t
  type 'role index

  (* value domain — minimal basis. [max]/[min]/[relu] are NOT here: they derive
     from [select]+[lt] at each call site (relu = select (lt x 0) 0 x), so a
     new activation costs no new primitive, Expr.Value constructor, or eval/pp arm. *)
  val const : float -> t
  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val div : t -> t -> t
  val exp : t -> t
  val sqrt : t -> t
  (* transcendental: genuinely primitive, not select-expressible (sqrt is the
     rms-norm normaliser; rsqrt is just 1 / sqrt) *)

  (* boolean domain + selection — the scalable basis for activations/clamps.
     [select c a b] is [a] when [c] holds else [b]; e.g.
     [relu x = select (lt x (const 0.)) (const 0.) x]. *)
  type b

  val lt : t -> t -> b
  val select : b -> t -> t -> t

  (* index domain — affine expressions in [delta]; [load] needs [position].
     [clamp_low] (max 0) converts delta→position soundly; [assume_index] is the
     one unchecked cast, encapsulated in Window_axis.window (where clip holds). *)
  val index_zero : position index
  val index_extent : Dim.extent Dim.t -> delta index
  val index_const : int -> delta index
  val of_index : position index -> delta index
  val index_add : delta index -> delta index -> delta index
  val index_scale : int -> delta index -> delta index
  val index_floor_div_pos : delta index -> Op_config.Pos.t -> delta index
  val index_ceil_div_pos : delta index -> Op_config.Pos.t -> delta index
  val index_min : delta index -> delta index -> delta index

  (* [index_max] exists for [Pad]'s reflect mirror, which needs [abs] on an
     index: [|x| = max x (-x)], and the mirror is
     [n-1 - |n-1 - |j||]. Expressible as [-(min x (-x))] with [index_min]
     alone, and deliberately not written that way -- the negation trick costs a
     reader more than the primitive costs the two semantics, and [Expr] already
     had the [Max] constructor with every eval/pp/compare/traversal arm, so this
     adds no AST node.

     What is NOT here, and should not be, is an index COMPARISON. [Expr.Bool]
     has only [Value_lt] and [Index_eq]; an [index_lt] would be a genuine new
     constructor. [Pad]'s "is this coordinate inside the source" test avoids
     needing one by clamping the coordinate into range and comparing the clamped
     value with the unclamped one through [index_eq] -- which also keeps the
     [load] in bounds, since [select] evaluates both arms. *)
  val index_max : delta index -> delta index -> delta index
  val index_eq : delta index -> delta index -> b
  val clamp_low : delta index -> position index
  val assume_index : delta index -> position index

  (* Carry an index into the value domain as its ordinal (a float). The one
     value/index bridge besides [load] — argmax (max_pool2d_with_indices) needs
     the flat position of the max as a value. *)
  val value_of_index : delta index -> t

  type input

  (* [Vec6.t], not a closure: most call sites only override one or two axes
     of an existing coordinate (see [Vec6.set_h] etc.), which pipes cleanly
     into a [Vec6.t]-typed argument but needs a fresh closure allocation
     every call if [load] instead took [Axis.t -> position index] (the
     original design — see .ai/pt2_inference_perf.md for why that mattered).
     Building the full 6-field [Vec6.t] is still allocation-free for
     [Direct] on the hot path via [load6] below. *)
  val load : input -> position index Vec6.t -> t

  (* Same as [load], 6 explicit indices instead of a [Vec6.t] — for the
     hottest call sites, where even the (functional-update, so allocating)
     [Vec6.set_*] pipeline used to build [load]'s argument costs more than
     passing scalars directly. See .ai/pt2_inference_perf.md.
     [conv.ml]'s innermost reduction uses this, since it's the
     highest-call-volume site in the engine. *)
  val load6 :
    input ->
    n:position index ->
    t:position index ->
    d:position index ->
    h:position index ->
    w:position index ->
    c:position index ->
    t

  (* Fixed-window pool primitives.  These stay scalar operations so symbolic
     semantics can retain the complete window geometry in one expression node
     rather than expanding it into nested generic reductions. *)
  val max_pool2d :
    input ->
    x_shape:Vec6.shape ->
    kernel:Dim.extent Dim.t Op_config.Hw.t ->
    stride:Op_config.Pos.t Op_config.Hw.t ->
    pad:Op_config.Nonneg.t Op_config.Hw.t ->
    position index Vec6.t ->
    t

  val max_pool2d_index :
    input ->
    x_shape:Vec6.shape ->
    kernel:Dim.extent Dim.t Op_config.Hw.t ->
    stride:Op_config.Pos.t Op_config.Hw.t ->
    pad:Op_config.Nonneg.t Op_config.Hw.t ->
    position index Vec6.t ->
    t

  val sum : lo:position index -> hi:delta index -> (position index -> t) -> t

  val max_reduce :
    lo:position index -> hi:delta index -> (position index -> t) -> t
end
