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
   [Expr.index_expr] AST node) is unaffected — [position]/[delta] stay purely
   phantom there either way. *)
type position = Dim.index
type delta = Dim.delta

module type SEMANTICS = sig
  type t
  type 'role index

  (* value domain — minimal basis. [max]/[min]/[relu] are NOT here: they derive
     from [select]+[lt] at each call site (relu = select (lt x 0) 0 x), so a
     new activation costs no new primitive, Expr constructor, or eval/pp arm. *)
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
  val index_eq : delta index -> delta index -> b
  val clamp_low : delta index -> position index
  val assume_index : delta index -> position index

  type input

  val load : input -> (Axis.t -> position index) -> t

  (* Same as [load], 6 explicit indices instead of a closure — for hot call
     sites where a closure built fresh per call (one per axis-lookup, so
     effectively per element) would allocate every time. See
     .ai/pt2_inference_perf.md. Not every op has migrated to this yet
     (harmless — [load] stays the general form); [conv.ml]'s innermost
     reduction has, since it's the highest-call-volume site in the engine. *)
  val load6 :
    input ->
    n:position index ->
    t:position index ->
    d:position index ->
    h:position index ->
    w:position index ->
    c:position index ->
    t

  val sum : lo:position index -> hi:delta index -> (position index -> t) -> t

  val max_reduce :
    lo:position index -> hi:delta index -> (position index -> t) -> t
end
