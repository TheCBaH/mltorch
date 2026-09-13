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
  val erf : t -> t
  val log : t -> t
  val cos : t -> t
  val sin : t -> t

  (* transcendental: genuinely primitive, not select-expressible (sqrt is the
     rms-norm normaliser; rsqrt is just 1 / sqrt; erf is gelu's exact form;
     log is Pow's general fallback, exp(exponent * log x) -- see
     [Pointwise.Pow]; cos/sin are EdgeNeXt's/ViT's Fourier positional
     encoding) *)
  val trunc : t -> t
  (* round toward zero -- genuinely primitive, not select-expressible (unlike
     [max]/[min]/[relu]): there is no finite composition of [add]/[mul]/[select]
     that discards a value's fractional part. Needed for [To_copy]'s
     integer-dtype target, matching ATen's [static_cast<IntT>] cast. *)

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

  (* Reads a tensor's own stored VALUE at a coordinate and reports it as a
     POSITION index for further indexing -- [index.Tensor]'s runtime gather,
     the value-to-index bridge going the opposite direction from
     [value_of_index]. [extent] is the gathered axis's extent, for
     bounds-checking/negative-index normalization: [Direct] resolves it
     eagerly (raising on failure, since [SEMANTICS]' index domain is
     total-by-type — see direct.ml); [Symbolic] defers it to grounding
     (construction only, no check, via [Expr.Index.data]). *)
  val load_index :
    input -> position index Vec6.t -> extent:Dim.extent Dim.t -> position index

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

  (* [max.dim]'s paired value/index reduction: same [lo]/[hi]/[f] shape as
     [max_reduce], but folding with [Max_op.pool_better] rather than
     [Float.max], and [max_dim_index] reports the winning position (carried
     into the value domain the way [value_of_index] does) rather than the
     winning value. The two MUST be called with the same [lo]/[hi]/[f] at a
     given call site -- that pairing, not a single call returning both
     halves, is what keeps them from falling out of step, exactly as
     [max_pool2d]/[max_pool2d_index] already rely on for the fixed-window
     case. See [Expr.Reduction.Argmax_value]/[Argmax_index]. *)
  val max_dim :
    lo:position index -> hi:delta index -> (position index -> t) -> t

  val max_dim_index :
    lo:position index -> hi:delta index -> (position index -> t) -> t
end

(* Carrier-indexed sibling of [SEMANTICS], added alongside it rather than
   replacing it (see .ai/'s evaluator dtype design, "Semantics functors and
   construction effects") -- migrating [SEMANTICS] itself to a functor
   parameter would still leave every call site needing ONE float-shaped [t],
   which cannot express a cast, an I64 comparison, or a mixed-carrier [select]
   without a second domain in the same computation. [Direct] instantiates
   [type 'a repr = 'a] (its values are already the OCaml values they denote);
   [Symbolic] instantiates [type 'a repr = 'a Expr.Value.t Expr.Builder.t]
   (mirroring [SEMANTICS.t]'s own builder-computation shape, for the same
   reason: a symbolic value is a construction, and using one twice would
   construct it twice -- see symbolic.mli).

   Deliberately narrower than the design's illustrative sketch in two ways,
   both because the machinery they'd need does not exist yet and building it
   ahead of a real caller would be exactly the premature abstraction
   CLAUDE.md warns against:
   - [input] stays a single, non-carrier-indexed type (matching [SEMANTICS]'s
     own [input]), not a checked ['a input]/[Output_spec]-shaped tensor view.
     [I64_load]'s format check already happens inside the read itself
     ([Tensor.read_i64_at6]'s [`Wrong_format] / [Expr.Index.data]'s deferred
     grounding-time check) -- see P2.1/P2.2's tracker notes, which found no
     real caller yet needing one checked entry point across formats and left
     that open against Region's own typed output-group work.
   - Arithmetic/comparison/cast are named per concrete carrier
     ([i64_binary], [i64_eq], [i64_lt], [float_to_i64], [i64_to_float]) rather
     than through one witness-indexed [binary]/[cast]/[equal] family: the
     underlying [Expr.Value.t]/[Expr.Bool.t] representation keeps a separate
     [i64_binary_op] from the float [binary_op] and keeps [bool_expr] a
     non-GADT type outside [_ value] (both deliberate choices recorded at
     [[P1.1]]/[[P1.2]] in the implementation tracker) -- unifying the op
     witnesses here would require re-deriving that representation split
     first, which is no part of this gate's scope.
   [b] mirrors [SEMANTICS.b]: the boolean PREDICATE domain, not a [bool repr]
   -- [bool_expr]'s own non-GADT status above is exactly why [select]'s
   condition is typed [b], not [bool repr]. [const] still takes a full
   [Scalar.t] witness (matching the design literally, unlike the narrowings
   above): its [Bool] arm is unreachable on both instances (there is no
   [bool Expr.Value.t] constructor to build and [Direct] never calls it,
   since nothing here produces a [bool Scalar.t] witness) and says so with
   [assert false], the same idiom [Eval.value]'s own [Scalar.t]-witnessed
   dispatch already uses for the identical gap. *)
module type TYPED_SEMANTICS = sig
  type 'a repr
  type 'role index
  type input
  type b

  (* [typed_]-prefixed, not [const]/[select]: [SEMANTICS] above already
     declares those names at a different (monomorphic [t]) type, and one
     signature cannot declare the same value name twice even when a single
     polymorphic implementation could satisfy both -- see direct.ml/
     symbolic.ml, where [const]/[select] stay the legacy entry points and
     [typed_const]/[typed_select] are genuinely separate bindings (Direct) or
     the same generic definition exposed a second time at a wider type
     (Symbolic, where the legacy [select] already generalizes). *)
  val typed_const : 'a Expr.Scalar.t -> 'a -> 'a repr
  val typed_select : b -> 'a repr -> 'a repr -> 'a repr

  val i64_binary :
    Expr.Value.i64_binary_op -> int64 repr -> int64 repr -> int64 repr

  val i64_eq : int64 repr -> int64 repr -> b
  val i64_lt : int64 repr -> int64 repr -> b
  val i64_load : input -> position index Vec6.t -> int64 repr
  val float_to_i64 : float repr -> int64 repr
  val i64_to_float : int64 repr -> float repr
end
