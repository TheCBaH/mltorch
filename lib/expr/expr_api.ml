module type S = sig
  (* The public surface of the expression language. See expr.ml for why nearly
     everything lives in one compilation unit, and
     .ai/native_expr_refactoring_design.md for the language.

     The order here mirrors expr.ml and is required, not stylistic: [Index] and
     [Intrinsic] must precede the recursive group that references them. *)

  module Axis = Expr_internal.Axis
  module Role = Expr_internal.Role
  module Coord = Expr_internal.Coord
  module Source = Expr_internal.Source

  module Max_op = Expr_internal.Max_op
  (** Lives here, not in [lib/native]: [Eval] cannot depend on [native], so
      leaving it there would force a second copy of the NaN/signed-zero rule --
      the exact duplication that produced the divergence it exists to prevent.
  *)

  type index_op = [ `Add | `Mul | `Sub ]
  (** The three checked-arithmetic operations that can overflow. *)

  module Index_overflow : sig
    type t = { op : index_op; lhs : int; rhs : int }
    (** Which operation overflowed, on which operands. Re-exported from the
        library-private checked arithmetic: the rows below are public, so a
        caller branching on one has to be able to name its payload. *)
  end

  module Reduce_var : sig
    (* Abstract, and with NO public constructor: only [Builder] mints one, which
       is what makes reducer scope enforceable. Identity is meaningful only within
       an expression; [Pp] assigns display names lexically, so nothing observable
       depends on the internal ordinal. *)
    type t

    val compare : t -> t -> int
    val equal : t -> t -> bool
    val hash : t -> int
    val pp : Format.formatter -> t -> unit

    module Map : Map.S with type key = t
    module Set : Set.S with type elt = t
  end

  module Local_var : sig
    type t

    val compare : t -> t -> int
    val equal : t -> t -> bool
    val hash : t -> int
    val pp : Format.formatter -> t -> unit

    module Map : Map.S with type key = t
    module Set : Set.S with type elt = t
  end

  module Index : sig
    (* [private]: consumers pattern-match but cannot construct, so an invalid
       divisor or a delta silently reused as a position cannot enter the AST.

       The role parameter is the point of the GADT. Arithmetic happens in
       [Role.Delta.t]; only [clamp_low] (sound, max 0) and [assume_position] (a
       recorded claim, provable only against a caller's coordinate domain) cross
       back to [Role.Position.t], which is what [Value.load] accepts. *)
    type _ t = private
      | Add : Role.Delta.t t * Role.Delta.t t -> Role.Delta.t t
      | Assume_position : Role.Delta.t t -> Role.Position.t t
      | Ceil_div_pos : Role.Delta.t t * int -> Role.Delta.t t
      | Clamp_low : Role.Delta.t t -> Role.Position.t t
      | Const : int -> Role.Delta.t t
      | Data : Source.t * Role.Position.t t Coord.t * int -> Role.Position.t t
      | Floor_div_pos : Role.Delta.t t * int -> Role.Delta.t t
      | Max : Role.Delta.t t * Role.Delta.t t -> Role.Delta.t t
      | Min : Role.Delta.t t * Role.Delta.t t -> Role.Delta.t t
      | Of_position : Role.Position.t t -> Role.Delta.t t
      | Output : Axis.t -> Role.Position.t t
      | Reduce : Reduce_var.t -> Role.Position.t t
      | Scale : int * Role.Delta.t t -> Role.Delta.t t
      | Zero : Role.Position.t t

    type error = [ `Non_positive_divisor of int ]

    val pp_error : Format.formatter -> [< error ] -> unit
    val output : Axis.t -> Role.Position.t t
    val reduce : Reduce_var.t -> Role.Position.t t
    val zero : Role.Position.t t
    val const : int -> Role.Delta.t t
    val of_position : Role.Position.t t -> Role.Delta.t t
    val add : Role.Delta.t t -> Role.Delta.t t -> Role.Delta.t t
    val scale : int -> Role.Delta.t t -> Role.Delta.t t
    val min : Role.Delta.t t -> Role.Delta.t t -> Role.Delta.t t
    val max : Role.Delta.t t -> Role.Delta.t t -> Role.Delta.t t
    val clamp_low : Role.Delta.t t -> Role.Position.t t
    val assume_position : Role.Delta.t t -> Role.Position.t t

    val data : Source.t -> Role.Position.t t Coord.t -> int -> Role.Position.t t
    (** The one primitive that reads a tensor's own stored VALUE and uses it as
        an index component -- [index.Tensor]'s runtime gather. [extent] is a
        plain [int] (the gathered axis's extent), not the host's own typed
        extent representation: this library must not depend on [native], where
        that type lives. No folding, and no validation at construction time --
        resolution happens once, where the value becomes concrete
        ([Eval.eval_index]'s own [Data] arm), never here. *)

    (* Result-returning: the divisor is a raw [int], and a zero or negative one
       must never reach the AST. [d = 1] folds to the operand.

       [add] and [scale] fold too -- x+0, 0+x, 1*x, k*0 -- and only those: exact
       integer identities, never a floating-point one, where dropping an addition
       of zero or reassociating is observable through signed zero, NaN and
       rounding. *)
    val floor_div_pos : Role.Delta.t t -> int -> (Role.Delta.t t, error) Err.t
    val ceil_div_pos : Role.Delta.t t -> int -> (Role.Delta.t t, error) Err.t
  end

  module Intrinsic : sig
    (* The compact max-pool family: one closed intrinsic, kept as a single node
       rather than expanded into nested reductions so the window geometry stays
       visible to later codegen and footprint analysis. *)
    module Window : sig
      type t = { hlo : int; hhi : int; wlo : int; whi : int }
    end

    module Max_pool : sig
      type result = Index | Value

      val result_name : result -> string
      (** The spelling used in printed output, alongside [Value.unary_name] and
          [Value.binary_sym]. Named rather than inlined at the printer because
          it is part of a rendered form that expect tests pin. *)

      type t = private {
        source : Source.t;
        in_h : int;
        in_w : int;
        kernel_h : int;
        kernel_w : int;
        stride_h : int;
        stride_w : int;
        pad_h : int;
        pad_w : int;
        out : Role.Position.t Index.t Coord.t;
        result : result;
      }
    end

    type t = private Max_pool of Max_pool.t

    type geometry_field =
      [ `In_h
      | `In_w
      | `Kernel_h
      | `Kernel_w
      | `Pad_h
      | `Pad_w
      | `Stride_h
      | `Stride_w ]
    (** The eight parameters {!max_pool} validates, closed. *)

    type geometry_bound = [ `Non_negative | `Positive ]
    (** Which bound the value failed. Recorded nowhere before: the message said
        only "must be valid", so the row could not say what would have been. *)

    module Bad_geometry : sig
      type t = { field : geometry_field; value : int; bound : geometry_bound }
    end

    type error =
      [ `Bad_geometry of Bad_geometry.t
      | `Index_overflow of Index_overflow.t
      | `Non_positive_divisor of int ]

    val pp_error : Format.formatter -> [< error ] -> unit

    val max_pool :
      source:Source.t ->
      in_h:int ->
      in_w:int ->
      kernel_h:int ->
      kernel_w:int ->
      stride_h:int ->
      stride_w:int ->
      pad_h:int ->
      pad_w:int ->
      out:Role.Position.t Index.t Coord.t ->
      result:Max_pool.result ->
      (t, error) Err.t
    (** The only way to build a descriptor. [Check] does NOT revalidate this: a
        smart constructor makes the invalid state unconstructable through the
        public API, and a rule no test can turn red is not worth carrying. *)

    val window : t -> out_h:int -> out_w:int -> (Window.t, error) Err.t
    (** The half-open input window an output position reads, clipped to the
        input extents.

        Public and result-returning because there are TWO interpreters: native's
        [Ground_eval] expands the same stencil, and as a consumer it cannot
        reach the library-private checked arithmetic. If only [Eval] went
        through this, the verifier's copy would stay unchecked and the two would
        drift — the same failure mode [Max_op] exists to prevent. Validating the
        descriptor's fields does not cover it either: [out_h * stride] is an
        aggregate and needs its own bound. *)

    val flat_index : t -> ih:int -> iw:int -> (int, error) Err.t
    (** The flattened input position a max-pool index result reports. Checked
        for the same reason — [ih * in_w] is an aggregate. *)
  end

  module rec Bool : sig
    type t = private
      | Index_eq of Role.Delta.t Index.t * Role.Delta.t Index.t
      | Value_lt of Value.t * Value.t

    val value_lt : Value.t -> Value.t -> t
    val index_eq : Role.Delta.t Index.t -> Role.Delta.t Index.t -> t
  end

  and Reduction : sig
    type kind = Max | Sum

    val kind_name : kind -> string
    (** [max_reduce], not [max]: this is the generic ordered reduction, distinct
        from the max-pool intrinsic, and printed output must keep the two
        distinguishable. Carries the name the old representation used. *)

    type t = private {
      kind : kind;
      var : Reduce_var.t;
      lo : Role.Position.t Index.t;
      hi : Role.Delta.t Index.t;
      body : Value.t;
    }
  end

  and Value : sig
    type binary_op = Add | Div | Mul | Sub
    type unary_op = Erf | Exp | Log | Sqrt | Trunc

    type t = private
      | Binary of binary_op * t * t
      | Const of float
      | Intrinsic of Intrinsic.t
      | Local of Local_var.t
      | Local_at of Local_var.t * Role.Position.t Index.t
      | Load of Source.t * Role.Position.t Index.t Coord.t
      | Reduce of Reduction.t
      | Round_f32 of t
      | Select of Bool.t * t * t
      | Unary of unary_op * t
      | Value_of_index of Role.Delta.t Index.t

    val const : float -> t
    val add : t -> t -> t
    val sub : t -> t -> t
    val mul : t -> t -> t
    val div : t -> t -> t
    val exp : t -> t
    val sqrt : t -> t
    val erf : t -> t
    val log : t -> t

    val trunc : t -> t
    (** Round toward zero -- ATen's `static_cast<IntT>` for a float-to-int
        [_to_copy.default]/[to.dtype] cast. Distinct from [round_f32], which
        rounds to f32 storage precision, not to an integer. *)

    val select : Bool.t -> t -> t -> t
    val value_of_index : Role.Delta.t Index.t -> t
    val load : Source.t -> Role.Position.t Index.t Coord.t -> t
    val round_f32 : t -> t
    val intrinsic : Intrinsic.t -> t
    val local : Local_var.t -> t

    val local_at : Local_var.t -> Role.Position.t Index.t -> t
    (** Reads a vector local's element at a computed index -- a [Region_local]
        vector's "body may mention the binder" applied at a call site. The
        counterpart to [Region_local.vector]'s own binder-parameterised body:
        substituting it out (during specialization) is a beta-reduction, not a
        lookup. *)

    (* [Ground_expr] stores these operator payloads and applies them, so they are
       public: it is what lets the ground language share one definition of the
       arithmetic instead of open-coding a second. *)
    val apply_binary : binary_op -> float -> float -> float
    val apply_unary : unary_op -> float -> float
    val binary_sym : binary_op -> string
    val unary_name : unary_op -> string

    val compare : t -> t -> int
    (** Structural, up to ALPHA-EQUIVALENCE: two expressions differing only in
        which ordinals their supplies handed out compare equal, because reducers
        are compared by binder level rather than identity. A free reducer keeps
        its identity and can still separate two otherwise-equal terms.

        Expressions differing in nesting, bounds, kind, operand order or body do
        not compare equal — a reduction is an ordered fold, not a set.

        Constants compare by their IEEE-754 bits, except that all NaNs are
        canonicalised. JavaScript [Number] values cannot portably preserve a NaN
        payload, so canonicalisation keeps comparison reflexive across backends.
        Signed zero and all distinct non-NaN representations remain distinct. *)

    val equal : t -> t -> bool

    val hash : t -> int
    (** Agrees with [compare] by construction — same information, same order,
        reducers by level. An optimisation only; structural equality remains the
        authority. *)
  end

  module Builder : sig
    (* A threaded immutable supply, not a global counter. [run] starts a fresh
       identity namespace; [run_from] continues an existing one, for when several
       computations must share a namespace instead of each restarting at ordinal
       0. Two computations run from [initial] deliberately reuse ordinals —
       identity is meaningful only within one expression, and composing across
       that boundary is what [Rewrite.freshen] is for. *)
    type state
    type 'a t

    exception Supply_exhausted
    (** Raised by [fresh_reduce] rather than wrapping the ordinal. A wrap would
        re-mint a live identity and invalidate scope, alpha-equivalence,
        comparison and hashing together, so it is a checked boundary. *)

    val initial : state
    val return : 'a -> 'a t
    val bind : 'a t -> ('a -> 'b t) -> 'b t
    val map : ('a -> 'b) -> 'a t -> 'b t
    val run : 'a t -> 'a
    val run_from : state -> 'a t -> 'a * state
    val fresh_reduce : Reduce_var.t t
    val fresh_local : Local_var.t t

    module Syntax : sig
      val ( let* ) : 'a t -> ('a -> 'b t) -> 'b t
      val ( let+ ) : 'a t -> ('a -> 'b) -> 'b t
    end

    val reduction :
      kind:Reduction.kind ->
      lo:Role.Position.t Index.t ->
      hi:Role.Delta.t Index.t ->
      (Role.Position.t Index.t -> Value.t t) ->
      Value.t t
    (** Allocates the variable, hands its index expression to the body, and
        threads the supply through. Going through here rather than assembling a
        [Reduction.t] is what makes scope correct by construction: the body
        cannot name a variable that is not this reduction's, and the variable
        cannot escape into a sibling. *)
  end

  module Fold : sig
    (* Concrete, reviewed queries rather than a generic visitor, so binder
       behaviour stays visible in each signature and a new constructor breaks the
       traversals that must handle it instead of falling through a default. *)

    val size : Value.t -> int
    (** Node count, index trees included — a load's addressing is where the bulk
        of a large expression lives, so treating it as a leaf would measure
        almost nothing. Unmetered: it walks the whole tree, which is why
        [Check.value] does not use it to enforce a limit. *)

    val depth : Value.t -> int
    (** Nesting depth, index trees included, for the same reason as [size]. Also
        unmetered, and computed by the same traversal, so the two cannot
        disagree about what counts as a node or a level. *)

    val measure_with_locals :
      local:(Local_var.t -> int * int) ->
      max_size:int ->
      max_depth:int ->
      Value.t ->
      int * int
    (** Measures a prospective substitution without constructing it. Each local
        leaf contributes the supplied expanded [(size, depth)] pair. Callers
        must first use [exceeds_with_locals] under the same bounds; this query
        is intentionally the exact, unmetered result after that preflight. *)

    val exceeds_with_locals :
      local:(Local_var.t -> int * int) ->
      max_size:int ->
      max_depth:int ->
      Value.t ->
      [ `Depth | `Size ] option
    (** Saturating prospective-substitution preflight. It stops at the first
        configured bound rather than allocating or measuring the expansion. *)

    val sources : Value.t -> Source.Set.t
    (** Every source the expression depends on, ordinary loads and intrinsic
        descriptors alike. What must be RESOLVED and ordered. *)

    val loads : Value.t -> (Source.t * Role.Position.t Index.t Coord.t) list
    (** Ordinary [Load] sites with their coordinates, in lexical order, WITH
        repeats. What may be SUBSTITUTED. Deliberately not derivable from
        [sources], which is a set — it loses multiplicity and addressing, and
        folds in intrinsic sources that no rewrite can replace. *)

    val intrinsic_sources : Value.t -> Source.t list
    (** Sources reached through an intrinsic descriptor, in lexical order. Real
        dependencies, but not loads: there is no node inside the descriptor a
        subtree could stand in for. Kept separate from [loads] because the two
        are treated differently, not because one is a subset of the other. *)

    val locals : Value.t -> Local_var.Set.t

    val scalar_locals : Value.t -> Local_var.Set.t
    (** The subset of [locals] read as a plain [Value.Local] -- what a
        scalar-shaped local may legally be. *)

    val vector_locals : Value.t -> Local_var.Set.t
    (** The subset of [locals] read as [Value.Local_at] -- what a vector-shaped
        local may legally be. Disjoint from [scalar_locals] in a well-formed
        program; the host's shape-agreement check is exactly what rules out the
        id appearing in both. *)

    val output_axes : Value.t -> Axis.t list
    val intrinsics : Value.t -> int

    val assume_sites : Value.t -> int
    (** How many [Index.assume_position] claims the expression contains.
        [assume_position] proves nothing — it records an author's claim — so
        being able to locate every one is the point of keeping it a distinct
        node. *)

    val free_reducers : Value.t -> Reduce_var.Set.t
    (** Scope-aware: a reducer under its own binder is bound, not free, and a
        reduction's BOUNDS sit outside its binder. A well-formed top-level
        expression has none. *)

    val binders : Value.t -> Reduce_var.t list
    (** Binders in lexical order, with repeats — one identity bound in two
        sibling scopes appears twice, which is what separates counting binders
        from counting identities. Inspection only: [Pp] and the structural
        comparison each carry a scoped environment instead, since a list keyed
        by identity cannot tell those siblings apart. *)
  end

  module Rewrite : sig
    (* Every rewrite here rebuilds with the RAW constructors, not the smart ones:
       those fold, and a structure-preserving rewrite must not silently change
       syntax. No consumer may recursively rewrite a [Value.t] while ignoring
       reducer scope — that is what this module exists to own. *)

    val freshen : Value.t -> Value.t Builder.t
    (** Replaces every BOUND reducer identity consistently; free ones are left
        alone.

        Freshen the fragment being INSERTED, before composing it. Two
        independently built fragments both start from [Builder.initial], so both
        mint ordinal 0 and the inner binder captures references meant for the
        outer one. Freshening the combined tree afterwards repairs nothing —
        once a nominal collision has captured a reference, there is no record of
        which binder it meant. *)

    val substitute_output :
      Role.Position.t Index.t Coord.t -> Value.t -> Value.t
    (** Replaces only output-axis variables, never reducers. If the result is
        placed beneath another reduction, the caller freshens it first; this
        function cannot know that context. *)

    val alpha_normalize : Value.t -> Value.t
    (** Deterministic renaming by lexical traversal — binders take the lowest
        ordinals NOT occurring free in the expression, in the order they are
        met: 0, 1, ... for a closed expression, skipping the free ones
        otherwise, since [freshen] must not capture them. Canonical either way,
        because alpha-equivalent expressions have the same free set and so skip
        the same ordinals. Does not reorder operations or reductions.
        Idempotent. *)

    val map_sources : (Source.t -> Source.t) -> Value.t -> Value.t
    (** Changes source symbols and nothing else, including inside an intrinsic
        descriptor. *)

    val substitute_loads :
      (Source.t -> Role.Position.t Index.t Coord.t -> Value.t Builder.t option) ->
      Value.t ->
      Value.t Builder.t
    (** Replaces ordinary [Load] nodes with whole subtrees; [None] keeps the
        load.

        Returning a computation is the point: the replacement is minted in the
        SAME namespace as the destination, so a caller freshens each inserted
        fragment by threading this one state. Running [freshen] from
        [Builder.initial] per fragment instead would have each mint ordinal 0
        again and reintroduce, at the moment of composing, exactly the collision
        [freshen] exists to prevent.

        INSERTED SUBTREES ARE NOT RE-TRAVERSED. A replacement containing its own
        loads keeps them, so a caller composing a chain must either iterate
        deliberately or restrict itself to non-overlapping edges — and any
        budget it enforces must account for the whole expansion rather than one
        step.

        Intrinsic descriptors are left structurally intact: a [Max_pool] holds a
        source and geometry, not a load node, so nothing inside it can be
        replaced by one scalar subtree. A caller needing to eliminate such a
        dependency must reject it. *)

    type local_binding =
      | Scalar of Value.t
      | Vector of { var : Reduce_var.t; body : Value.t }
          (** What a local resolves to at a use site. [Scalar] substitutes at a
              [Value.Local] occurrence; [Vector] substitutes [var] (the binder
              its [body] is parameterised over) with the occurrence's own read
              index at a [Value.Local_at] one -- a beta-reduction, not a lookup.
              Which node kind a given local may legally appear as is
              [Region_program.check]'s shape-agreement rule, not this
              function's: passing a [Vector] binding for a bare [Local]
              occurrence (or a [Scalar] one for a [Local_at]) raises, the same
              as any other well-formedness violation this module assumes its
              caller has already ruled out. *)

    val substitute_locals :
      (Local_var.t -> local_binding option) -> Value.t -> Value.t Builder.t
  end

  module Check : sig
    type error =
      [ `Duplicate_binder of Reduce_var.t
      | `Free_reducer of Reduce_var.t
      | `Too_deep of int
      | `Too_large of int
      | `Unbound_local of Local_var.t ]
    (** [`Too_large] and [`Too_deep] carry the LIMIT, not the measure: reporting
        the actual size would mean measuring the whole tree, which is what the
        limit is there to avoid. *)

    val pp_error : Format.formatter -> [< error ] -> unit

    val value :
      ?max_size:int -> ?max_depth:int -> Value.t -> (unit, error) Err.t

    val fragment :
      ?max_size:int ->
      ?max_depth:int ->
      ?allowed_free:Reduce_var.Set.t ->
      locals:Local_var.Set.t ->
      Value.t ->
      (unit, error) Err.t
    (** Deliberately narrow. Division parameters, load roles and intrinsic
        dimensions are NOT rechecked: the smart constructors and the [Index]
        GADT make each unconstructable through the public API, so such a rule
        could never be turned red, and CLAUDE.md is explicit that a check which
        has never failed is not evidence. What remains is what COMPOSITION can
        still violate — a free reducer, a binder shadowing itself (how two
        unfreshened fragments capture each other), and the optional size/depth
        limits.

        [allowed_free] exempts specific reducer identities from the free-
        reducer check -- for a vector local's own binder, deliberately free
        within its stored value (a [Region_local.vector]'s "body may mention the
        binder"). Empty by default, so an ordinary fragment's contract is
        unchanged.

        A configured limit is checked FIRST and is metered, stopping at the
        first node past it: the scope traversals recurse over the whole tree, so
        running them first would exhaust the stack on exactly the oversized
        input the limit exists to reject. An expression that is both oversized
        and ill-scoped is therefore reported as oversized.

        When both limits are given they are enforced by ONE traversal carrying
        both budgets, so the recursion is bounded by the tighter of the two. A
        walk per limit does not achieve that: whichever ran first would still
        descend the full input whenever its own bound was loose. *)
  end

  module Eval : sig
    module Gather_index_out_of_range : sig
      type t = { raw : int64; extent : int }
    end

    val pp_gather_index_out_of_range :
      Format.formatter -> Gather_index_out_of_range.t -> unit

    val resolve_gather_index :
      int64 ->
      extent:int ->
      ( int,
        [> `Gather_index_out_of_range of Gather_index_out_of_range.t ] )
      Err.t
    (** [index.Tensor]'s per-element gather-value validation: checks a raw
        stored [int64] against ATen's own valid range for a single index,
        [-extent, extent-1], BEFORE narrowing to [int] -- narrowing first would
        let a value near [Int64.min_int]/[Int64.max_int] wrap into a spuriously
        in-range [int]. Normalizes a negative value the way ATen does ([-1] =
        last element). Shared by every caller that resolves a [Data] index
        component ([Direct.load_index], [Eval]'s own [Data] arm, native's
        [Ground_eval]), so the bound is checked identically regardless of which
        evaluation path reaches it. *)

    type index_error =
      [ `Data_index_unexpected_here
      | `Gather_index_out_of_range of Gather_index_out_of_range.t
      | `Index_not_exact_in_float of int
      | `Index_overflow of Index_overflow.t
      | `Non_positive_divisor of int
      | `Unbound_reducer of Reduce_var.t ]
    (** Everything index evaluation can raise, and no more. [error] widens this
        in stage 4 with the load and intrinsic cases. The first two [Index_*]
        rows come from the library-private checked arithmetic; a public row may
        name a private module's variants without consumers needing to reach the
        module, so they are spelled out here rather than aliased.
        [`Data_index_unexpected_here] is the public [index]'s own "impossible
        resolver" tag -- a [Data] node cannot appear in an expression that never
        constructs one, so callers with no such node never see it in practice,
        but the public [index] entry point still has to name it. *)

    val pp_index_error : Format.formatter -> [< index_error ] -> unit

    val index :
      output:int Coord.t ->
      reducers:(Reduce_var.t -> int option) ->
      'role Index.t ->
      (int, index_error) Err.t
    (** Interprets an index at a concrete output coordinate. Every addition,
        scaling and division is bounds-checked BEFORE it is performed, so an
        intermediate cannot wrap back into range and sail past a later check --
        which is the whole point of the checked domain. Instantiates
        [eval_index] with a resolver that can never legitimately be called: a
        [Data] node cannot appear in a caller's input by construction unless
        that caller builds one itself, via [eval_index] directly. *)

    val eval_index :
      ([> index_error ] as 'e) Err.Escape.t ->
      widen:(index_error -> 'e) ->
      output:int Coord.t ->
      reducers:(Reduce_var.t -> int option) ->
      resolve_data:(Source.t -> int Coord.t -> (int64, 'e) Err.t) ->
      'role Index.t ->
      int
    (** [index]'s private evaluator, generalized over the caller's own error row
        ['e] (which must be at least as wide as [index_error]) and given an
        explicit resolver for [Data] sources. [~widen] lifts a bare
        [index_error] (from the ordinary checked-arithmetic/unbound-reducer
        arms) into ['e]; it has to be an explicit function rather than an inline
        coercion, since ['e] is still abstract at the point this function's own
        arithmetic-error arms are defined -- every real caller supplies it as
        [(fun e -> (e :> 'e))] at ITS OWN call site, where its concrete ['e] is
        already known. Two real callers instantiate this directly, each at their
        own row, with no extra error-mapping beyond that one [~widen] closure:
        [value] below (at ['e = error], via [Env.load_index]) and native's
        [Ground_eval] (at ['e = Ground_eval.error], via its own
        [resolve_data_source]). [resolve_data] is a RAW fetch -- it returns the
        stored [int64], not an already-validated [int] -- because
        normalization/bounds-checking (via [resolve_gather_index]) happens
        exactly once, uniformly, inside this function's own [Data] arm,
        regardless of which caller's resolver produced the raw value. *)

    val float_of_index : int -> (float, index_error) Err.t
    (** Carries an index into the value domain, failing rather than rounding
        silently. Both interpreters must go through this: native's [Ground_eval]
        converts independently in two places today, and a helper returning [int]
        does not make the conversion that follows it exact. *)

    type error =
      [ `Coord_out_of_range of Source.t * Axis.t * int * int Coord.t
      | `Data_source_wrong_format of string
      | index_error
      | Intrinsic.error
      | `Unbound_local of Local_var.t
      | `Unknown_source of Source.t ]
    (** [`Coord_out_of_range]/[`Unknown_source] are raised by the host's
        [Env.load], not by the language, which knows nothing about what a source
        is. [`Data_source_wrong_format] is raised the same way by
        [Env.load_index]: the bound tensor is not the [I64] format a [Data]
        source requires. Its payload is a bare format-name [string], not the
        host's own structured format type -- this library must not depend on
        [native], where that type lives. *)

    val pp_error : Format.formatter -> [< error ] -> unit

    module Env : sig
      type t = {
        load : Source.t -> int Coord.t -> (float, error) Err.t;
        load_index : Source.t -> int Coord.t -> (int64, error) Err.t;
      }
      (** The whole boundary between the language and its host: [Expr] supplies
          evaluated coordinates and consumes a working float (or, for
          [load_index], a raw stored integer), while storage format,
          quantization and tensor ownership stay on the other side. This is what
          keeps the library independent of [native], and what makes the
          evaluator testable against a plain map. *)
    end

    val value :
      ?local:(Local_var.t -> float option) ->
      ?local_at:(Local_var.t -> int -> float option) ->
      ?reducer:Reduce_var.t * int ->
      ?on_reduction:(unit -> unit) ->
      Env.t ->
      output:int Coord.t ->
      Value.t ->
      (float, error) Err.t
    (** The reference interpreter.

        [local_at] resolves a [Value.Local_at] read: the caller has already
        evaluated a vector local's whole body once per position (the same loop
        shape [Reduce]'s own fold uses, one iteration per key), and supplies the
        stored result at the requested position; [None] behaves as an unbound
        local, same as [local]. [reducer], when given, seeds evaluation with ONE
        reducer identity pre-bound to a concrete position -- what running a
        vector local's own body (which mentions its binder free, not under a
        [Reduce]) needs, mirroring how [Reduce]'s internal fold already binds
        its own [var] per iteration.

        [Select] evaluates only the selected branch. [Reduce] is the ordered
        half-open left fold, with the same seeds and the same association as the
        engine — a rewrite that reassociated it would change the answer, not
        just the shape. Max-pool advances value and index together under
        [Max_op.pool_better]; updating them separately is how they originally
        fell out of step.

        Internally it recurses through a private exception and converts once
        here, rather than allocating an [Ok] per node per output pixel: this
        runs inside the grounding loop the transform verifier drives for every
        walk config. *)
  end

  module Pp : sig
    val index :
      names:(Reduce_var.t -> string) ->
      Format.formatter ->
      'role Index.t ->
      unit
    (** [names] is supplied rather than derived from the opaque identity, so
        output cannot depend on allocation history. *)

    val value : Format.formatter -> Value.t -> unit

    val value_open :
      names:(Local_var.t -> string option) ->
      Format.formatter ->
      Value.t ->
      unit
    (** Assigns reducer display names r1, r2, ... in LEXICAL order, so two
        structurally identical formulas built by independent supplies print
        identically. The naming environment is SCOPED, so sibling scopes binding
        the same identity — well-scoped, and accepted by [Check] — still get
        distinct names, and a free reference prints as [?#n] rather than picking
        up a sibling's binder. Every form renders exactly as the representation
        it replaced — [Round_f32] is the one genuinely new node and has no old
        spelling to preserve — which is what keeps the migration's golden diff
        attributable to a single cause.

        For diagnostics and expect tests. Parsing it back is not supported. *)
  end
end
