(* The semantic Kernel IR: a closed, validated computation over unchanged
   [Expr.Value.t] stage bodies. See .ai/native_kernel_dsl_design.md.

   A kernel owns ordered boundary inputs, topologically ordered logical values,
   and ordered outputs. It does NOT own placement: whether a value is stored or
   recomputed is a [Fusion_plan] decision over unchanged semantics, so there is
   no [Materialized | Virtual] anywhere in here.

   Construction is result-valued and total in its checks. [Stage_program.t] is a
   public record and this is advertised as a standalone hand-built boundary, so
   nothing below is a trusted precondition. *)

module Binding : sig
  (* Where a boundary input's data comes from. [Filled] is a synthetic operand
     the adapter invented for an omitted optional argument; it is an ordinary
     input with a locally owned value, never an evaluator special case. *)
  type t = Caller | Captured_constant | Filled of float

  val pp : Format.formatter -> t -> unit
end

module Input : sig
  type t = { id : Tensor_id.t; sg : Tensor_sig.t; binding : Binding.t }
end

module Result_conversion : sig
  (* The conversion at a logical value's boundary, independent of whether a
     schedule stores it. Removing it where a buffer used to be is a semantic
     change, not an optimisation. *)
  type t = Round_f32

  val apply : t -> Expr.Value.t -> Expr.Value.t
  (** The ONE place the round is expressed. Every consumer — interpreter and
      elaborator — goes through here, so [Expr.Eval]'s existing [Round_f32] case
      stays the only rounding code and the two cannot disagree. *)

  val name : t -> string
end

module Value : sig
  type t = {
    id : Tensor_id.t;
    sg : Tensor_sig.t;  (** [sg.id] must equal [id] *)
    computation : Region_program.t;
        (** at an arbitrary output coordinate, WITHOUT [result] applied *)
    result : Result_conversion.t;
  }
end

module Output : sig
  type t = private { value : Tensor_id.t; sg : Tensor_sig.t }
  (** [sg] is DERIVED by [create] from the named value, so there is no signature
      for a caller to get wrong and no mismatch error to carry. *)
end

module Use : sig
  (* An INTERNAL producer->consumer edge: BOTH endpoints are logical values.
     A load of a boundary input is a dependency of the body but never a [Use.t]
     — otherwise a forged {producer = some input; consumer = some value} would
     name a real load, pass a dependency check, and reach an elaborator with no
     producer body or result conversion to substitute, while not being unknown
     either. *)
  type t = { producer : Tensor_id.t; consumer : Tensor_id.t }

  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit

  module Set : Set.S with type elt = t
end

module Limits : sig
  (* Four independent budget dimensions. Bounding one leaves the others open,
     and the fourth exists only because the first two compose:

       per-expression  max_size, max_depth        one [Expr.Value.t]
       value DAG       max_values, max_dep_depth  the logical-value graph
       interface       max_inputs, max_outputs    the public arity
       combined        derived eval_depth         the recursive value_at stack

     See the design record for why the combined one is not implied by the other
     two, and for the node measurements behind [Hard]. *)
  type t = private {
    max_size : int;
    max_depth : int;
    max_values : int;
    max_dep_depth : int;
    max_inputs : int;
    max_outputs : int;
    max_extent : int64;
    max_numel : int64;
    max_local_slots : int;
        (** Region trace/scalar/vector storage: total slot count across a
            computation's locals, [(steps+1)*width] per trace local. *)
    max_scan_state : int;
        (** Peak live scan state during one evaluation -- see
            [Expr.Scan_limits.max_state]. *)
    max_scan_updates_per_key : int64;
        (** Recurrence iterations admitted for one Region key -- see
            [Expr.Scan_limits.max_updates]. *)
    max_scan_updates_total : int64;
        (** Summed [keys * per_key] across a Kernel's logical values. Enforced
            only at [create]: there is no whole-graph choke point upstream of
            Direct or Stage-ground execution, so this bound is Kernel-scoped,
            not universal. *)
  }

  module Invalid : sig
    type t = { name : string; value : int64 }
  end

  type error = [ `Invalid_limit of Invalid.t ]

  val pp_error : Format.formatter -> [< error ] -> unit

  module Hard : sig
    (* Non-negotiable cross-backend ceilings. [depth] and [eval_depth] are
       EMPIRICAL stack limits under node, which has the tighter stack, not
       census maxima; test/native/depth_probe.ml pins both on both backends and
       is what keeps them honest. The rest bound memory and time. *)
    val size : int
    val depth : int
    val values : int
    val dep_depth : int
    val inputs : int
    val outputs : int
    val eval_depth : int

    (* Producer transitions [Kernel_eval.value_at] may nest before it reports
        rather than overflows. Measured under node against a REAL recursive
        chain, not a flat expression: a transition costs far more frames than an
        expression level, so [eval_depth] — which counts levels — does not bound
        it. This is enforced at runtime, where the recursion is, rather than by
        rejecting a deep DAG at construction: the buffer-based [run] never
        recurses at all, so a long chain is perfectly executable and only the
       on-demand path is limited. *)
    val eval_recursion : int
    val extent : int64
    val numel : int64

    (* Memory- and array-length-bound, not stack-bound -- policy ceilings with
       deliberate headroom over the scan design record's censuses, not
       empirically discovered frontiers. *)
    val max_local_slots : int
    val max_scan_state : int
    val max_scan_updates_per_key : int64
    val max_scan_updates_total : int64
  end

  val default : t
  (** Census-derived, with headroom over the largest body/DAG/interface sizes
      observed across the fixture graphs and resnet18-shaped bodies. *)

  val create :
    max_size:int ->
    max_depth:int ->
    max_values:int ->
    max_dep_depth:int ->
    max_inputs:int ->
    max_outputs:int ->
    max_extent:int64 ->
    max_numel:int64 ->
    max_local_slots:int ->
    max_scan_state:int ->
    max_scan_updates_per_key:int64 ->
    max_scan_updates_total:int64 ->
    (t, error) Err.t
  (** Custom limits may TIGHTEN, never widen: every field is checked against its
      [Hard] counterpart, and non-positive values are rejected. Widening matters
      as much as the runtime domain does — [Expr.Check.value] is itself
      recursive and bounded by the CONFIGURED depth, so an over-wide [max_depth]
      lets validation overflow before reaching the limit it was asked to
      enforce. *)

  val scan_limits : t -> Expr.Scan_limits.t
  (** [max_scan_state]/[max_scan_updates_per_key] narrowed to what [Expr] needs.
      A pure accessor, never fallible: both fields already cleared the same
      [Hard] ceilings [Expr.Scan_limits.create] enforces. *)
end

type t = private {
  inputs : Input.t list;
  values : Value.t list;  (** topologically ordered *)
  outputs : Output.t list;
  limits : Limits.t;
  by_id : Value.t Tensor_id.Map.t;
      (** Derived index over [values], maintained by [create]. Read it through
          [value], whose lookup is therefore O(log values) — a balanced map, not
          a hash table, so not constant time. The point is that the whole-list
          scan is gone: it was multiplied by every caller resolving endpoints
          per candidate. *)
}

(* Each error payload gets its own module: three of them would otherwise share
   an [at]/[id] label and trip warning 30, which CLAUDE.md forbids silencing. *)

module Sig_mismatch : sig
  type t = { record : Tensor_id.t; sg : Tensor_id.t }
end

module Unresolved : sig
  type t = { at : Tensor_id.t; source : Expr.Source.t }
end

module Forward_ref : sig
  type t = { at : Tensor_id.t; depends_on : Tensor_id.t }
  (** [depends_on], not [loads]: validation covers every source, so a forward
      reference can come from an intrinsic descriptor with no load involved. *)
end

module Extent_bound : sig
  type t = { id : Tensor_id.t; axis : Expr.Axis.t; extent : int64 }
end

module Format_rule : sig
  type role = Filled_input | Stored_value
  type t = { id : Tensor_id.t; role : role; fmt : Payload.packed_fmt }
end

module Body_error : sig
  type t = { at : Tensor_id.t; error : Region_program.error }
end

type error =
  [ `Body of Body_error.t
  | `Dependency_too_deep of int
  | `Duplicate_id of Tensor_id.t
  | `Eval_too_deep of int
  | `Extent_too_large of Extent_bound.t
  | `Forward_reference of Forward_ref.t
  | `Not_materializable of Format_rule.t
  | `Numel_too_large of Tensor_id.t
  | `Quant_contract of Tensor_id.t
  | `Scan_updates_total_over_limit of int64
  | `Signature_id_mismatch of Sig_mismatch.t
  | `Too_many_inputs of int
  | `Too_many_outputs of int
  | `Too_many_values of int
  | `Unknown_output of Tensor_id.t
  | `Unreachable_value of Tensor_id.t
  | `Unresolved_source of Unresolved.t ]
(** [`Too_*] and [`*_too_deep] carry the LIMIT, not the measure, for the reason
    [Expr.Check] does: reporting the actual figure would mean measuring the
    whole input the limit exists to avoid walking. *)

val pp_error : Format.formatter -> [< error ] -> unit

val create :
  ?limits:Limits.t ->
  inputs:Input.t list ->
  values:Value.t list ->
  outputs:Tensor_id.t list ->
  unit ->
  (t, error) Err.t
(** Validates, in an order that matters: cheap arity guards, then the metered
    per-body budgets, then everything unmetered. [Expr.Fold]'s queries recurse
    over a whole tree, so running one before [Expr.Check.value] would exhaust
    the stack on exactly the oversized input the limit exists to reject. *)

val pp : Format.formatter -> t -> unit
val value : t -> Tensor_id.t -> Value.t option
val pixel_expression : Value.t -> Expr.Value.t option

val over_limit : int -> 'a list -> bool
(** Does the list hold more than [limit] cells? Stops one cell past the limit
    instead of walking to the end, so a guard over an untrusted list does not
    itself become the unbounded work it was added to prevent. A negative limit
    is exceeded by every list, the empty one included. *)

val over_limit_2 : int -> 'a list -> 'b list -> bool
(** The same, over two lists sharing one budget. Continues the second from the
    first's remainder rather than adding two counts: a sum of counts is an
    unchecked [int] aggregate, and under js_of_ocaml a wrapped negative passes a
    [> limit] test. *)

val uses : t -> Use.Set.t
(** Every internal dependency edge, from [Expr.Fold.sources] — which includes a
    source reached only through an intrinsic descriptor — restricted to sources
    resolving to a logical value. For DAG analysis. *)

val load_uses : t -> Use.Set.t
(** The same restriction over [Expr.Fold.loads]: ordinary-load edges between two
    logical values. These are the SUBSTITUTABLE ones, and what elaboration and
    the planner validate against. A producer named only by a max-pool descriptor
    is in [uses] and not here; validating elaboration against [uses] would
    accept such a pair and then leave the body silently unchanged. *)
