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
    body : Expr.Value.t;
        (** at an arbitrary output coordinate, WITHOUT [result] applied *)
    result : Result_conversion.t;
  }
end

module Output : sig
  type t = private { value : Tensor_id.t; sg : Tensor_sig.t }
  (** [sg] is DERIVED by [create] from the named value, so there is no signature
      for a caller to get wrong and no mismatch error to carry. *)
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
    val extent : int64
    val numel : int64
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
    (t, error) Core.result
  (** Custom limits may TIGHTEN, never widen: every field is checked against its
      [Hard] counterpart, and non-positive values are rejected. Widening matters
      as much as the runtime domain does — [Expr.Check.value] is itself
      recursive and bounded by the CONFIGURED depth, so an over-wide [max_depth]
      lets validation overflow before reaching the limit it was asked to
      enforce. *)
end

type t = private {
  inputs : Input.t list;
  values : Value.t list;  (** topologically ordered *)
  outputs : Output.t list;
  limits : Limits.t;
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
  type t = { at : Tensor_id.t; error : Expr.Check.error }
end

type error =
  [ `Duplicate_id of Tensor_id.t
  | `Signature_id_mismatch of Sig_mismatch.t
  | `Unresolved_source of Unresolved.t
  | `Forward_reference of Forward_ref.t
  | `Unknown_output of Tensor_id.t
  | `Unreachable_value of Tensor_id.t
  | `Too_many_values of int
  | `Too_many_inputs of int
  | `Too_many_outputs of int
  | `Dependency_too_deep of int
  | `Eval_too_deep of int
  | `Extent_too_large of Extent_bound.t
  | `Numel_too_large of Tensor_id.t
  | `Not_materializable of Format_rule.t
  | `Quant_contract of Tensor_id.t
  | `Body of Body_error.t ]
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
  (t, error) Core.result
(** Validates, in an order that matters: cheap arity guards, then the metered
    per-body budgets, then everything unmetered. [Expr.Fold]'s queries recurse
    over a whole tree, so running one before [Expr.Check.value] would exhaust
    the stack on exactly the oversized input the limit exists to reject. *)

val pp : Format.formatter -> t -> unit
