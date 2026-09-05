(* Turning one graph's stage program into [Ground_expr.t] at a coordinate.

   [at] stops at every [Load], so a cluster is checked locally first — one
   expression, not a whole-graph trace. [expand] pushes the frontier down one
   stage on demand. Splitting the two is what lets the driver deepen
   iteratively: try a local proof, go deeper only on failure.

   Expansion crosses ALL cells, not only the ones without a counterpart in the
   other graph. [reuse_permute_sub_order]'s map mentions only a deletion and a
   creation, yet proving its untouched output cluster needs the frontier to pass
   through an edge that IS present and corresponding in both graphs.

   Both [at]'s traversal and [expand] are TOTAL. [at] does not meter itself: it
   builds a single op's body at a single coordinate, bounded by the op (a conv's
   kernel x in-channels) rather than by the graph. [expand] does, because one
   round is quadratic where a conv feeds a conv. [out_of_bounds] is checked once
   by the driver at the end.

   See .ai/native_transform_verify.md. *)

module Env : sig
  type t

  (* [side] tags the cells grounding emits, so the two graphs' [t2]s are
     distinguishable — they share one numeric namespace. It is the ONLY thing
     the side is used for: every lookup here is about one graph and is keyed by
     that graph's raw id.

     Deciding that two side-qualified cells name one value is a claim about the
     MAP, and it lives in [Ground_expr.project], applied to a copy at comparison
     time. Keeping it out of here is what stops it reaching [stored_f32],
     [expand] or a stage lookup, none of which have any business believing it.
     See .ai/native_transform_verify.md §7.

     The program's synthetic [Stage_program.consts] — fresh-id, constant-filled
     stand-ins for absent optional operands, whose ids differ between the two
     graphs being compared — are always bound to their fill value. Without that
     the two sides could never match where one has a bias and the other does
     not.
     [constants] binds this graph's model constants (from [Rewrite.constants])
     to their payloads, turning those cells into [Const] leaves. It is
     PER-GRAPH, not shared: [Rewrite.apply] filters the payload map to live
     destination ids, so a constant a fold consumed and deleted survives only in
     the before-state. Binding narrows what a proof quantifies over — every
     input, for these constants, rather than every payload — which is why the
     driver tries without it first. *)
  val of_program :
    ?constants:Tensor.packed Tensor_id.Map.t ->
    ?constant_store:Constant_store.t ->
    Stage_program.t ->
    side:[ `Dst | `Src ] ->
    t

  (* Keyed by ORIGIN, so it answers questions about cells. A [Boundary] cell
     names no single edge and so answers [None]/[false] throughout: a projected
     term must never be normalised or expanded, and answering conservatively is
     what makes that a wrong answer rather than an unsound one. *)
  val shape_of : t -> Ground_expr.Origin.t -> Vec6.shape option

  (* This graph's [id] as a cell origin: [Src id] or [Dst id], never a variable. *)
  val origin : t -> Tensor_id.t -> Ground_expr.Origin.t

  (* USER data: a graph input of this program that is not a model constant. The
     σ hypothesis is granted on this and nothing else, so it is also what lets a
     member of the cluster under test project — see
     .ai/native_transform_verify.md §7a. *)
  val is_user_input : t -> Tensor_id.t -> bool

  (* Whether reading this cell yields a value already representable in f32, and
     so whether [Ground_expr.normalise] may collapse its [Round]. Unknown cells
     answer [false]: refusing to collapse is the conservative direction. *)
  val stored_f32 : t -> Ground_expr.Cell.t -> bool

  (* Whether this cell is a MODEL CONSTANT with no payload bound here. Sigma is
     about user data, so a constant does not get a shared variable and the two
     graphs' constants are distinct cells; without payloads there is then
     nothing to compare them by. The driver must report that rather than let a
     probe "separate" them, which would refute a pair that may hold identical
     bytes. Synthetic optional-operand fills are not constants in this sense —
     [leaf] resolves them to [Const] before a cell exists. *)
  val unbound_constant : t -> Ground_expr.Cell.t -> bool
end

(* [Expr.Eval.error] joins the row: grounding evaluates indices, and checked
   arithmetic can fail where it previously wrapped.
   [`Data_index_unresolved] is a [Data] source [resolve_data_source] cannot
   resolve to an exact value (anything but a directly-bound constant) -- it
   feeds [map_verify_check.ml]'s existing generic
   [Ground_eval.error -> Unproved] conversion unchanged. *)
type error =
  [ Expr.Eval.error
  | `Data_index_unresolved
  | `Region of Region_program.error
  | `Scan_at_unsupported
  | `Unknown_edge of Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit

(* [id]'s value at [coord], with every [Load] left as a [Cell]. Wraps the
   stage's symbolic view in [Round]: Region-authored stages derive that view
   from their structural program, while legacy Pixel stages use their body.
   The stage's result is materialized too, and constant folding depends on that
   outermost rounding. A graph input has no stage and yields a bare [Cell] —
   this graph does not materialize it.

   The stage is found by [id] DIRECTLY, so a root is always expanded and never
   replaced by a correspondence variable. That is what stops a cluster
   discharging itself: naming both its sides the same thing before either
   definition is looked at proves nothing about either. Projection applies to
   what the expanded root reads, not to the root.

   The only failure is [id] belonging to neither, which is checked once here
   rather than inside the traversal. *)
val at : Env.t -> Tensor_id.t -> Vec6.coord -> (Ground_expr.t, error) Err.t

(* Replace every [Cell] that has a stage AND no boundary variable by
   [Round (that stage's body at the cell's coord)]. The [Round] lands exactly
   where the stored value was.

   [boundary] is the local frontier. A cell it gives a variable for is a free
   variable of this obligation — its own cluster is a separate entry — so
   expanding through it would re-prove someone else's obligation inside this
   one, which is the cascade this design exists to stop.

   [budget] bounds ONE round, and has to: a single substitution step is
   quadratic where a conv feeds a conv, so measuring only afterwards lets a term
   reach tens of millions of nodes first. Running out leaves the remaining cells
   unexpanded, which is sound rather than approximate — an unexpanded cell keeps
   [expandable] true, so the driver reports a budget verdict and no probe may
   run against a truncated frontier. *)
val expand :
  boundary:(Ground_expr.Origin.t -> Cluster_var.t option) ->
  budget:int ->
  Env.t ->
  Ground_expr.t ->
  (Ground_expr.t, error) Err.t
(* Result-valued because expansion evaluates indices, and checked arithmetic can
   fail. The caller must convert a failure into a VERDICT about the cluster, the
   way [at]'s failure already is -- not widen its own error row and abandon the
   run. See [Map_verify.unproved_of_eval_error]. *)

val expandable :
  boundary:(Ground_expr.Origin.t -> Cluster_var.t option) ->
  Env.t ->
  Ground_expr.t ->
  bool

(* A cell whose coordinate falls outside its edge's extents, if any. Grounding
   cannot produce one from a well-formed graph — the ops clamp their windows —
   so this is a backstop against an op whose index arithmetic is wrong, checked
   once by the driver rather than at every constructed leaf. *)
val out_of_bounds : Env.t -> Ground_expr.t -> Ground_expr.Cell.t option
