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

  (* [origin] classifies each of this graph's edges: which side it belongs to,
     or that both graphs define it identically, or that it is a graph input
     standing for a σ variable — see .ai/native_transform_verify.md §7.

     It replaces a [rename : Tensor_id.t -> Tensor_id.t] whose correctness rested
     on being the identity off the graph inputs, so that a cell's id worked both
     as a comparison key (renamed) and as a stage key (original). The two keyings
     are now separate types: cells are keyed by origin, stages by this graph's
     raw id.

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
    Stage_program.t ->
    origin:(Tensor_id.t -> Ground_expr.Origin.t) ->
    t

  (* Keyed by ORIGIN, so it answers questions about cells. Use [origin] to get
     there from an id named by a graph or a correspondence cluster. *)
  val shape_of : t -> Ground_expr.Origin.t -> Vec6.shape option
  val origin : t -> Tensor_id.t -> Ground_expr.Origin.t

  (* Whether reading this cell yields a value already representable in f32, and
     so whether [Ground_expr.normalise] may collapse its [Round]. Unknown cells
     answer [false]: refusing to collapse is the conservative direction. *)
  val stored_f32 : t -> Ground_expr.Cell.t -> bool
end

type error = [ `Unknown_edge of Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit

(* [id]'s value at [coord], with every [Load] left as a [Cell]. Wraps the
   stage's OWN body in [Round]: the stage's result is materialized too, and
   constant folding depends on that outermost rounding. A graph input has no
   stage and yields a bare [Cell] — this graph does not materialize it.

   The only failure is [id] belonging to neither, which is checked once here
   rather than inside the traversal. *)
val at :
  Env.t -> Tensor_id.t -> Vec6.coord -> (Ground_expr.t, error) Core.result

(* Replace every [Cell] that has a stage by [Round (that stage's body at the
   cell's coord)]. The [Round] lands exactly where the stored value was.

   [budget] bounds ONE round, and has to: a single substitution step is
   quadratic where a conv feeds a conv, so measuring only afterwards lets a term
   reach tens of millions of nodes first. Running out leaves the remaining cells
   unexpanded, which is sound rather than approximate — an unexpanded cell keeps
   [expandable] true, so the driver reports a budget verdict and no probe may
   run against a frontier that never reached the inputs. *)
val expand : budget:int -> Env.t -> Ground_expr.t -> Ground_expr.t

(* Whether any cell still has a stage below it. A probe may only run when this
   is [false] on both sides: cells left at a truncated frontier are internal
   stage results constrained by their producers, so assigning them
   independently can manufacture a counterexample no graph input can realise. *)
val expandable : Env.t -> Ground_expr.t -> bool

(* A cell whose coordinate falls outside its edge's extents, if any. Grounding
   cannot produce one from a well-formed graph — the ops clamp their windows —
   so this is a backstop against an op whose index arithmetic is wrong, checked
   once by the driver rather than at every constructed leaf. *)
val out_of_bounds : Env.t -> Ground_expr.t -> Ground_expr.Cell.t option
