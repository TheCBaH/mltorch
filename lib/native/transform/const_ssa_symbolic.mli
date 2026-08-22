(* Symbolic view of the deliberately small M1 Const-SSA language.  This keeps
   plan traversal out of [Ground_eval], which only knows how to ground one
   stage-program edge. *)

(* Expand a plan-backed graph value at [coord].  [None] means that no M1
   symbolic definition is available, so callers retain ordinary graph/payload
   grounding behaviour. *)
val ground :
  Constant_store.t -> Graph_ir.Tensor_id.t -> Vec6.coord -> Ground_expr.t option

(* A capture can appear more than once in a plan. Its storage format is usable
   only when every such leaf agrees; an absent or conflicting format deliberately
   answers [None] so rounding normalisation remains conservative. *)
val captured_fmt :
  Constant_store.t -> Const_ssa.Capture.t -> Payload.packed_fmt option
