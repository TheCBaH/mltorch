(** A generic bottom-up rewrite of every [Loop_index.t] reachable in a statement
    list: inside [For] bounds, [Load]/[Store]/[Array_set] coordinates, index
    temporaries, [Fail_if] predicates and failure payloads. [f] sees a node only
    after its own subexpressions have already been rewritten, so a caller's
    peephole rule needs to look only at its immediate (already-normalized)
    children -- both variable substitution ([Loop_opt_unit_loops]) and constant
    folding ([Loop_opt_fold]) are instances of this one traversal. *)

val stmts :
  f:(Loop_index.t -> Loop_index.t) -> Loop_stmt.t list -> Loop_stmt.t list

val index : f:(Loop_index.t -> Loop_index.t) -> Loop_index.t -> Loop_index.t
val pred : f:(Loop_index.t -> Loop_index.t) -> Loop_expr.pred -> Loop_expr.pred

val stmt : f:(Loop_index.t -> Loop_index.t) -> Loop_stmt.t -> Loop_stmt.t
(** Recurses into a [For]'s or [If]'s body with the same [f]: a pass whose [f]
    depends on the enclosing scope handles those two itself. *)
