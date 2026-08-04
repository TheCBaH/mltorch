(* Symbolic interpretation over the [Expr] library: a value is an
   [Expr.Value.t], an index is an ['role Expr.Index.t], and an input is a
   [Tensor_sig.t]. The replacement for [Symbolic], which builds the older
   [Symbolic_expr.t]; both exist until the cutover so the two can be
   differentially tested against each other. See
   .ai/native_expr_refactoring_design.md.

   Unlike [Symbolic], the phantom role is NOT erased here: ['role index] is
   ['role Expr.Index.t], so position-versus-delta typing survives construction
   into the AST. That is only possible because [Dim.index]/[Dim.delta] are
   manifest aliases of [Expr.Role.Position.t]/[Delta.t] — a phantom parameter
   cannot be bridged by a conversion.

   [input] stays [Tensor_sig.t] rather than becoming [Expr.Source.t]: it IS the
   adapter-owned source descriptor the design calls for, and keeping it means
   [Eval_op] and every op module are untouched. [load] projects the symbol. *)

(** [let module S = Make() in ...] gives a fresh instance with its own reducer
    supply, numbered independently of any other. *)
module Make () :
  Semantics.SEMANTICS
    with type t = Expr.Value.t
     and type 'role index = 'role Expr.Index.t
     and type input = Tensor_sig.t

val out_vec : Semantics.position Expr.Index.t Vec6.t
(** The output coordinate to pass as an op's [out], every axis an
    [Expr.Index.output] placeholder. A single constant, not built per call. *)
