(* Symbolic interpretation over the [Expr] library: a value is an
   [Expr.Value.t], an index is an ['role Expr.Index.t], and an input is a
   [Tensor_sig.t]. Running an op functor at [Make()] with the output coord bound
   to [out_vec] yields the op's per-pixel expression — used for codegen/fusion
   and the footprint analysis. See .ai/native_expr_refactoring_design.md.

   The phantom role is NOT erased: ['role index] is ['role Expr.Index.t], so
   position-versus-delta typing survives construction into the AST, where the
   representation this replaced collapsed it to one untyped node. That is only
   possible because [Dim.index]/[Dim.delta] are manifest aliases of
   [Expr.Role.Position.t]/[Delta.t] — a phantom parameter cannot be bridged by
   a conversion.

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
