(* Symbolic interpretation over the [Expr] library: a value is a CONSTRUCTION
   COMPUTATION producing an [Expr.Value.t], an index is an ['role Expr.Index.t],
   and an input is a [Tensor_sig.t]. Running an op functor with the output coord
   bound to [out_vec], then [Expr.Builder.run]ning the result, yields the op's
   per-pixel expression — used for codegen/fusion and the footprint analysis.
   See .ai/native_expr_refactoring_design.md.

   STATELESS, and deliberately not a generative functor. Reducer identities come
   from the [Expr.Builder] supply threaded through the computation, so there is
   no instance state to share, no allocation order to depend on, and no
   read-advance-write for concurrent callers to interleave. What this does NOT
   give is globally unique identities: two computations run from
   [Expr.Builder.initial] reuse ordinals, by design — see [Expr.Reduce_var],
   where identity is meaningful only within one expression. Composition safety
   comes from lexical scope and from [Expr.Rewrite.freshen] before insertion.

   A value being a computation means USING ONE TWICE RUNS IT TWICE. For a [load]
   or a [const] that costs nothing. For anything containing a reduction it builds
   the reduction twice, with distinct identities: alpha-equivalent, equal under
   [Expr.Value.equal], same answer, but duplicated work. Nothing here can enforce
   sharing — an op that needs it must bind the built value, not the computation.

   The phantom role is NOT erased: ['role index] is ['role Expr.Index.t], so
   position-versus-delta typing survives construction into the AST, where the
   representation this replaced collapsed it to one untyped node. That is only
   possible because [Dim.index]/[Dim.delta] are manifest aliases of
   [Expr.Role.Position.t]/[Delta.t] — a phantom parameter cannot be bridged by
   a conversion. Indices are not computations: they bind nothing, so they carry
   no supply.

   [input] stays [Tensor_sig.t] rather than becoming [Expr.Source.t]: it IS the
   adapter-owned source descriptor the design calls for, and keeping it means
   [Eval_op] and every op module are untouched. [load] projects the symbol. *)

include
  Semantics.SEMANTICS
    with type t = Expr.Value.t Expr.Builder.t
     and type 'role index = 'role Expr.Index.t
     and type input = Tensor_sig.t
     and type b = Expr.Bool.t Expr.Builder.t

val out_vec : Semantics.position Expr.Index.t Vec6.t
(** The output coordinate to pass as an op's [out], every axis an
    [Expr.Index.output] placeholder. A single constant, not built per call. *)
