(* Symbolic interpretation: a value is an [Expr.t] (a formula over the inputs), an
   index is an [Expr.iexpr], and an input is a [Tensor_sig.t] (no data). Running an
   op functor at [Make()] with the output coord bound to [Ivar] yields the op's
   per-pixel expression — used for codegen/fusion and the footprint analysis.
   See .ai/native_symbolic_language.md §2.4.

   [Make] is a generative functor: each application [Make()] allocates a fresh,
   independent reduction-variable counter. No global state, no reset needed. *)

(** [let module S = Make() in ...] produces a fresh Symbolic instance with its
    own counter; reduction variables in [S] are numbered independently of any
    other instance. *)
module Make () :
  Semantics.SEMANTICS
    with type t = Expr.t
     and type 'role ix = Expr.iexpr
     and type input = Tensor_sig.t

val out_coord : Axis.t -> Expr.iexpr
(** Pass [out_coord] as the [out] argument of an op's [pixel] to build its
    symbolic formula. *)
