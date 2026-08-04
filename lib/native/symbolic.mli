(* Symbolic interpretation: a value is an [Symbolic_expr.t] (a formula over the inputs), an
   index is an [Symbolic_expr.index_expr], and an input is a [Tensor_sig.t] (no data). Running an
   op functor at [Make()] with the output coord bound to [Index_var] yields the op's
   per-pixel expression — used for codegen/fusion and the footprint analysis.
   See .ai/native_symbolic_language.md §2.4.

   [Make] is a generative functor: each application [Make()] allocates a fresh,
   independent reduction-variable counter. No global state, no reset needed. *)

(** [let module S = Make() in ...] produces a fresh Symbolic instance with its
    own counter; reduction variables in [S] are numbered independently of any
    other instance. *)
module Make () :
  Semantics.SEMANTICS
    with type t = Symbolic_expr.t
     and type 'role index = Symbolic_expr.index_expr
     and type input = Tensor_sig.t

val out_vec : Symbolic_expr.index_expr Vec6.t
(** Pass [out_vec] as the [out] argument of an op's [pixel] to build its
    symbolic formula. A single constant, not built per call: unlike [Direct]
    (where [out] is a real per-pixel coordinate), every axis here is just an
    [Index_var] placeholder, the same 6-field value regardless of which node is
    being built. *)
