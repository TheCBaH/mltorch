(* Symbolic evaluation of a native graph: turns it into a whole-graph stage DAG
   ([Stage_program.t]) through the stateless [Symbolic] adapter, so each node's
   per-pixel expression loads its operands' signatures and downstream stages
   reference upstream ones automatically. Structural groups are ignored by the
   flat stage DAG. See .ai/native_graph_design.md.

   Each stage body is built as an [Expr.Builder] computation and run on its own,
   so stages reuse reducer ordinals rather than sharing one namespace. That is
   the Expr contract, not an accident: identity is local to an expression, and a
   consumer that composes two stages freshens the inserted one. *)

open Graph_ir

type regionized = {
  program : Stage_program.t;
  candidates : (Region_program.t, Regionizer.error) Err.t Tensor_id.Map.t;
}

val run : graph -> Stage_program.t
val run_regionized : ?limits:Kernel.Limits.t -> graph -> regionized
