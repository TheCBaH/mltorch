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

val run : ?limits:Kernel.Limits.t -> graph -> Stage_program.t
(** [limits] is applied to every Region-authored stage's construction (via
    [Region_computation.program]); it defaults to [Kernel.Limits.default] rather
    than hardcoding it internally, so a caller with its own [Kernel.Limits.t]
    (e.g. [Region_kernel.of_graph]) can thread the same value through
    construction and later execution. *)

val node_program : ?limits:Kernel.Limits.t -> graph -> node -> Stage_program.t
(** A one-node [Stage_program.t], built by the same per-node construction [run]
    folds over the whole graph. [inputs] is the node's own operand signatures
    (every operand is a boundary input, since a one-node program has no earlier
    stage of its own to resolve one from); [outputs] is the node's full output
    list, unfiltered by ordinal. Ids [fill] mints only need to be disjoint
    within this one program, so they reuse [run]'s own base ([first_free_tid g])
    without colliding with it. *)
