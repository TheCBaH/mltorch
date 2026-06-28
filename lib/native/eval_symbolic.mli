(* Symbolic evaluation of a native graph: turns it into a whole-graph stage DAG
   ([Stage_program.t]) using a single [Symbolic.Make ()] instance, so each node's
   per-pixel expression loads its operands' signatures and downstream stages
   reference upstream ones automatically. Subgraphs are inline-expanded (a flat
   DAG; nested-program form is future work). See .ai/native_graph_design.md. *)

open Graph_ir

val run : graph -> Stage_program.t
