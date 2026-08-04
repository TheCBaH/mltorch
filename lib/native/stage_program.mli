(* The whole-graph symbolic form: a stage DAG (one stage per node) layered over the
   unchanged [Symbolic_expr]. Each stage's [body] is the node's per-pixel expression, whose
   [Load]s carry the producing edges' signatures — so a downstream stage references
   an upstream stage purely through the signature it loads (the "source = Input |
   Stage" distinction is recovered by membership, not an [Symbolic_expr] variant). [ground]
   evaluates the stages in topo order, feeding each result into the next stage's
   binding, which is what makes symbolic execution extend through the whole graph.
   See .ai/native_symbolic_language.md and .ai/native_graph_design.md. *)

open Graph_ir

module Stage : sig
  type t = {
    id : Tensor_id.t; (* the producing edge id (this stage's identity) *)
    sg : Tensor_sig.t; (* the stage's output signature *)
    body : Expr.Value.t;
        (* per-pixel expression; its Loads carry producers' sigs *)
  }
end

type t = {
  inputs : (Tensor_id.t * Tensor_sig.t) list;
      (* graph inputs (Load "sources") *)
  input_kinds : Input.kind Tensor_id.Map.t;
      (* same source classification as the originating graph *)
  consts : (Tensor_sig.t * float) list; (* synthetic constant-filled operands *)
  stages : Stage.t list; (* topo-ordered *)
  outputs : Tensor_id.t list;
}

val pp : Format.formatter -> t -> unit

(* Chain-ground the stages: [bind] supplies a tensor for each graph input id; the
   result maps every stage's edge id to its grounded tensor (the intermediates),
   so the graph outputs are looked up by id. *)
val ground :
  t -> bind:(Tensor_id.t -> Tensor.packed) -> Tensor.packed Tensor_id.Map.t
