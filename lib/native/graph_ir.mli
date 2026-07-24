(* The native graph IR: a holdable, evaluable, transformable representation of a
   native-op computation. Every edge (tensor) and node carries a stable id unique
   within the graph tree; each op names its operands as TYPED FIELDS (no positional
   list, no string keys), with optional operands as [tensor_ref option]; nested
   graphs are EMBEDDED by value in [Subgraph] (no registry / id indirection).

   [Node.t] / [Graph.t] each live in their own module (type named [t]), per the
   project rule (CLAUDE.md): record types get distinct namespaces so field labels
   are unique by construction (e.g. [Node.outputs] vs [Graph.outputs]). [op] is
   parametrised over the embedded-graph type so it can be defined once (not copied
   into the [module rec] group). See .ai/native_graph_design.md. *)

module Tensor_id = Tensor_id

module Node_id : sig
  type t = private int

  val of_int : int -> t
  val to_int : t -> int
  val pp : Format.formatter -> t -> unit
end

(* A reference to a produced edge. Edges are single-assignment, so a reference is
   just the producing edge's id. *)
type tensor_ref = Tensor_id.t

(* The op variant, parametrised over the embedded subgraph type ['g] so it needn't
   join the recursive module group; [op] below ties ['g] to [Graph.t]. (Named [gop]
   so the concrete [op = Graph.t gop] alias can keep the short name.) *)
type 'g gop =
  (* Constructors are kept in global alphabetical order so any op is easy to
     locate. Each non-[Subgraph] op carries its own payload record (params +
     operand refs), owned by that op's module; the shared serialise / dataflow /
     pp logic iterates a registry of those modules rather than matching every
     constructor (so adding an op no longer means editing parallel matches here).
     [Eval_op]/[Graph_shape] still match per op, since they need shape/semantics
     context the payload can't carry. *)
  | Add of Pointwise.Add.t
  | Avg_pool2d of Pool.AvgPool2d.t
  | Batch_norm of Norm.BatchNorm.t
  | Bmm of Matmul.Bmm.t
  | Conv2d of Conv.Conv2d.t
  | Conv2d_padding of Conv.Conv2d_padding.t
  | Convolution of Conv.Convolution.t
  | Discard of { x : tensor_ref }
    (* A sink: consumes one edge and produces NO output (its [Node.outputs] is
       empty). Used to route a dead op output — e.g. the argmax indices of
       [Max_pool2d_with_indices] — so the op keeps its full ATen arity while the
       edge is explicitly marked unused for a future pruning pass. Like
       [Subgraph], it is handled inline wherever the [op_registry] is folded. *)
  | Linear of Linear.Linear.t
  | Max_pool2d of Pool.MaxPool2d.t
  | Max_pool2d_with_indices of Pool.MaxPool2dWithIndices.t
  | Mean of Reduce.Mean.t
  | Mul of Pointwise.Mul.t
  | Permute of Permute.Permute.t
  | Relu of Pointwise.Relu.t
  | Reshape of Reshape.Reshape.t
  | Rms_norm of Norm.RmsNorm.t
  | Subgraph of { graph : 'g; args : tensor_ref list }
(* [args] map positionally to [graph.inputs] *)

module rec Node : sig
  type t = { id : Node_id.t; op : Graph.t gop; outputs : Tensor_id.t list }
  (* [outputs] is singleton for every current op; a list only to admit a future
     multi-output op (split/topk). *)
end

and Graph : sig
  type t = {
    name : string; (* unique within its parent (the subgraph's name) *)
    nodes : Node.t list; (* topo-ordered by construction *)
    tensors : Tensor_sig.t Tensor_id.Map.t; (* metadata for every edge id *)
    inputs : Tensor_id.t list; (* ordered = the graph's signature *)
    outputs : Tensor_id.t list;
  }
end

type op = Graph.t gop
type node = Node.t
type graph = Graph.t

(* Generic dataflow view — the only generic accessors needed (evaluation reads the
   typed fields directly). [operands] lists an op's input edges; [map_operands]
   rewrites them (for transform/remap). Both are one exhaustiveness-checked match. *)
val operands : op -> tensor_ref list
val map_operands : (tensor_ref -> tensor_ref) -> op -> op

(* Deterministic graph dump — used by tests and by bin/native_graph. It
   prints graph inputs, every node in topo order, op operands/parameters, and
   outputs; subgraphs are printed recursively under their call site. *)
val pp : Format.formatter -> graph -> unit
val op_jsont : op Jsont.t
val graph_jsont : graph Jsont.t
