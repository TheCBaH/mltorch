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

module Tensor_id : sig
  type t = private int

  val of_int : int -> t (* builder-internal allocation *)
  val to_int : t -> int
  val equal : t -> t -> bool
  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit

  module Map : Map.S with type key = t
end

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
  | Relu of { x : tensor_ref }
  | Add of { a : tensor_ref; b : tensor_ref }
  | Bmm of { input : tensor_ref; mat2 : tensor_ref }
  | Conv2d of {
      params : Conv.Conv2d.params;
      x : tensor_ref;
      weight : tensor_ref;
      bias : tensor_ref option;
    }
  | Permute of { perm : Permute.Permute.perm; x : tensor_ref }
  | Mean of { params : Reduce.Mean.params; x : tensor_ref }
  | Rms_norm of {
      params : Norm.RmsNorm.params;
      x : tensor_ref;
      weight : tensor_ref option;
    }
  | Linear of {
      params : Linear.Linear.params;
      x : tensor_ref;
      weight : tensor_ref;
      bias : tensor_ref option;
    }
  | Max_pool2d of { params : Pool.MaxPool2d.params; x : tensor_ref }
  | Avg_pool2d of { params : Pool.AvgPool2d.params; x : tensor_ref }
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
