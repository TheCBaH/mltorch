(* A validated, indexed, read-only view of a graph: the framework's trust
   boundary and the only thing a matcher reads.

   Construction VALIDATES, which is what makes the accessors unambiguous. Without
   it [def] returning [None] could mean either "this is a graph input" or "this
   operand has no definition at all", and every matcher built on that distinction
   would silently misbehave on a malformed graph. See
   .ai/native_transform_design.md §11.

   Unversioned, deliberately, and it stays that way: a matcher reads structure,
   and a tag here would buy it nothing. The version lives one layer up, in
   [Snapshot], which carries a view together with the id universes at that
   version — see .ai/native_transform_versioning.md. What made a versioned view
   itself unworkable was [of_graph : graph -> ('v t, _)] letting the caller pick
   ['v]; [Snapshot.create] answers with an existential instead.

   Version safety for matching does not depend on either. It comes from
   [Rewrite.plan], which stamps a recipe from the state it is given, plus the
   id-identity rule: an id is never reused within a pipeline, so an id present in
   two versions denotes the same tensor and a stale match is either caught by
   [apply] (the id is gone) or harmless (it still means what it meant). *)

open Graph_ir

type arity = { node : Node_id.t; expected : int; actual : int }
type sig_key = { key : Tensor_id.t; recorded : Tensor_id.t }

(* Parameterised over the dialect, with the Native specialization included at
   the end so no existing caller changes. See .ai/native4d_plan.md stage 4. *)
module Make (D : Dialect.S) : sig
  type node = D.op Graph_common.Node.t
  type graph = D.op Graph_common.Graph.t

  (* [`Graph_shape] WRAPS the dialect's shape row rather than inheriting it:
     [D.shape_error] is abstract, and OCaml can inherit only a type known to
     expand to a polymorphic-variant row. The tag is not [`Shape], because a
     dialect's own shape row may be inherited directly elsewhere and two
     same-named tags with different payloads cannot union.

     [`Invalid_sig] is new: [D.validate_sig] applied to every LIVE signature,
     which is the only place a dialect invariant that no operation implies —
     Native4D's four-axis rule on graph inputs and captured constants — can be
     enforced. *)
  type error =
    [ `Graph_shape of D.shape_error
    | `Invalid_sig of Tensor_id.t * D.shape_error
    | `Duplicate_group_id of Group_id.t
    | `Duplicate_group_item of Node_id.t
    | `Duplicate_node_id of Node_id.t
    | `Duplicate_tensor_def of Tensor_id.t
    | `Input_defined_by_node of Tensor_id.t
    | `Node_not_grouped of Node_id.t
    | `Not_topological of Node_id.t
    | `Output_arity of arity
    | `Output_shape_mismatch of Tensor_id.t
    | `Sig_key_mismatch of sig_key
    | `Unknown_group_item of Node_id.t
    | `Unknown_input_kind of Tensor_id.t
    | `Unknown_operand of Tensor_id.t
    | `Unknown_output of Tensor_id.t ]

  val pp_error : Format.formatter -> [< error ] -> unit

  type t

  (* Checks, in this order: unique node ids; unique group ids; the group tree is a
     partition of [nodes] (every node owned exactly once, no unknown or repeated
     item); every [tensors] entry keyed by its own [Tensor_sig.id]; inputs unique
     and not also node-defined; single assignment; every operand defined or
     declared an input; every graph output present; every [input_kinds] key a
     member of [inputs] — totality is NOT required, the map is sparse by design and
     [Graph_ir.input_kind] supplies the effective classification; [nodes] genuinely
     in topological order; and each node's output arity matching
     [Graph_shape.output_shape]. Shape inference makes this the one costly check,
     which is why it runs once per step rather than per query. *)
  val of_graph : graph -> (t, error) Core.result
  val graph : t -> graph

  (* Dataflow. [def] is [None] exactly for a graph input, never for a dangling
     operand — validation ruled that out. *)
  val node : t -> Node_id.t -> node option
  val def : t -> Tensor_id.t -> node option
  val uses : t -> Tensor_id.t -> node list
  val sig_of : t -> Tensor_id.t -> Tensor_sig.t option
  val is_graph_output : t -> Tensor_id.t -> bool
  val is_constant : t -> Tensor_id.t -> bool

  (* Position in [Graph.nodes], i.e. execution order. Used to decide "strictly
     earlier" for [Pattern.chain]'s progress rule and to order group items. *)
  val topo_index : t -> Node_id.t -> int option

  (* Structure. [common_group] is the nearest group whose subtree contains every
     given node — the root for an empty list or for nodes in unrelated groups — and
     is where a replacement spanning them is placed. *)
  val group_of : t -> Node_id.t -> Group_id.t
  val common_group : t -> Node_id.t list -> Group_id.t

  (* Stable topological sort of a node list, over the operand links WITHIN that
     list; ties keep the list's relative order, so a rewrite's output is
     deterministic. Fails with [`Cycle] naming a node on the cycle.

     Takes no view on purpose. Its caller is [Rewrite.apply], re-sorting a list
     that contains freshly inserted nodes whose outputs are absent from any view of
     the pre-rewrite graph; resolving producers through a view would make
     dependencies between new nodes invisible. *)
  val topo_sort : node list -> (node list, [> `Cycle of Node_id.t ]) Core.result
end

(* The Native specialization: the interface every existing caller already uses.
   Only the shape error moves, from an inherited row to [`Graph_shape]. *)
include module type of Make (Native_dialect)
