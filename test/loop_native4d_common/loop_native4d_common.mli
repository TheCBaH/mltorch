type case = {
  name : string;
  graph : Native4d.Graph.graph;
  inputs : (Tensor_id.t * Tensor.packed) list;
  constants : (Tensor_id.t * Tensor.packed) list;
}

val per_op : unit -> case list
(** Every Native4D operation, one graph each ([Fixtures4.per_op]). *)

val lowered_case : string -> Graph_ir.graph -> case
(** A Native graph lowered to Native4D, with deterministic payloads. *)

val lowered : unit -> case list
(** Native fixtures lowered to Native4D: multi-node graphs, live index outputs,
    a multi-output Region (Lstm), an Sdpa with its mask omitted. *)

val deferred_constant : unit -> case
(** A convolution weight that is a deferred Const-SSA export after the canonical
    fold, materialized from a captured source. Fails if the lowering leaves
    nothing deferred. *)

val all : unit -> case list

val reference : case -> Tensor.packed Tensor_id.Map.t
(** [Eval_direct4.run]: the per-edge truth every backend is compared to. *)
