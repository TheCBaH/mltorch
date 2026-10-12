(** The operation-level entry points {!Loop_bundle.build_with} needs from a
    graph dialect, so the bundle builder never names an operation type.

    [loop_ir] depends on [native], never on a further dialect: a dialect
    supplies its own record (Native4D's lives in [loop_native4d]) and the
    builder stays polymorphic in ['op]. Everything downstream of these functions
    — [Stage_program.t], [Region_program.t], [Fusion_plan.t], [Storage_script.t]
    — names no operation either.

    ['err] is the dialect's own error row, carried by {!field-storage_script}
    alone. Region construction reports the shared {!Region_computation.error},
    which every dialect already uses. *)

type ('op, 'err) t = {
  fresh_synthetic_ids :
    'op Graph_common.Graph.t ->
    (Region_computation.synthetic_role * Tensor_id.t) list;
      (** One id per synthetic-default role, disjoint from every graph tensor.
      *)
  is_region_authored : 'op -> bool;
  node_program :
    limits:Kernel.Limits.t ->
    'op Graph_common.Graph.t ->
    'op Graph_common.Node.t ->
    Stage_program.t;
      (** The one-node program over the node's own operands. *)
  region_group :
    limits:Kernel.Limits.t ->
    op:'op ->
    operand:(Tensor_id.t -> Tensor_sig.t option) ->
    (Region_group.t, Region_computation.error) Err.t;
  region_program :
    limits:Kernel.Limits.t ->
    op:'op ->
    output:Output_ordinal.t ->
    output_shape:Vec6.shape ->
    operand:(Tensor_id.t -> Tensor_sig.t option) ->
    fill:
      (Region_computation.synthetic_role -> float -> Vec6.shape -> Tensor_sig.t) ->
    (Region_program.t, Region_computation.error) Err.t;
  storage_script :
    retain:Release_schedule.Retain.t ->
    Storage_script.Config.t ->
    'op Graph_common.Graph.t ->
    (Storage_script.t, 'err) Err.t;
}

val native : (Graph_ir.op, Eval_direct.error) t
(** The Native dialect: exactly the calls {!Loop_bundle.build} made before the
    seam existed. *)
