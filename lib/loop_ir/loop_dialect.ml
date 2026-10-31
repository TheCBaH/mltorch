type ('op, 'err) t = {
  fresh_synthetic_ids :
    'op Graph_common.Graph.t ->
    (Region_computation.synthetic_role * Tensor_id.t) list;
  granted_only : 'op -> Tensor_id.t list option;
  is_region_authored : 'op -> bool;
  node_program :
    limits:Kernel.Limits.t ->
    'op Graph_common.Graph.t ->
    'op Graph_common.Node.t ->
    Stage_program.t;
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

let native : (Graph_ir.op, Eval_direct.error) t =
  {
    fresh_synthetic_ids = Eval_direct.fresh_synthetic_ids;
    granted_only =
      (function
      | Graph_ir.Attention_layer l ->
          Some (Attention_layer.Attention_layer.operands l)
      | _ -> None);
    is_region_authored = Region_computation.is_region_authored;
    node_program =
      (fun ~limits g node -> Eval_symbolic.node_program ~limits g node);
    region_group = Region_computation.group;
    region_program = Region_computation.program;
    storage_script =
      (fun ~retain config g -> Eval_direct.storage_script ~retain config g);
  }
