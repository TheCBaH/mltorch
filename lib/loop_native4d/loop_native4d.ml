open Native4d

let dialect : (Op.t, Eval_direct4.error) Loop_ir.Loop_dialect.t =
  {
    Loop_ir.Loop_dialect.fresh_synthetic_ids = Eval_direct4.fresh_synthetic_ids;
    is_region_authored = Region_computation4.is_region_authored;
    node_program =
      (fun ~limits g node -> Eval_symbolic4.node_program ~limits g node);
    region_group = Region_computation4.group;
    region_program = Region_computation4.program;
    storage_script =
      (fun ~retain config g ->
        Eval_storage4.storage_script ~retain config g
        |> Err.map_error (fun e -> (e :> Eval_direct4.error)));
  }

let build ?limits ?config ?plan g =
  Loop_ir.Loop_bundle.build_with ~dialect ?limits ?config ?plan g

let pp_error ppf e =
  Loop_ir.Loop_bundle.pp_generic_error Eval_direct4.pp_error ppf e
