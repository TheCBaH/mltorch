(* A policy's own schedule as the oracle: the same plan on the generic Machine
   IR route. For a policy whose sums are reassociated (relaxed), where no
   sequential oracle applies, a target route must equal this bitwise. *)

module M = Machine_model.Mir_model

let make ~pipeline =
  Model_test.Custom
    (fun g constants inputs ->
      let m =
        match M.prepare ~pipeline (Model_test.bundle g) with
        | Ok m -> m
        | Error _ -> failwith "the generic route refused the plan"
      in
      match
        Result.bind
          (M.Context.create m ~constants:(fun id -> List.assoc_opt id constants))
          (fun cx ->
            M.Context.run cx ~inputs:(fun id -> List.assoc_opt id inputs))
      with
      | Ok outs ->
          List.fold_left2
            (fun acc id t -> Graph_ir.Tensor_id.Map.add id t acc)
            Graph_ir.Tensor_id.Map.empty g.Graph_ir.Graph.outputs outs
      | Error s -> Fmt.failwith "%a" M.Stop.pp s)
