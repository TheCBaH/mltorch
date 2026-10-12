module Common = Loop_native4d_common

let bits t =
  let (Tensor.Tensor tt) = t in
  let acc = ref [] in
  Vec6.iter tt.Tensor.shape (fun c ->
      acc := Int64.bits_of_float (Tensor.read_at t (Vec6.get c)) :: !acc);
  List.rev !acc

let map_of l id = List.assoc_opt id l

(* One line per case, and only when it is not "all outputs identical":
   anything printed is a defect or a refusal worth reading. *)
let run_case ?kernel ~label (c : Common.case) =
  let name = c.Common.name ^ label in
  let reference = Common.reference c in
  match Err.payload (Loop_native4d.build c.Common.graph) with
  | Error e -> Fmt.pr "%s: bundle refused: %a@." name Loop_native4d.pp_error e
  | Ok b -> (
      match
        Err.payload
          (Loop_bundle_exec.prepare ?kernel b
             ~constants:(map_of c.Common.constants))
      with
      | Error e ->
          Fmt.pr "%s: prepare refused: %a@." name Loop_bundle_exec.pp_error e
      | Ok p -> (
          match
            Err.payload (Loop_bundle_exec.run p ~bind:(map_of c.Common.inputs))
          with
          | Error e ->
              Fmt.pr "%s: run failed: %a@." name Loop_bundle_exec.pp_error e
          | Ok out ->
              List.iter
                (fun id ->
                  match Tensor_id.Map.find_opt id out with
                  | None ->
                      Fmt.pr "%s: output %a missing@." name Tensor_id.pp id
                  | Some t ->
                      if bits t <> bits (Tensor_id.Map.find id reference) then
                        Fmt.pr "%s: output %a DIFFERS from Eval_direct4@." name
                          Tensor_id.pp id)
                c.Common.graph.Graph_common.Graph.outputs))

let%expect_test
    "every Native4D case: generated JavaScript equals Eval_direct4 bitwise" =
  List.iter (run_case ~label:"") (Common.all ());
  List.iter
    (run_case
       ~kernel:(Ssa_backends.js ~pipeline:Ssa_backends.Pipeline.Exact)
       ~label:" (ssa exact)")
    (Common.all ());
  [%expect {| |}]
