open Loop_ir
module Common = Loop_native4d_common

let bits t =
  let (Tensor.Tensor tt) = t in
  let acc = ref [] in
  Vec6.iter tt.Tensor.shape (fun c ->
      acc := Int64.bits_of_float (Tensor.read_at t (Vec6.get c)) :: !acc);
  List.rev !acc

let map_of l id = List.assoc_opt id l

let with_dir f =
  let dir = Loop_c_exec.Proc.temp_dir "native4d_c_test" in
  Fun.protect
    ~finally:(fun () -> Loop_c_exec.Proc.remove_tree dir)
    (fun () -> f dir)

(* One line per case, and only when it is not "all outputs identical":
   anything printed is a defect or a refusal worth reading. *)
let run_case ?kernel ~label (c : Common.case) =
  let name = c.Common.name ^ label in
  let reference = Common.reference c in
  match
    Err.payload
      (Loop_native4d.build ~config:Loop_bundle_c.default_config c.Common.graph)
  with
  | Error e -> Fmt.pr "%s: bundle refused: %a@." name Loop_native4d.pp_error e
  | Ok b ->
      with_dir (fun dir ->
          match
            Err.payload
              (Loop_c_exec.Host.prepare ?kernel ~dir b
                 ~constants:(map_of c.Common.constants))
          with
          | Error e ->
              (* First line only: a compiler diagnostic quotes the whole
                 generated kernel. *)
              let message = Fmt.str "%a" Loop_c_exec.Host.pp_error e in
              Fmt.pr "%s: host refused: %s@." name
                (List.hd (String.split_on_char '\n' message))
          | Ok p -> (
              match
                Err.payload
                  (Loop_c_exec.Host.run ~poison:true p
                     ~bind:(map_of c.Common.inputs))
              with
              | Error e ->
                  Fmt.pr "%s: run failed: %a@." name Loop_c_exec.Host.pp_error e
              | Ok outs ->
                  let outputs = c.Common.graph.Graph_common.Graph.outputs in
                  if List.length outs <> List.length outputs then
                    Fmt.pr "%s: %d outputs, expected %d@." name
                      (List.length outs) (List.length outputs)
                  else
                    List.iter2
                      (fun id t ->
                        if bits t <> bits (Tensor_id.Map.find id reference) then
                          Fmt.pr "%s: output %a DIFFERS from Eval_direct4@."
                            name Tensor_id.pp id)
                      outputs outs))
