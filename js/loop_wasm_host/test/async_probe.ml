(* The asynchronous preparation path: the continuation is called exactly once,
   from a later turn of the event loop, with a model that runs correctly. *)
open Graph_ir
open Loop_ir
module H = Loop_wasm_host

let tensor_of ~salt (sg : Tensor_sig.t) =
  let v c =
    float_of_int
      ((((Vec6.offset sg.Tensor_sig.shape c :> int) + salt) mod 7) - 3)
    /. 4.
  in
  Tensor.materialize sg.Tensor_sig.shape v

let bits t =
  let (Tensor.Tensor tt) = t in
  let acc = ref [] in
  Vec6.iter tt.Tensor.shape (fun c ->
      acc := Int32.bits_of_float (Tensor.read_at t (Vec6.get c)) :: !acc);
  List.rev !acc

let () =
  let g = Native_test.Graph_fixtures.chain () in
  let b =
    Err.or_raise ~pp_error:Loop_bundle.pp_error
      (Loop_bundle.build ~config:Loop_bundle_wasm.default_config g)
  in
  let sig_of id = Tensor_id.Map.find id g.Graph.tensors in
  let constants =
    List.map
      (fun id -> (id, tensor_of ~salt:3 (sig_of id)))
      b.Loop_bundle.constants
  in
  let calls = ref 0 in
  let synchronous = ref true in
  H.prepare_async b
    ~constants:(fun id -> List.assoc_opt id constants)
    (fun r ->
      incr calls;
      (* Output after the main program returned is not flushed at exit. *)
      Printf.printf "continuation called synchronously: %b\n%!" !synchronous;
      match Err.payload r with
      | Error e -> Format.printf "prepare failed: %a@." H.pp_error e
      | Ok m -> (
          let inputs =
            List.map
              (fun id -> (id, tensor_of ~salt:0 (sig_of id)))
              b.Loop_bundle.inputs
          in
          let reference =
            Err.or_raise ~pp_error:Eval_direct.pp_error
              (Eval_direct.run g ~constants ~inputs)
          in
          match
            Err.payload (H.run m ~bind:(fun id -> List.assoc_opt id inputs))
          with
          | Error e -> Format.printf "run failed: %a@." H.pp_error e
          | Ok outs ->
              List.iter2
                (fun id t ->
                  Printf.printf "output t%d identical to the reference: %b\n%!"
                    (Tensor_id.to_int id)
                    (bits t = bits (Tensor_id.Map.find id reference)))
                g.Graph.outputs outs;
              H.dispose m));
  synchronous := false;
  ignore calls
