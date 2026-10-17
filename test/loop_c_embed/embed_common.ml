(* What the embedding tests share: the loader under test ([Embed_loader], which
   each suite supplies), the chain fixture and its reference. *)

open Graph_ir
open Loop_ir
module E = Loop_c_embed.Make (Embed_loader)

(* The whole-model backend run in this process through the dialect, against the
   per-node reference evaluator, compared bitwise. *)

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

let sig_of_edge (b : Loop_bundle.t) id =
  List.find_map
    (fun (inv : Loop_bundle.invocation) ->
      List.find_map
        (fun ((buf : Loop_buffer.t), edge) ->
          if Tensor_id.equal edge id then Some buf.Loop_buffer.sg else None)
        (List.combine inv.Loop_bundle.program.Loop_program.buffers
           inv.Loop_bundle.edges))
    b.Loop_bundle.invocations
  |> Option.get

let map_of l id = List.assoc_opt id l

let setup () =
  let g = Native_test.Graph_fixtures.chain () in
  let b =
    Err.or_raise ~pp_error:Loop_bundle.pp_error
      (Loop_bundle.build ~config:Loop_bundle_c.default_config g)
  in
  let constants =
    map_of
      (List.map
         (fun id -> (id, tensor_of ~salt:3 (sig_of_edge b id)))
         b.Loop_bundle.constants)
  in
  (g, b, constants)

let inputs b ~salt =
  List.map
    (fun id -> (id, tensor_of ~salt (sig_of_edge b id)))
    b.Loop_bundle.inputs

let reference g b ~constants ~salt =
  Err.or_raise ~pp_error:Eval_direct.pp_error
    (Eval_direct.run g
       ~constants:
         (List.filter_map
            (fun id -> Option.map (fun t -> (id, t)) (constants id))
            b.Loop_bundle.constants)
       ~inputs:(inputs b ~salt))

let check g b ctx ~constants ~salt ~poison =
  let expected = reference g b ~constants ~salt in
  match E.run ~poison ctx ~bind:(map_of (inputs b ~salt)) |> Err.payload with
  | Error e -> Fmt.pr "run failed: %a@." Loop_c_embed.pp_error e
  | Ok outs ->
      List.iter2
        (fun id t ->
          Fmt.pr "output t%d identical to the reference: %b@."
            (Tensor_id.to_int id)
            (bits t = bits (Tensor_id.Map.find id expected)))
        g.Graph.outputs outs

let prepare b = Err.or_raise ~pp_error:Loop_c_embed.pp_error (E.prepare b)

let context p ~constants =
  Err.or_raise ~pp_error:Loop_c_embed.pp_error (E.context p ~constants)
