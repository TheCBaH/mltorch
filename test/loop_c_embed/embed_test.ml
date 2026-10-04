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

let%expect_test "one loaded unit, called repeatedly, matches the reference" =
  let g, b, constants = setup () in
  let p = prepare b in
  let ctx = context p ~constants in
  check g b ctx ~constants ~salt:0 ~poison:false;
  check g b ctx ~constants ~salt:2 ~poison:true;
  check g b ctx ~constants ~salt:5 ~poison:true;
  E.close_context ctx;
  E.close p;
  [%expect
    {|
    output t9 identical to the reference: true
    output t9 identical to the reference: true
    output t9 identical to the reference: true |}]

let%expect_test "contexts share the loaded unit and keep their own memory" =
  let g, b, constants = setup () in
  let p = prepare b in
  let a = context p ~constants and c = context p ~constants in
  check g b a ~constants ~salt:1 ~poison:true;
  check g b c ~constants ~salt:4 ~poison:true;
  check g b a ~constants ~salt:1 ~poison:true;
  E.close_context a;
  (match Err.payload (E.run a ~bind:(map_of (inputs b ~salt:1))) with
  | Error `Closed -> print_endline "closed context refused"
  | _ -> print_endline "unexpected");
  check g b c ~constants ~salt:4 ~poison:false;
  E.close_context c;
  E.close p;
  [%expect
    {|
    output t9 identical to the reference: true
    output t9 identical to the reference: true
    output t9 identical to the reference: true
    closed context refused
    output t9 identical to the reference: true |}]

let%expect_test "the unit has no directive and calls only declared host symbols"
    =
  let _, b, _ = setup () in
  let p = prepare b in
  let text = E.source p in
  let directive =
    List.find_opt
      (fun l -> String.length l > 0 && l.[0] = '#')
      (String.split_on_char '\n' text)
  in
  Fmt.pr "%s@." (Option.value directive ~default:"no directive");
  E.close p;
  [%expect {| no directive |}]

let%expect_test "a missing input and a failing invocation are typed" =
  let _, b0, constants = setup () in
  let p = prepare b0 in
  let ctx = context p ~constants in
  (match Err.payload (E.run ctx ~bind:(fun _ -> None)) with
  | Error (`Missing_input id) ->
      Fmt.pr "missing input t%d@." (Tensor_id.to_int id)
  | _ -> print_endline "unexpected");
  E.close_context ctx;
  E.close p;
  let failing =
    C_host_fixtures.poisoned_bundle b0 ~at:1 Loop_failure.I64_division_by_zero
  in
  let p = prepare failing in
  let ctx = context p ~constants in
  (match Err.payload (E.run ctx ~bind:(map_of (inputs b0 ~salt:0))) with
  | Error (`Inference_failed (i, e)) ->
      Fmt.pr "invocation %d, %a@." i Loop_interp.pp_error e
  | Error e -> Fmt.pr "%a@." Loop_c_embed.pp_error e
  | Ok _ -> print_endline "no failure");
  E.close_context ctx;
  E.close p;
  [%expect {|
    missing input t0
    invocation 1, I64 division by zero |}]
