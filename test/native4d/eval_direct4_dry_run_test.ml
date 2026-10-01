(* [Eval_direct4.dry_run]: the allocation script a run follows, computed from
   the Native4D graph's own nodes, shapes and release schedule, without running
   anything. The twin of test/native's dry_run_test.ml. *)

open Native4d

let only_empty = Release_schedule.Retain.Only Tensor_id.Set.empty

let traced ~retain (g : Graph.graph) ~inputs ~constants =
  let events = ref [] in
  let trace e = events := e :: !events in
  Result.map
    (fun _ -> List.rev !events)
    (Eval_direct4.run ~trace ~retain ~constants g ~inputs)

(* One line per graph: the run's own script equals the dry run's. *)
let agrees name (g : Graph.graph) ~inputs ~constants =
  let pp_err ppf e = Eval_direct4.pp_error ppf (Err.Error.kind e) in
  List.iter
    (fun (label, retain) ->
      match
        (Eval_direct4.dry_run ~retain g, traced ~retain g ~inputs ~constants)
      with
      | Ok dry, Ok run -> (
          match Alloc_script.first_difference dry run with
          | None -> ()
          | Some _ -> Fmt.pr "%s (%s): SCRIPTS DIFFER@." name label)
      | Error e, _ | _, Error e -> Fmt.pr "%s (%s): %a@." name label pp_err e)
    [ ("all", Release_schedule.Retain.All); ("only", only_empty) ]

let%expect_test "every per-op graph: the run follows its dry run" =
  List.iter
    (fun (name, g, inputs, constants) -> agrees name g ~inputs ~constants)
    (Fixtures4.per_op ());
  [%expect {| |}]

(* x -> relu -> sqrt -> relu: the middle edges are released, the output is
   not, so only the intermediates are eligible. *)
let%expect_test "a chain's script" =
  let g =
    Fixtures4.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:Fixtures4.flat () in
       let* a = relu x in
       let* b = sqrt a in
       relu b)
  in
  (match Err.payload (Eval_direct4.dry_run ~retain:only_empty g) with
  | Ok s -> Fmt.pr "%a@." Alloc_script.pp s
  | Error e -> Fmt.pr "error: %a@." Eval_direct4.pp_error e);
  [%expect
    {|
    node n0
    alloc t1 float32 16 bytes
    node n1
    alloc t2 float32 16 bytes
    free t1
    node n2
    alloc t3 float32 16 bytes (outside the arena)
    free t2 |}]

let tensor_of_sig (sg : Tensor_sig.t) =
  match sg.Tensor_sig.fmt with
  | Payload.Fmt Payload.I64 ->
      Tensor.materialize_i64 sg.Tensor_sig.shape (fun _ -> 0L)
  | _ -> Tensor.materialize sg.Tensor_sig.shape (fun _ -> 0.25)

let allocs script =
  List.length
    (List.filter
       (function Alloc_script.Event.Alloc _ -> true | _ -> false)
       script)

(* Native fixtures lowered to Native4D: the destination's own script, which a
   legalization can make differ from the source's, and the run follows it. *)
let lowered name native =
  let source =
    match Err.payload (Eval_direct.dry_run ~retain:only_empty native) with
    | Ok s -> Fmt.str "%d" (allocs s)
    | Error _ -> "error"
  in
  let line =
    Lower_fixtures.described native ~render:(fun dst ->
        let bound kind =
          List.filter_map
            (fun id ->
              if Graph.input_kind dst id = kind then
                Some
                  ( id,
                    tensor_of_sig
                      (Tensor_id.Map.find id dst.Graph_common.Graph.tensors) )
              else None)
            dst.Graph_common.Graph.inputs
        in
        agrees name dst
          ~inputs:(bound Graph_ir.Input.Input)
          ~constants:(bound Graph_ir.Input.Constant);
        match Err.payload (Eval_direct4.dry_run ~retain:only_empty dst) with
        | Ok s -> Fmt.str "native %s allocs, native4d %d" source (allocs s)
        | Error e -> Fmt.str "%a" Eval_direct4.pp_error e)
  in
  Fmt.pr "%s: %s@." name line

(* Lowering removes the clone, so its edge is allocated in Native only. *)
let clone () =
  Lower_fixtures.build "clone"
    (let open Graph_builder in
     let* x = input ~shape:(Fixtures.nhwc ~n:1 ~h:2 ~w:2 ~c:3) () in
     let* c = clone x in
     relu c)

let%expect_test "lowered multi-node graphs: each dialect's own script" =
  List.iter
    (fun (name, g) -> lowered name g)
    Fixtures.
      [
        ("clone", clone ());
        ("adaptive_maxpool_indices_live", adaptive_maxpool_indices_live ());
        ("bmm_batch", bmm_batch 1 ());
        ("expand", expand ());
        ("group_norm_tiny", group_norm_tiny ());
        ("layer_norm_tiny", layer_norm_tiny ());
        ("linear_layer", linear_layer ());
        ("lstm_states_live", lstm_states_live ());
        ("maxpool_indices_live", maxpool_indices_live ());
        ("repeat", repeat ());
        ("sdpa", sdpa 1 ());
        ("split_with_sizes_w_batch2", split_with_sizes_w_batch2 ());
        ("unbind_n", unbind_n ());
      ];
  [%expect
    {|
    clone: native 2 allocs, native4d 1
    adaptive_maxpool_indices_live: native 4 allocs, native4d 4
    bmm_batch: native 1 allocs, native4d 1
    expand: native 1 allocs, native4d 1
    group_norm_tiny: native 1 allocs, native4d 1
    layer_norm_tiny: native 1 allocs, native4d 1
    linear_layer: native 1 allocs, native4d 1
    lstm_states_live: native 5 allocs, native4d 5
    maxpool_indices_live: native 4 allocs, native4d 4
    repeat: native 1 allocs, native4d 1
    sdpa: native 1 allocs, native4d 1
    split_with_sizes_w_batch2: native 2 allocs, native4d 2
    unbind_n: native 2 allocs, native4d 2 |}]
