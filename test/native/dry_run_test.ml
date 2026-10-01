(* [Eval_direct.dry_run]: the allocation script a run follows, emitted without
   computing anything. It shares the run's fold, so these tests check what
   sharing cannot: that a real run holds exactly what the script says, that the
   script's own byte fold agrees with the release schedule's, and that
   [Alloc_script.first_difference] finds the event two scripts part at. *)

open Graph_ir

let only_empty = Release_schedule.Retain.Only Tensor_id.Set.empty
let pp_error ppf e = Eval_direct.pp_error ppf (Err.Error.kind e)

let dry ?(retain = only_empty) g =
  match Eval_direct.dry_run ~retain g with
  | Ok s -> s
  | Error e -> Fmt.failwith "dry run: %a" pp_error e

let fixture name = (List.assoc name Graph_fixtures.all) ()

let%expect_test "scripts of small graphs" =
  let show name retain =
    Fmt.pr "@[<v>-- %s@,%a@]@." name Alloc_script.pp
      (dry ~retain (fixture name))
  in
  show "residual" only_empty;
  show "residual" Release_schedule.Retain.All;
  show "multi_output" only_empty;
  [%expect
    {|
    -- residual
    node n0
    alloc t1 float32 16 bytes
    node n1
    alloc t2 float32 16 bytes
    free t1
    node n2
    alloc t3 float32 16 bytes (outside the arena)
    free t2
    -- residual
    node n0
    alloc t1 float32 16 bytes (outside the arena)
    node n1
    alloc t2 float32 16 bytes (outside the arena)
    node n2
    alloc t3 float32 16 bytes (outside the arena)
    -- multi_output
    node n0
    alloc t1 float32 32 bytes
    node n2
    alloc t3 float32 32 bytes (outside the arena)
    free t1 |}]

(* ---- the script's bytes equal the release schedule's ------------------- *)

let is_index_output (op : op) output =
  Output_ordinal.equal output Output_ordinal.one
  &&
  match op with
  | Adaptive_max_pool2d_with_indices _ | Max_dim _ | Max_pool2d_with_indices _
    ->
      true
  | _ -> false

let is_sink = function Discard _ -> true | _ -> false

let input_bytes (g : graph) =
  List.fold_left
    (fun acc id ->
      let sg = Tensor_id.Map.find id g.Graph.tensors in
      let numel =
        match
          Err.payload
            (Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel
               sg.Tensor_sig.shape)
        with
        | Ok n -> n
        | Error _ -> assert false
      in
      Int64.add acc
        (Int64.mul numel
           (Int64.of_int (Payload.packed_cell_bytes sg.Tensor_sig.fmt))))
    0L g.Graph.inputs

let%expect_test "the script's peak plus the inputs equals peak_bytes" =
  List.iter
    (fun (name, build) ->
      let g = build () in
      List.iter
        (fun (label, retain) ->
          let sched =
            Release_schedule.schedule ~operands:Graph_ir.operands ~is_sink
              ~retain g
          in
          let reference =
            match
              Err.payload (Release_schedule.peak_bytes ~is_index_output g sched)
            with
            | Ok n -> n
            | Error _ -> assert false
          in
          let mine =
            match Err.payload (Alloc_script.peak_bytes (dry ~retain g)) with
            | Ok n -> Int64.add n (input_bytes g)
            | Error _ -> assert false
          in
          if not (Int64.equal mine reference) then
            Fmt.pr "%s (%s): script %Ld, schedule %Ld@." name label mine
              reference)
        [ ("only", only_empty); ("all", Release_schedule.Retain.All) ])
    Graph_fixtures.all;
  Fmt.pr "%d fixtures agree@." (List.length Graph_fixtures.all);
  [%expect {| 45 fixtures agree |}]

(* ---- a real run holds what the dry run says ---------------------------- *)

let tensor_of_sig (sg : Tensor_sig.t) =
  let v c =
    float_of_int (((Vec6.offset sg.Tensor_sig.shape c :> int) mod 7) - 3) /. 4.
  in
  match sg.Tensor_sig.fmt with
  | Payload.Fmt Payload.I64 ->
      Tensor.materialize_i64 sg.Tensor_sig.shape (fun c ->
          Int64.of_int ((Vec6.offset sg.Tensor_sig.shape c :> int) mod 2))
  | Payload.Fmt Payload.Bool ->
      Tensor.materialize_bool sg.Tensor_sig.shape (fun c -> v c > 0.)
  | _ -> Tensor.materialize sg.Tensor_sig.shape v

let traced ~retain (g : graph) =
  let bound kind =
    List.filter_map
      (fun id ->
        if input_kind g id = kind then
          Some (id, tensor_of_sig (Tensor_id.Map.find id g.Graph.tensors))
        else None)
      g.Graph.inputs
  in
  let events = ref [] in
  match
    Eval_direct.run ~retain
      ~trace:(fun e -> events := e :: !events)
      ~constants:(bound Input.Constant) g ~inputs:(bound Input.Input)
  with
  | Ok _ -> List.rev !events
  | Error e -> Fmt.failwith "run: %a" pp_error e

let%expect_test "a real run's trace equals the dry-run script" =
  List.iter
    (fun (name, build) ->
      let g = build () in
      List.iter
        (fun (label, retain) ->
          match
            Alloc_script.first_difference (dry ~retain g) (traced ~retain g)
          with
          | None -> ()
          | Some d ->
              Fmt.pr "%s (%s): differs at %a: %a / %a@." name label
                Alloc_script.Position.pp d.Alloc_script.Difference.position
                Fmt.(option ~none:(any "end") Alloc_script.Event.pp)
                d.left
                Fmt.(option ~none:(any "end") Alloc_script.Event.pp)
                d.right)
        [ ("only", only_empty); ("all", Release_schedule.Retain.All) ])
    Graph_fixtures.all;
  Fmt.pr "%d fixtures agree@." (List.length Graph_fixtures.all);
  [%expect {| 45 fixtures agree |}]

(* ---- first_difference -------------------------------------------------- *)

let show_difference a b =
  match Alloc_script.first_difference a b with
  | None -> Fmt.pr "equal@."
  | Some { Alloc_script.Difference.position; left; right } ->
      let pp = Fmt.(option ~none:(any "end") Alloc_script.Event.pp) in
      Fmt.pr "at %a: %a / %a@." Alloc_script.Position.pp position pp left pp
        right

let%expect_test "first_difference on dry runs" =
  let residual = fixture "residual" in
  show_difference (dry residual) (dry residual);
  (* Under [All] nothing is released, so an intermediate's [Alloc] is
     ineligible; the difference is there, before any [Free]. *)
  show_difference (dry residual)
    (dry ~retain:Release_schedule.Retain.All residual);
  (* Two independent nodes swapped: the first [Node] marker differs. *)
  let two =
    Graph_fixtures.build2 "two_chains"
      Graph_builder.(
        let* x = input ~shape:(Graph_fixtures.s1c 4) () in
        let* y = input ~shape:(Graph_fixtures.s1c 4) () in
        let* a = relu x in
        let* b = relu y in
        let* a2 = relu a in
        let* b2 = relu b in
        return (a2, b2))
  in
  let swapped =
    match two.Graph.nodes with
    | n0 :: n1 :: rest -> { two with Graph.nodes = n1 :: n0 :: rest }
    | _ -> assert false
  in
  show_difference (dry two) (dry swapped);
  [%expect
    {|
    equal
    at @1: alloc t1 float32 16 bytes / alloc t1 float32 16 bytes (outside the arena)
    at @0: node n0 / node n1 |}]

(* Scripts differing only in a [Free] cannot come from two dry runs: a changed
   release changes the edge's eligibility, so its earlier [Alloc] already
   differs (the [Only]/[All] case above). [first_difference] is a pure function
   over scripts, so these are built by hand. *)
let%expect_test "first_difference on hand-built scripts" =
  let sg i =
    Tensor_sig.create ~id:(Tensor_id.of_int i) ~name:""
      ~shape:(Graph_fixtures.s1c 4) ~fmt:(Payload.Fmt Payload.F32) ()
  in
  let alloc i =
    match
      Err.payload
        (Alloc_script.alloc
           ~released:
             (Tensor_id.Set.of_list [ Tensor_id.of_int 1; Tensor_id.of_int 2 ])
           (sg i))
    with
    | Ok a -> Alloc_script.Event.Alloc a
    | Error _ -> assert false
  in
  let node i = Alloc_script.Event.Node (Node_id.of_int i) in
  let free i = Alloc_script.Event.Free (Tensor_id.of_int i) in
  let full = [ node 0; alloc 1; node 1; alloc 2; free 1; free 2 ] in
  show_difference full full;
  show_difference full [ node 0; alloc 1; node 1; alloc 2; free 1 ];
  show_difference full [ node 0; alloc 1; node 1; alloc 2; free 2; free 1 ];
  show_difference full [ node 0; alloc 1; free 1; node 1; alloc 2; free 2 ];
  show_difference [] full;
  [%expect
    {|
    equal
    at @5: free t2 / end
    at @4: free t1 / free t2
    at @2: node n1 / free t1
    at @0: end / node n0 |}]
