(* [Eval_direct.run ?arena]: the same answers as a release-only run, from slots
   the plan placed. Each check here has a mutation in the tracker that turns it
   red; a check that never failed would prove nothing. *)

open Graph_ir

let only_empty = Release_schedule.Retain.Only Tensor_id.Set.empty
let pp_error ppf e = Eval_direct.pp_error ppf (Err.Error.kind e)

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

let bound (g : graph) kind =
  List.filter_map
    (fun id ->
      if input_kind g id = kind then
        Some (id, tensor_of_sig (Tensor_id.Map.find id g.Graph.tensors))
      else None)
    g.Graph.inputs

let run ?arena ?node_executor ?region_executor ?region_group_executor
    ?(retain = only_empty) (g : graph) =
  Eval_direct.run ?arena ?node_executor ?region_executor ?region_group_executor
    ~retain ~constants:(bound g Input.Constant) g ~inputs:(bound g Input.Input)

let acquire ?poison ?(retain = only_empty) g =
  match
    Err.payload
      (Arena_run.acquire ?poison ~retain ~admission:Arena.Admission.Best_effort
         g)
  with
  | Ok (Arena_run.Arena a) -> a
  | Ok (Arena_run.Release_only _) -> Fmt.failwith "arena declined"
  | Error _ -> Fmt.failwith "acquire failed"

let fixture name = (List.assoc name Graph_fixtures.all) ()

let scratch_of g (node : node) output =
  let oid = List.nth node.Node.outputs (output : Output_ordinal.t :> int) in
  match
    Err.payload (Tensor.create_of_sig (Tensor_id.Map.find oid g.Graph.tensors))
  with
  | Ok t -> t
  | Error _ -> assert false

(* Outputs bit-identical, or the same error. *)
let compare_runs name (g : graph) a b =
  match (a, b) with
  | Ok a, Ok b ->
      if
        List.for_all
          (fun id ->
            Tensor.equal_bits (Tensor_id.Map.find id a)
              (Tensor_id.Map.find id b))
          g.Graph.outputs
      then None
      else Some (name ^ ": OUTPUTS DIFFER")
  | Error a, Error b ->
      let a = Fmt.str "%a" pp_error a and b = Fmt.str "%a" pp_error b in
      if String.equal a b then None else Some (name ^ ": ERRORS DIFFER")
  | Ok _, Error _ | Error _, Ok _ -> Some (name ^ ": ONE FAILED")

(* A graph that plans but fails part way: the second relu's output is declared
   F16 (so it lives in a 16-bit pool), and a Long cast from F16 is rejected when
   the cast runs, not when the graph is planned. *)
let fails_midway () =
  let g =
    Graph_fixtures.build "fails_midway"
      Graph_builder.(
        let* x = input ~shape:(Graph_fixtures.s1c 4) () in
        let* a = relu x in
        let* b = relu a in
        to_copy Pointwise.To_copy.Long b)
  in
  let b = List.hd (List.nth g.Graph.nodes 1).Node.outputs in
  Graph_fixtures.with_fmt b (Payload.Fmt Payload.F16) g

let%expect_test "arena runs equal release runs on every fixture" =
  let bad = ref 0 in
  List.iter
    (fun (name, build) ->
      let g = build () in
      let arena = acquire g in
      match compare_runs name g (run g) (run ~arena g) with
      | None -> ()
      | Some msg ->
          incr bad;
          Fmt.pr "%s@." msg)
    Graph_fixtures.all;
  (* An error mid-graph is the same error at the same node. *)
  let g = fails_midway () in
  (match compare_runs "fails_midway" g (run g) (run ~arena:(acquire g) g) with
  | None -> ()
  | Some msg ->
      incr bad;
      Fmt.pr "%s@." msg);
  Fmt.pr "release run: %a@."
    Fmt.(result ~ok:(any "ran") ~error:pp_error)
    (run g);
  Fmt.pr "%d fixtures, %d differ@." (List.length Graph_fixtures.all + 1) !bad;
  [%expect
    {|
    release run: to_copy: Long target has no exact I64 output for a f16 source
    46 fixtures, 0 differ |}]

(* Two different poisons must give the same bits: every cell a run reads was
   written by the run. *)
let%expect_test "paired poison-on-acquire runs agree" =
  let bad = ref 0 in
  List.iter
    (fun (name, build) ->
      let g = build () in
      let a = run ~arena:(acquire ~poison:Arena.Poison.A g) g
      and b = run ~arena:(acquire ~poison:Arena.Poison.B g) g in
      match compare_runs name g a b with
      | None -> ()
      | Some msg ->
          incr bad;
          Fmt.pr "%s@." msg)
    Graph_fixtures.all;
  Fmt.pr "%d fixtures, %d differ@." (List.length Graph_fixtures.all) !bad;
  [%expect {| 45 fixtures, 0 differ |}]

(* A writer that leaves the last cell of its destination alone, into a slot an
   earlier edge of the same run filled completely. Poison written each time a
   slot is acquired shows the missed cell; poison written once, at the start of
   the run, would leave the earlier edge's value there, the same under both
   poisons. *)
let%expect_test
    "poison on acquire catches a writer that skips a cell of a reused slot" =
  let g =
    Graph_fixtures.build "relu_chain"
      Graph_builder.(
        let* x = input ~shape:(Graph_fixtures.s1c 4) () in
        let* a = relu x in
        let* b = relu a in
        let* c = relu b in
        relu c)
  in
  let third = (List.nth g.Graph.nodes 2).Node.id in
  let skips_last_cell =
    {
      Node_executor.run =
        (fun g node ~output ~out_shape:_ ~operands:_ ~dst ~direct ->
          if not (Node_id.equal node.Node.id third) then direct ~dst
          else
            (* Compute in full elsewhere, then write every cell but the last. *)
            let full = scratch_of g node output in
            let r = direct ~dst:full in
            (match Err.payload r with
            | Ok (Tensor.Tensor f) ->
                let shape = f.Tensor.shape in
                let n = (Vec6.numel shape :> int) and i = ref 0 in
                Vec6.iter shape (fun c ->
                    incr i;
                    if !i < n then Tensor.set_float dst c (Tensor.read full c))
            | Error _ -> ());
            Result.map (fun _ -> dst) r);
    }
  in
  let arena = acquire g in
  let slot n =
    match
      Arena_plan.slot (Arena.plan arena)
        (List.hd (List.nth g.Graph.nodes n).Node.outputs)
    with
    | Some s -> Int64.to_int s.Arena_plan.Slot.offset
    | None -> -1
  in
  Fmt.pr "slots of the first three relus: %d %d %d@." (slot 0) (slot 1) (slot 2);
  let outputs poison =
    match
      Err.payload
        (run ~arena:(acquire ~poison g) ~node_executor:skips_last_cell g)
    with
    | Ok env -> Tensor_id.Map.find (List.hd g.Graph.outputs) env
    | Error _ -> assert false
  in
  Fmt.pr "paired runs agree: %b@."
    (Tensor.equal_bits (outputs Arena.Poison.A) (outputs Arena.Poison.B));
  [%expect
    {|
    slots of the first three relus: 4 0 4
    paired runs agree: false |}]

(* ---- no arena memory escapes the run -------------------------------------- *)

let snapshot t =
  let (Tensor.Tensor r) = t in
  Tensor.materialize r.Tensor.shape (fun c -> Tensor.read t c)

let%expect_test "nothing a run returns aliases the arena" =
  let bad = ref 0 in
  List.iter
    (fun (name, build) ->
      let g = build () in
      (* Retain an intermediate as well as the outputs. *)
      let inner =
        List.find_map
          (fun (n : node) ->
            List.find_opt
              (fun id -> not (List.mem id g.Graph.outputs))
              n.Node.outputs)
          g.Graph.nodes
      in
      let retain =
        Release_schedule.Retain.Only
          (match inner with
          | Some id -> Tensor_id.Set.singleton id
          | None -> Tensor_id.Set.empty)
      in
      let arena = acquire ~retain g in
      match Err.payload (run ~arena ~retain g) with
      | Error _ -> ()
      | Ok env ->
          let before = Tensor_id.Map.map snapshot env in
          Arena.fill_for_test arena Arena.Poison.A;
          Tensor_id.Map.iter
            (fun id t ->
              if not (Tensor.equal_bits t (Tensor_id.Map.find id before)) then (
                incr bad;
                Fmt.pr "%s: %a changed when the pools were overwritten@." name
                  Tensor_id.pp id))
            env)
    Graph_fixtures.all;
  Fmt.pr "%d fixtures, %d aliased@." (List.length Graph_fixtures.all) !bad;
  [%expect {| 45 fixtures, 0 aliased |}]

(* ---- one run at a time ------------------------------------------------------ *)

let%expect_test "a busy arena is refused, and the flag clears on every exit" =
  let g = fixture "residual" in
  let arena = acquire g in
  let inner = ref None in
  let node_executor =
    {
      Node_executor.run =
        (fun _ _ ~output:_ ~out_shape:_ ~operands:_ ~dst ~direct ->
          if Option.is_none !inner then
            inner :=
              Some
                (match Err.payload (run ~arena g) with
                | Ok _ -> "ran"
                | Error e -> Fmt.str "%a" Eval_direct.pp_error e);
          direct ~dst);
    }
  in
  ignore (run ~arena ~node_executor g);
  Fmt.pr "reentrant: %s@." (Option.value !inner ~default:"never called");
  (* After a run that failed part way, the arena is free again. *)
  let bad = fails_midway () in
  let arena_bad = acquire bad in
  (match Err.payload (run ~arena:arena_bad bad) with
  | Ok _ -> Fmt.pr "failing run: ok@."
  | Error e -> Fmt.pr "failing run: %a@." Eval_direct.pp_error e);
  Fmt.pr "after: %s@."
    (match Err.payload (run ~arena:arena_bad bad) with
    | Ok _ -> "ran"
    | Error `Arena_busy -> "busy"
    | Error _ -> "free, failed again");
  [%expect
    {|
    reentrant: arena: already in use by a run
    failing run: to_copy: Long target has no exact I64 output for a f16 source
    after: free, failed again |}]

(* ---- a plan runs only on the run it was built from -------------------------- *)

(* A node executor that counts how many nodes were handed to it. *)
let counting () =
  let calls = ref 0 in
  ( calls,
    {
      Node_executor.run =
        (fun _ _ ~output:_ ~out_shape:_ ~operands:_ ~dst ~direct ->
          incr calls;
          direct ~dst);
    } )

let two_chains () =
  Graph_fixtures.build2 "two_chains"
    Graph_builder.(
      let* x = input ~shape:(Graph_fixtures.s1c 4) () in
      let* y = input ~shape:(Graph_fixtures.s1c 4) () in
      let* a = relu x in
      let* b = relu y in
      let* a2 = relu a in
      let* b2 = relu b in
      return (a2, b2))

let%expect_test
    "a plan is refused for any run but its own, before any node runs" =
  let residual = fixture "residual" in
  let arena = acquire residual in
  let attempt label ?retain g =
    let calls, node_executor = counting () in
    let result =
      match retain with
      | Some retain -> run ~arena ~node_executor ~retain g
      | None ->
          (* [retain] omitted: the effective retain is [All]. *)
          Eval_direct.run ~arena ~node_executor
            ~constants:(bound g Input.Constant) g ~inputs:(bound g Input.Input)
    in
    Fmt.pr "%s: %a; %d executor calls@." label
      Fmt.(result ~ok:(any "RAN") ~error:pp_error)
      result !calls
  in
  attempt "the same run" ~retain:only_empty residual;
  (* Ids, order, shapes and formats unchanged; the last node reads an earlier
     edge instead. *)
  let reads_earlier =
    let nodes =
      List.map
        (fun (n : node) ->
          match n.Node.op with
          | Add ({ Pointwise.Bin.a; _ } as r) ->
              { n with Node.op = Add { r with Pointwise.Bin.b = a } }
          | Relu { Pointwise.Relu.x } when Tensor_id.to_int x = 1 -> n
          | _ -> n)
        residual.Graph.nodes
    in
    let a = List.hd (List.hd residual.Graph.nodes).Node.outputs in
    let nodes =
      List.map
        (fun (n : node) ->
          match n.Node.op with
          | Add r -> { n with Node.op = Add { r with Pointwise.Bin.b = a } }
          | _ -> n)
        nodes
    in
    { residual with Graph.nodes }
  in
  attempt "a later node reads another edge" ~retain:only_empty reads_earlier;
  let out_a = List.hd (List.hd residual.Graph.nodes).Node.outputs in
  attempt "other graph outputs" ~retain:only_empty
    { residual with Graph.outputs = [ out_a ] };
  attempt "retain changed"
    ~retain:(Release_schedule.Retain.Only (Tensor_id.Set.singleton out_a))
    residual;
  attempt "retain omitted" residual;
  let two = two_chains () in
  let swapped =
    match two.Graph.nodes with
    | n0 :: n1 :: rest -> { two with Graph.nodes = n1 :: n0 :: rest }
    | _ -> assert false
  in
  let arena_two = acquire two in
  let calls, node_executor = counting () in
  Fmt.pr "swapped nodes: %a; %d executor calls@."
    Fmt.(result ~ok:(any "RAN") ~error:pp_error)
    (run ~arena:arena_two ~node_executor swapped)
    !calls;
  attempt "another graph" ~retain:only_empty (fixture "chain");
  [%expect
    {|
    the same run: RAN; 3 executor calls
    a later node reads another edge: arena: the plan was built for a different run: at event @4 the plan has free t1, this run has free t2; 0 executor calls
    other graph outputs: arena: the plan was built for a different run: at event @1 the plan has alloc t1 float32 16 bytes, this run has alloc t1 float32 16 bytes (outside the arena); 0 executor calls
    retain changed: arena: the plan was built for a different run: at event @1 the plan has alloc t1 float32 16 bytes, this run has alloc t1 float32 16 bytes (outside the arena); 0 executor calls
    retain omitted: arena: the plan was built for a different run: at event @1 the plan has alloc t1 float32 16 bytes, this run has alloc t1 float32 16 bytes (outside the arena); 0 executor calls
    swapped nodes: arena: the plan was built for a different run: at event @0 the plan has node n0, this run has node n1; 0 executor calls
    another graph: arena: the plan was built for a different run: at event @1 the plan has alloc t1 float32 16 bytes, this run has alloc t7 float32 108 bytes; 0 executor calls |}]

(* ---- an executor that answers in storage of its own -------------------------- *)

let%expect_test
    "an executor that ignores its destination still gives release's answers" =
  let g = fixture "residual" in
  let release = run g in
  let fresh =
    {
      Node_executor.run =
        (fun g node ~output ~out_shape:_ ~operands:_ ~dst:_ ~direct ->
          direct ~dst:(scratch_of g node output));
    }
  in
  (* Answers with its own operand whenever that is the right answer (a relu of
     a non-negative edge), so the tensor it returns is an arena slot too. *)
  let operand_returning =
    {
      Node_executor.run =
        (fun g node ~output ~out_shape:_ ~operands ~dst:_ ~direct ->
          let reference = direct ~dst:(scratch_of g node output) in
          match (Err.payload reference, Graph_ir.operands node.Node.op) with
          | Ok r, x :: _ -> (
              match Tensor_id.Map.find_opt x operands with
              | Some t when Tensor.equal_bits t r -> Ok t
              | _ -> reference)
          | _ -> reference);
    }
  in
  List.iter
    (fun (label, node_executor) ->
      let arena = acquire g in
      let result = run ~arena ~node_executor g in
      Fmt.pr "%s: %s, %Ld copies@." label
        (match compare_runs label g release result with
        | None -> "equal to release"
        | Some m -> m)
        (Arena.copies arena).Arena.Copies.count)
    [ ("fresh", fresh); ("operand", operand_returning) ];
  (* The default executors write into the slots: nothing to copy. *)
  let arena = acquire g in
  ignore (run ~arena g);
  Fmt.pr "default: %Ld copies@." (Arena.copies arena).Arena.Copies.count;
  [%expect
    {|
    fresh: equal to release, 2 copies
    operand: equal to release, 2 copies
    default: 0 copies |}]

(* Quantized edges are never arena-backed; their neighbours are. *)
let%expect_test "quantized unbind and split next to arena-backed edges" =
  let shape = Vec6.shape ~n:2 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3 in
  let sizes = [ Dim.extent 1; Dim.extent 2 ] in
  let q = Quant.per_tensor ~scale:0.25 ~zero_point:2 in
  let g =
    Graph_fixtures.buildn "quantized_beside_f32"
      Graph_builder.(
        let* xq = input ~shape ~fmt:(Payload.Fmt Payload.I8) ~quant:q () in
        let* f = input ~shape:(Graph_fixtures.s1c 4) () in
        let* slices = unbind { Split.Unbind.axis = Axis.N } xq in
        let* parts =
          split_with_sizes { Split.Split_with_sizes.axis = Axis.C; sizes } xq
        in
        let* a = relu f in
        let* b = relu a in
        return (slices @ parts @ [ b ]))
  in
  let inputs =
    List.map
      (fun id ->
        let sg = Tensor_id.Map.find id g.Graph.tensors in
        match sg.Tensor_sig.fmt with
        | Payload.Fmt Payload.I8 ->
            let n = (Vec6.numel shape :> int) in
            let data = Bigarray.(Array1.create int8_signed c_layout n) in
            for i = 0 to n - 1 do
              data.{i} <- (i * 9) - 20
            done;
            ( id,
              Tensor.Tensor
                {
                  shape;
                  payload =
                    { Payload.fmt = Payload.I8; quant = Payload.Quant q; data };
                } )
        | _ -> (id, tensor_of_sig sg))
      g.Graph.inputs
  in
  let go ?arena () = Eval_direct.run ?arena ~retain:only_empty g ~inputs in
  let arena = acquire g in
  let raw_of (Tensor.Tensor t) =
    let n = (Vec6.numel t.Tensor.shape :> int) in
    let cells : type e b q. (e, b, q) Tensor.t -> string =
     fun t ->
      match t.Tensor.payload.Payload.fmt with
      | Payload.I8 ->
          String.concat ","
            (List.init n (fun i ->
                 string_of_int t.Tensor.payload.Payload.data.{i}))
      | _ -> "-"
    in
    ( cells t,
      match t.Tensor.payload.Payload.quant with
      | Payload.Quant q -> Some q
      | Payload.No_quant -> None )
  in
  (match (Err.payload (go ()), Err.payload (go ~arena ())) with
  | Ok release, Ok arena_run ->
      List.iter
        (fun id ->
          let a = Tensor_id.Map.find id release
          and b = Tensor_id.Map.find id arena_run in
          let ca, qa = raw_of a and cb, qb = raw_of b in
          Fmt.pr "%a: %s@." Tensor_id.pp id
            (if
               String.equal ca cb
               && Option.equal Quant.equal qa qb
               && Tensor.equal_bits a b
             then "same cells and quantization"
             else "DIFFERS"))
        g.Graph.outputs
  | _ -> Fmt.pr "a run failed@.");
  let plan = Arena.plan arena in
  Fmt.pr "arena-backed edges: %d@." (List.length (Arena_plan.slots plan));
  [%expect
    {|
    t2: same cells and quantization
    t3: same cells and quantization
    t4: same cells and quantization
    t5: same cells and quantization
    t7: same cells and quantization
    arena-backed edges: 1 |}]

(* ---- admission --------------------------------------------------------------- *)

let%expect_test "admission" =
  let g = fixture "residual" in
  let footprint =
    match Err.payload (Arena.footprint (Arena.plan (acquire g))) with
    | Ok n -> n
    | Error _ -> assert false
  in
  Fmt.pr "footprint %Ld bytes@." footprint;
  let attempt label ?limits ~admission () =
    let calls, node_executor = counting () in
    let result =
      Arena_run.with_arena ?limits ~retain:only_empty ~admission g (fun arena ->
          Eval_direct.run ?arena ~node_executor ~retain:only_empty g
            ~inputs:(bound g Input.Input)
          |> Err.map_error (fun e -> (e :> Arena_run.error)))
    in
    Fmt.pr "%s: %s; %d executor calls@." label
      (match Err.payload result with
      | Ok (_, Arena_run.Outcome.Used r) ->
          Fmt.str "arena, pool %Ld bytes, outside %Ld"
            r.Arena_run.Report.pool_bytes r.Arena_run.Report.out_of_arena_bytes
      | Ok (_, Arena_run.Outcome.Declined reason) ->
          Fmt.str "release-only (%a)" Arena_run.pp_error reason
      | Error e -> Fmt.str "REJECTED: %a" Arena_run.pp_error e)
      !calls
  in
  attempt "required, enough" ~admission:(Arena.Admission.Required footprint) ();
  attempt "required, one byte short"
    ~admission:(Arena.Admission.Required (Int64.pred footprint))
    ();
  let d = Kernel.Limits.default in
  let tight =
    match
      Err.payload
        (Kernel.Limits.create ~max_size:d.max_size ~max_depth:d.max_depth
           ~max_values:d.max_values ~max_dep_depth:d.max_dep_depth
           ~max_inputs:d.max_inputs ~max_outputs:d.max_outputs
           ~max_extent:d.max_extent ~max_numel:d.max_numel ~max_bytes:8L
           ~max_local_slots:d.max_local_slots ~max_scan_state:d.max_scan_state
           ~max_scan_updates_per_key:d.max_scan_updates_per_key
           ~max_scan_updates_total:d.max_scan_updates_total)
    with
    | Ok l -> l
    | Error _ -> assert false
  in
  attempt "required, over a ceiling" ~limits:tight
    ~admission:(Arena.Admission.Required Int64.max_int) ();
  attempt "best effort, over a ceiling" ~limits:tight
    ~admission:Arena.Admission.Best_effort ();
  [%expect
    {|
    footprint 48 bytes
    required, enough: arena, pool 32 bytes, outside 16; 3 executor calls
    required, one byte short: REJECTED: arena: the run needs 48 bytes but the budget is 47; 0 executor calls
    required, over a ceiling: REJECTED: arena: the float32 pool needs 8 cells (32 bytes), over 8 bytes; 0 executor calls
    best effort, over a ceiling: release-only (arena: the float32 pool needs 8 cells (32 bytes), over 8 bytes); 3 executor calls |}]
