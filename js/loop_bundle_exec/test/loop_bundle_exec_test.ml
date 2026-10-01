open Graph_ir
open Loop_ir

(* Deterministic, sign-varying values from a buffer's own signature, offset by
   [salt] so two runs of one prepared bundle see different inputs. *)
let tensor_of ~salt (sg : Tensor_sig.t) =
  let v c =
    float_of_int
      ((((Vec6.offset sg.Tensor_sig.shape c :> int) + salt) mod 7) - 3)
    /. 4.
  in
  Tensor.materialize sg.Tensor_sig.shape v

let values t =
  let (Tensor.Tensor tt) = t in
  let acc = ref [] in
  Vec6.iter tt.Tensor.shape (fun c ->
      acc := Tensor.read_at t (Vec6.get c) :: !acc);
  List.rev !acc

let input_sigs (b : Loop_bundle.t) ids =
  List.map
    (fun id ->
      let buf =
        List.find_map
          (fun (inv : Loop_bundle.invocation) ->
            List.find_map
              (fun ((buf : Loop_buffer.t), edge) ->
                if Tensor_id.equal edge id then Some buf else None)
              (List.combine inv.Loop_bundle.program.Loop_program.buffers
                 inv.Loop_bundle.edges))
          b.Loop_bundle.invocations
        |> Option.get
      in
      (id, buf.Loop_buffer.sg))
    ids

let bundle g = Err.or_raise ~pp_error:Loop_bundle.pp_error (Loop_bundle.build g)
let map_of l id = List.assoc_opt id l

let compare_run g p (b : Loop_bundle.t) ~constants ~salt =
  let inputs =
    List.map
      (fun (id, sg) -> (id, tensor_of ~salt sg))
      (input_sigs b b.Loop_bundle.inputs)
  in
  let reference =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.run g ~constants ~inputs)
  in
  match Err.payload (Loop_bundle_exec.run p ~bind:(map_of inputs)) with
  | Error e -> Fmt.pr "run failed: %a@." Loop_bundle_exec.pp_error e
  | Ok out ->
      List.iter
        (fun id ->
          let same =
            values (Tensor_id.Map.find id out)
            = values (Tensor_id.Map.find id reference)
          in
          Fmt.pr "output t%d identical to per-node evaluator: %b@."
            (Tensor_id.to_int id) same)
        g.Graph.outputs

let%expect_test "chain: repeated prepared runs match the per-node evaluator" =
  let g = Native_test.Graph_fixtures.chain () in
  let b = bundle g in
  let constants =
    List.map
      (fun (id, sg) -> (id, tensor_of ~salt:3 sg))
      (input_sigs b b.Loop_bundle.constants)
  in
  let p =
    Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
      (Loop_bundle_exec.prepare b ~constants:(map_of constants))
  in
  compare_run g p b ~constants ~salt:0;
  compare_run g p b ~constants ~salt:2;
  compare_run g p b ~constants ~salt:5;
  [%expect
    {|
    output t9 identical to per-node evaluator: true
    output t9 identical to per-node evaluator: true
    output t9 identical to per-node evaluator: true
    |}]

let%expect_test "an unbound input is refused before anything executes" =
  let g = Native_test.Graph_fixtures.chain () in
  let b = bundle g in
  let constants =
    List.map
      (fun (id, sg) -> (id, tensor_of ~salt:3 sg))
      (input_sigs b b.Loop_bundle.constants)
  in
  let p =
    Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
      (Loop_bundle_exec.prepare b ~constants:(map_of constants))
  in
  (match Err.payload (Loop_bundle_exec.run p ~bind:(fun _ -> None)) with
  | Ok _ -> Fmt.pr "ran@."
  | Error e -> Fmt.pr "%a@." Loop_bundle_exec.pp_error e);
  [%expect {| no binding for input t0 |}]

(* An I64 pool and a data-dependent failure: [relu] -> [index_tensor] (index an
   I64 input) -> [relu]. The index is the only thing that varies between runs. *)
let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c

let gather_graph () =
  Graph_builder.build ~name:"gather"
    ~outputs:(fun o -> [ o ])
    Graph_builder.(
      let* self = input ~shape:(s 1 1 1 1 3 2) () in
      let* index =
        input ~shape:(s 1 1 1 1 1 2) ~fmt:(Payload.Fmt Payload.I64) ()
      in
      let* r = relu self in
      let* g =
        index_tensor
          {
            Index_tensor.Index_tensor.axis = Axis.W;
            index_rank = Rank.of_int 1;
          }
          ~self:r ~index
      in
      relu g)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let index_tensor_of (a, b) =
  Tensor.materialize_i64 (s 1 1 1 1 1 2) (fun c ->
      if Dim.to_int (Vec6.get c Axis.C) = 0 then a else b)

let%expect_test
    "an I64 pool round-trips; a failing kernel reports its node, and the next \
     run is unaffected" =
  let g = gather_graph () in
  let b = bundle g in
  let p =
    Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
      (Loop_bundle_exec.prepare b ~constants:(fun _ -> None))
  in
  let self_id, index_id =
    match b.Loop_bundle.inputs with [ a; i ] -> (a, i) | _ -> assert false
  in
  let self =
    tensor_of ~salt:1 (List.assoc self_id (input_sigs b [ self_id ]))
  in
  let attempt idx =
    let index = index_tensor_of idx in
    let bind id =
      if Tensor_id.equal id self_id then Some self
      else if Tensor_id.equal id index_id then Some index
      else None
    in
    let reference =
      match
        Eval_direct.run g ~inputs:[ (self_id, self); (index_id, index) ]
      with
      | r -> Some (Err.or_raise ~pp_error:Eval_direct.pp_error r)
      | exception _ ->
          Fmt.pr "per-node: raised@.";
          None
    in
    match Err.payload (Loop_bundle_exec.run p ~bind) with
    | Ok out ->
        let id = List.hd g.Graph.outputs in
        Fmt.pr "ok, identical=%b@."
          (Option.map
             (fun r ->
               values (Tensor_id.Map.find id out)
               = values (Tensor_id.Map.find id r))
             reference
          = Some true)
    | Error e -> Fmt.pr "error: %a@." Loop_bundle_exec.pp_error e
  in
  attempt (2L, 0L);
  attempt (0L, 7L);
  attempt (1L, 2L);
  [%expect
    {|
    ok, identical=true
    per-node: raised
    error: node n1: gather index 7 out of range [-3, 2]
    ok, identical=true |}]

(* Result leases: a run pins its execution arena set until released. *)
let%expect_test
    "leases: bounded exhaustion, two live results, release and reuse" =
  let g = Native_test.Graph_fixtures.chain () in
  let b = bundle g in
  let constants =
    List.map
      (fun (id, sg) -> (id, tensor_of ~salt:3 sg))
      (input_sigs b b.Loop_bundle.constants)
  in
  let p =
    Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
      (Loop_bundle_exec.prepare ~max_outstanding:2 b
         ~constants:(map_of constants))
  in
  let out_id = List.hd g.Graph.outputs in
  let inputs salt =
    List.map
      (fun (id, sg) -> (id, tensor_of ~salt sg))
      (input_sigs b b.Loop_bundle.inputs)
  in
  let reference salt =
    values
      (Tensor_id.Map.find out_id
         (Err.or_raise ~pp_error:Eval_direct.pp_error
            (Eval_direct.run g ~constants ~inputs:(inputs salt))))
  in
  let lease salt = Loop_bundle_exec.run_leased p ~bind:(map_of (inputs salt)) in
  let get l = Err.or_raise ~pp_error:Loop_bundle_exec.pp_error l in
  let read l = values (get (Loop_bundle_exec.output l out_id)) in
  let a = get (lease 0) and b' = get (lease 4) in
  (match Err.payload (lease 1) with
  | Error e -> Fmt.pr "third run: %a@." Loop_bundle_exec.pp_error e
  | Ok _ -> Fmt.pr "third run: ran@.");
  Fmt.pr "a = ref0: %b, b = ref4: %b@."
    (read a = reference 0)
    (read b' = reference 4);
  let copy = get (Loop_bundle_exec.output a out_id) in
  Loop_bundle_exec.release a;
  (match Err.payload (Loop_bundle_exec.output a out_id) with
  | Error e -> Fmt.pr "after release: %a@." Loop_bundle_exec.pp_error e
  | Ok _ -> Fmt.pr "after release: readable@.");
  let c = get (lease 1) in
  Fmt.pr "c = ref1: %b, b still = ref4: %b, copy-out of a still = ref0: %b@."
    (read c = reference 1)
    (read b' = reference 4)
    (values copy = reference 0);
  [%expect
    {|
    third run: every execution arena is running or holds a live result
    a = ref0: true, b = ref4: true
    after release: the result lease was already released
    c = ref1: true, b still = ref4: true, copy-out of a still = ref0: true |}]

(* [Shared_execution]: inputs, intermediates and outputs share one arena, so a
   dead slot is reused; the answer must not change. *)
let%expect_test "Shared_execution layout matches the per-node evaluator" =
  let g = Native_test.Graph_fixtures.chain () in
  let config : Storage_script.Config.t =
    {
      layout = Storage_script.Layout.Shared_execution;
      constants = Storage_script.Ownership.Copied;
      inputs = Storage_script.Ownership.Copied;
    }
  in
  let b =
    Err.or_raise ~pp_error:Loop_bundle.pp_error (Loop_bundle.build ~config g)
  in
  let constants =
    List.map
      (fun (id, sg) -> (id, tensor_of ~salt:3 sg))
      (input_sigs b b.Loop_bundle.constants)
  in
  let p =
    Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
      (Loop_bundle_exec.prepare b ~constants:(map_of constants))
  in
  compare_run g p b ~constants ~salt:0;
  compare_run g p b ~constants ~salt:6;
  [%expect
    {|
    output t9 identical to per-node evaluator: true
    output t9 identical to per-node evaluator: true |}]

(* Forwarded and repeated outputs: a graph output that IS a graph input, and one
   edge named twice. *)
let forwarded ?(ownership = Storage_script.Ownership.Copied) layout =
  let g =
    Graph_builder.build ~name:"forwarded"
      ~outputs:(fun (x, o) -> [ x; o; o ])
      Graph_builder.(
        let* x = input ~shape:(s 1 1 1 1 1 4) () in
        let+ o = relu x in
        (x, o))
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  let config : Storage_script.Config.t =
    { layout; constants = ownership; inputs = ownership }
  in
  let b =
    Err.or_raise ~pp_error:Loop_bundle.pp_error (Loop_bundle.build ~config g)
  in
  let p =
    Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
      (Loop_bundle_exec.prepare b ~constants:(fun _ -> None))
  in
  let x_id = List.hd b.Loop_bundle.inputs in
  let x = tensor_of ~salt:1 (List.assoc x_id (input_sigs b [ x_id ])) in
  let out =
    Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
      (Loop_bundle_exec.run p ~bind:(map_of [ (x_id, x) ]))
  in
  let reference =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.run g ~inputs:[ (x_id, x) ])
  in
  List.iter
    (fun id ->
      Fmt.pr "t%d identical=%b@." (Tensor_id.to_int id)
        (values (Tensor_id.Map.find id out)
        = values (Tensor_id.Map.find id reference)))
    (List.sort_uniq Tensor_id.compare g.Graph.outputs);
  Fmt.pr "input unchanged=%b@."
    (values x
    = values (tensor_of ~salt:1 (List.assoc x_id (input_sigs b [ x_id ]))))

let%expect_test "forwarded input and repeated output are copied out intact" =
  List.iter
    (fun layout ->
      Fmt.pr "layout %s@."
        (match layout with
        | Storage_script.Layout.Separate -> "separate"
        | Storage_script.Layout.Shared_execution -> "shared");
      forwarded layout)
    [ Storage_script.Layout.Separate; Storage_script.Layout.Shared_execution ];
  [%expect
    {|
    layout separate
    t0 identical=true
    t1 identical=true
    input unchanged=true
    layout shared
    t0 identical=true
    t1 identical=true
    input unchanged=true |}]

(* [Borrowed]: constants and inputs stay the caller's own typed arrays, used in
   place and never written. *)
let%expect_test "Borrowed constants and inputs: in place, never written" =
  let g = Native_test.Graph_fixtures.chain () in
  let config : Storage_script.Config.t =
    {
      layout = Storage_script.Layout.Separate;
      constants = Storage_script.Ownership.Borrowed;
      inputs = Storage_script.Ownership.Borrowed;
    }
  in
  let b =
    Err.or_raise ~pp_error:Loop_bundle.pp_error (Loop_bundle.build ~config g)
  in
  let constants =
    List.map
      (fun (id, sg) -> (id, tensor_of ~salt:3 sg))
      (input_sigs b b.Loop_bundle.constants)
  in
  let p =
    Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
      (Loop_bundle_exec.prepare b ~constants:(map_of constants))
  in
  compare_run g p b ~constants ~salt:0;
  compare_run g p b ~constants ~salt:5;
  Fmt.pr "constants unchanged=%b@."
    (List.for_all
       (fun (id, sg) ->
         values (List.assoc id constants) = values (tensor_of ~salt:3 sg))
       (input_sigs b b.Loop_bundle.constants));
  [%expect
    {|
    output t9 identical to per-node evaluator: true
    output t9 identical to per-node evaluator: true
    constants unchanged=true |}];
  (* In place, not a snapshot: overwriting the caller's constants after
     [prepare] is seen by the next run. A [Copied] bundle prepared from the same
     tensors keeps the values it copied, so it now disagrees. *)
  let copied =
    let b = bundle g in
    Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
      (Loop_bundle_exec.prepare b ~constants:(map_of constants))
  in
  List.iter (fun (_, t) -> Tensor.zero_fill t) constants;
  Fmt.pr "borrowed sees the overwrite:@.";
  compare_run g p b ~constants ~salt:0;
  Fmt.pr "copied keeps its snapshot:@.";
  compare_run g copied b ~constants ~salt:0;
  [%expect
    {|
    borrowed sees the overwrite:
    output t9 identical to per-node evaluator: true
    copied keeps its snapshot:
    output t9 identical to per-node evaluator: false |}];
  Fmt.pr "forwarded, borrowed:@.";
  forwarded ~ownership:Storage_script.Ownership.Borrowed
    Storage_script.Layout.Separate;
  [%expect
    {|
    forwarded, borrowed:
    t0 identical=true
    t1 identical=true
    input unchanged=true |}]

(* Region-authored ops, whole graph in one entry call: the permute (a Node
   kernel) feeds a Region kernel whose omitted operands are synthetic defaults,
   and the Region program's local ids collide with graph ids. *)
let%expect_test "Region-authored nodes match the per-node evaluator" =
  List.iter
    (fun (name, g) ->
      Fmt.pr "%s@." name;
      let b = bundle g in
      let p =
        Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
          (Loop_bundle_exec.prepare b ~constants:(fun _ -> None))
      in
      compare_run g p b ~constants:[] ~salt:0;
      compare_run g p b ~constants:[] ~salt:3)
    [
      ("layer_norm", Native_test.Graph_fixtures.sink_permute_layer_norm ());
      ("sdpa", Native_test.Graph_fixtures.sink_permute_sdpa ());
    ];
  [%expect
    {|
    layer_norm
    output t2 identical to per-node evaluator: true
    output t2 identical to per-node evaluator: true
    sdpa
    output t4 identical to per-node evaluator: true
    output t4 identical to per-node evaluator: true |}]

(* A grouped Region node: Lstm's three outputs come from one recurrence. The
   bundle runs it as ONE invocation writing all three, which must still equal
   the per-node evaluator's. *)
let lstm_graph () =
  let k = 2 and isz = 2 and seq = 3 in
  let mat rows cols = s rows 1 1 1 1 cols in
  let vec n = s n 1 1 1 1 1 in
  let state = s 1 1 1 1 1 k in
  let params : Lstm.Lstm.params =
    { hidden_size = k; input_size = isz; batch_first = false }
  in
  Graph_builder.(
    build ~name:"lstm" ~outputs:(fun (o, h, c) -> [ o; h; c ])
    @@
    let* x = input ~shape:(s 1 1 1 seq 1 isz) () in
    let* wih = input ~shape:(mat (4 * k) isz) () in
    let* whh = input ~shape:(mat (4 * k) k) () in
    let* bih = input ~shape:(vec (4 * k)) () in
    let* bhh = input ~shape:(vec (4 * k)) () in
    let* h0 = input ~shape:state () in
    let* c0 = input ~shape:state () in
    let layer : Lstm.Lstm.Layer.t =
      {
        forward = { weight_ih = wih; weight_hh = whh; bias = Some (bih, bhh) };
        reverse = None;
      }
    in
    lstm params ~input:x ~layers:[ layer ] ~h0 ~c0 ())
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let%expect_test "a grouped Region node (Lstm) matches the per-node evaluator" =
  let g = lstm_graph () in
  let b = bundle g in
  let p =
    Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
      (Loop_bundle_exec.prepare b ~constants:(fun _ -> None))
  in
  Fmt.pr "invocations=%d outputs=%d@."
    (List.length b.Loop_bundle.invocations)
    (List.length (List.hd b.Loop_bundle.invocations).Loop_bundle.outputs);
  compare_run g p b ~constants:[] ~salt:0;
  compare_run g p b ~constants:[] ~salt:2;
  [%expect
    {|
    invocations=1 outputs=3
    output t7 identical to per-node evaluator: true
    output t8 identical to per-node evaluator: true
    output t9 identical to per-node evaluator: true
    output t7 identical to per-node evaluator: true
    output t8 identical to per-node evaluator: true
    output t9 identical to per-node evaluator: true |}]

(* Two bidirectional layers: the shared recurrence spans layers and directions. *)
let stacked_lstm_graph ~outputs =
  let k = 2 and isz = 2 and seq = 3 in
  let mat rows cols = s rows 1 1 1 1 cols in
  let vec n = s n 1 1 1 1 1 in
  let state = s 1 1 1 4 1 k in
  let params : Lstm.Lstm.params =
    { hidden_size = k; input_size = isz; batch_first = false }
  in
  Graph_builder.(
    build ~name:"lstm_stacked" ~outputs
    @@
    let direction cols =
      let* wih = input ~shape:(mat (4 * k) cols) () in
      let* whh = input ~shape:(mat (4 * k) k) () in
      let* bih = input ~shape:(vec (4 * k)) () in
      let+ bhh = input ~shape:(vec (4 * k)) () in
      {
        Lstm.Lstm.Direction.weight_ih = wih;
        weight_hh = whh;
        bias = Some (bih, bhh);
      }
    in
    let layer cols =
      let* forward = direction cols in
      let+ reverse = direction cols in
      { Lstm.Lstm.Layer.forward; reverse = Some reverse }
    in
    let* x = input ~shape:(s 1 1 1 seq 1 isz) () in
    let* l0 = layer isz in
    let* l1 = layer (2 * k) in
    let* h0 = input ~shape:state () in
    let* c0 = input ~shape:state () in
    lstm params ~input:x ~layers:[ l0; l1 ] ~h0 ~c0 ())
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let%expect_test "a stacked bidirectional Lstm matches the per-node evaluator" =
  let g = stacked_lstm_graph ~outputs:(fun (o, h, c) -> [ o; h; c ]) in
  let b = bundle g in
  let p =
    Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
      (Loop_bundle_exec.prepare b ~constants:(fun _ -> None))
  in
  Fmt.pr "invocations=%d@." (List.length b.Loop_bundle.invocations);
  compare_run g p b ~constants:[] ~salt:0;
  [%expect
    {|
    invocations=1
    output t19 identical to per-node evaluator: true
    output t20 identical to per-node evaluator: true
    output t21 identical to per-node evaluator: true |}]

(* Two gathers of the same shape intern to ONE kernel function. When the second
   fails, the row still names the second invocation's own node. *)
let shared_gather_graph () =
  Graph_builder.build ~name:"shared_gather"
    ~outputs:(fun (a, b) -> [ a; b ])
    Graph_builder.(
      let i64 = Payload.Fmt Payload.I64 in
      let params =
        { Index_tensor.Index_tensor.axis = Axis.W; index_rank = Rank.of_int 1 }
      in
      let* x = input ~shape:(s 1 1 1 1 3 2) () in
      let* y = input ~shape:(s 1 1 1 1 3 2) () in
      let* ia = input ~shape:(s 1 1 1 1 1 2) ~fmt:i64 () in
      let* ib = input ~shape:(s 1 1 1 1 1 2) ~fmt:i64 () in
      let* a = index_tensor params ~self:x ~index:ia in
      let+ b = index_tensor params ~self:y ~index:ib in
      (a, b))
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let%expect_test "a failure in a shared kernel names its own invocation" =
  let g = shared_gather_graph () in
  let b = bundle g in
  let js =
    Err.or_raise ~pp_error:Loop_bundle_js.pp_error (Loop_bundle_js.build b)
  in
  Fmt.pr "invocations=%d distinct kernels=%d@."
    (List.length b.Loop_bundle.invocations)
    js.Loop_bundle_js.distinct_kernels;
  let p =
    Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
      (Loop_bundle_exec.prepare b ~constants:(fun _ -> None))
  in
  let x, y, ia, ib =
    match b.Loop_bundle.inputs with
    | [ x; y; ia; ib ] -> (x, y, ia, ib)
    | _ -> assert false
  in
  let tx = tensor_of ~salt:1 (List.assoc x (input_sigs b [ x ])) in
  let ty = tensor_of ~salt:2 (List.assoc y (input_sigs b [ y ])) in
  let attempt a bad =
    let bind id =
      if Tensor_id.equal id x then Some tx
      else if Tensor_id.equal id y then Some ty
      else if Tensor_id.equal id ia then Some (index_tensor_of a)
      else if Tensor_id.equal id ib then Some (index_tensor_of bad)
      else None
    in
    match Err.payload (Loop_bundle_exec.run p ~bind) with
    | Ok _ -> Fmt.pr "ok@."
    | Error e -> Fmt.pr "%a@." Loop_bundle_exec.pp_error e
  in
  attempt (0L, 1L) (2L, 0L);
  attempt (0L, 9L) (2L, 0L);
  attempt (0L, 1L) (2L, 9L);
  [%expect
    {|
    invocations=2 distinct kernels=1
    ok
    node n0: gather index 9 out of range [-3, 2]
    node n1: gather index 9 out of range [-3, 2] |}]

(* An exception out of the caller's [bind] must not strand the execution set. *)
let%expect_test "an exception during a run frees the execution set" =
  let g = Native_test.Graph_fixtures.chain () in
  let b = bundle g in
  let constants =
    List.map
      (fun (id, sg) -> (id, tensor_of ~salt:3 sg))
      (input_sigs b b.Loop_bundle.constants)
  in
  let p =
    Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
      (Loop_bundle_exec.prepare b ~constants:(map_of constants))
  in
  (match Loop_bundle_exec.run p ~bind:(fun _ -> failwith "boom") with
  | exception Failure m -> Fmt.pr "raised %s@." m
  | _ -> Fmt.pr "no exception@.");
  compare_run g p b ~constants ~salt:1;
  [%expect
    {|
    raised boom
    output t9 identical to per-node evaluator: true |}]

(* A prepared bundle owns exactly one constant version; a different version is a
   different [prepare], and neither disturbs the other's runs. *)
let%expect_test "two prepared constant versions run interleaved" =
  let g = Native_test.Graph_fixtures.chain () in
  let b = bundle g in
  let version salt =
    let constants =
      List.map
        (fun (id, sg) -> (id, tensor_of ~salt sg))
        (input_sigs b b.Loop_bundle.constants)
    in
    ( constants,
      Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
        (Loop_bundle_exec.prepare b ~constants:(map_of constants)) )
  in
  (* Salts 3 and 4 keep batch-norm's running variance positive; other salts
     make the reference NaN, which [=] on floats then calls unequal. *)
  let c1, p1 = version 3 and c2, p2 = version 4 in
  compare_run g p1 b ~constants:c1 ~salt:0;
  compare_run g p2 b ~constants:c2 ~salt:0;
  compare_run g p1 b ~constants:c1 ~salt:2;
  [%expect
    {|
    output t9 identical to per-node evaluator: true
    output t9 identical to per-node evaluator: true
    output t9 identical to per-node evaluator: true |}]
