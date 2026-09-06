(* Real-scale resource measurement for `aten.lstm.input` (project step 16):
   both checked-in `sequencer2d_s` corpus shapes (the submodule's own
   `lstm.input`-bearing model -- `csatv2` is a different, unrelated model;
   see the step 16 evidence note in _ai_/project_todo.md for that
   correction), `(B,L,I,K)=(16,16,384,96)` (28 occurrences) and
   `(32,32,192,48)` (8 occurrences), Q=1, R=2, biases, batch-first --
   lstm-plan.md §2's own re-read of the corpus. Reconciles the real
   `Region_execution.counters` this implementation produces against step 4's
   estimates (max per-key updates `2*16*192=2*32*96=6144`; summed
   worst-case Kernel total across all 36 occurrences and up to 3 live outputs
   each, `12,976,128`).

   A fixed constant fill, not random: only SHAPE drives scan/key/slot counts
   in this op's arithmetic (no value-dependent branching), so a real dataset
   would count identically. Declaration order below matches
   `g.Graph_ir.Graph.inputs`'s own order (input, weight_ih, weight_hh,
   bias_ih, bias_hh, [reverse's own four], h0, c0) -- see
   test/native/lstm_graph_test.ml's own header comment for why this pairing
   is positional, not name-based. *)

let filled shape v = Tensor.materialize shape (fun _ -> v)

let measure ?limits ?(states_live = true) ~label ~batch ~seq ~input_size
    ~hidden_size ~bidirectional ~batch_first () =
  let directions = if bidirectional then 2 else 1 in
  let mat_shape ~rows ~cols = Vec6.shape ~n:rows ~t:1 ~d:1 ~h:1 ~w:1 ~c:cols in
  let vec_shape ~n = Vec6.shape ~n ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let state_shape =
    Vec6.shape ~n:1 ~t:1 ~d:1 ~h:directions ~w:batch ~c:hidden_size
  in
  let seq_shape =
    if batch_first then Vec6.shape ~n:1 ~t:1 ~d:1 ~h:batch ~w:seq ~c:input_size
    else Vec6.shape ~n:1 ~t:1 ~d:1 ~h:seq ~w:batch ~c:input_size
  in
  let wih_shape = mat_shape ~rows:(4 * hidden_size) ~cols:input_size in
  let whh_shape = mat_shape ~rows:(4 * hidden_size) ~cols:hidden_size in
  let bias_shape = vec_shape ~n:(4 * hidden_size) in
  let params : Lstm.Lstm.params = { hidden_size; input_size; batch_first } in
  let body =
    Graph_builder.(
      let* input_id = input ~shape:seq_shape ~name:"input" () in
      let* wih_id = input ~shape:wih_shape ~name:"weight_ih" () in
      let* whh_id = input ~shape:whh_shape ~name:"weight_hh" () in
      let* bih_id = input ~shape:bias_shape ~name:"bias_ih" () in
      let* bhh_id = input ~shape:bias_shape ~name:"bias_hh" () in
      let* reverse =
        if bidirectional then
          let* wih_id_r = input ~shape:wih_shape ~name:"weight_ih_r" () in
          let* whh_id_r = input ~shape:whh_shape ~name:"weight_hh_r" () in
          let* bih_id_r = input ~shape:bias_shape ~name:"bias_ih_r" () in
          let* bhh_id_r = input ~shape:bias_shape ~name:"bias_hh_r" () in
          return
            (Some
               {
                 Lstm.Lstm.Direction.weight_ih = wih_id_r;
                 weight_hh = whh_id_r;
                 bias = Some (bih_id_r, bhh_id_r);
               })
        else return None
      in
      let* h0_id = input ~shape:state_shape ~name:"h0" () in
      let* c0_id = input ~shape:state_shape ~name:"c0" () in
      let layer : Lstm.Lstm.Layer.t =
        {
          forward =
            {
              weight_ih = wih_id;
              weight_hh = whh_id;
              bias = Some (bih_id, bhh_id);
            };
          reverse;
        }
      in
      Graph_builder.lstm params ~input:input_id ~layers:[ layer ] ~h0:h0_id
        ~c0:c0_id ())
  in
  let g =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      (if states_live then
         Graph_builder.build ~name:"lstm_scale"
           ~outputs:(fun (out, hn, cn) -> [ out; hn; cn ])
           body
       else
         Graph_builder.build ~name:"lstm_scale" ~outputs:Fun.id
           Graph_builder.(
             let* out_id, hn_id, cn_id = body in
             let* () = discard hn_id in
             let* () = discard cn_id in
             return [ out_id ]))
  in
  let forward_tensors =
    [
      filled seq_shape 0.01;
      filled wih_shape 0.01;
      filled whh_shape 0.01;
      filled bias_shape 0.01;
      filled bias_shape 0.01;
    ]
  in
  let reverse_tensors =
    if bidirectional then
      [
        filled wih_shape 0.01;
        filled whh_shape 0.01;
        filled bias_shape 0.01;
        filled bias_shape 0.01;
      ]
    else []
  in
  let state_tensors = [ filled state_shape 0.01; filled state_shape 0.01 ] in
  let input_tensors = forward_tensors @ reverse_tensors @ state_tensors in
  let inputs = List.combine g.Graph_ir.Graph.inputs input_tensors in
  let counters = Region_execution.counters () in
  (* The lstm NODE's own three output edges, not [g.Graph_ir.Graph.outputs] --
     which shrinks to one when [states_live] is false. [Eval_direct]
     materialises every node's full output arity regardless of graph-level
     exposure (.ai/native_multi_output_design.md §2: a [Discard]'d edge is
     still computed), so binding counters only to [graph.outputs] would
     measure "how much of the work did this test bother to count", not "how
     much work happened" -- the two questions the discard-comparison test
     below depends on not conflating. *)
  let lstm_node_outputs =
    match
      List.find_opt
        (fun (n : Graph_ir.node) ->
          match n.Graph_ir.Node.op with Graph_ir.Lstm _ -> true | _ -> false)
        g.Graph_ir.Graph.nodes
    with
    | Some n -> n.Graph_ir.Node.outputs
    | None -> invalid_arg "lstm_scale_test: no lstm node in the graph"
  in
  let region_counters =
    List.fold_left
      (fun m id -> Tensor_id.Map.add id counters m)
      Tensor_id.Map.empty lstm_node_outputs
  in
  match Eval_direct.run ?limits ~region_counters ~inputs g with
  | Error e ->
      Fmt.pr "%s: rejected: %a@." label Eval_direct.pp_error (Err.Error.kind e)
  | Ok (_ : Tensor.packed Tensor_id.Map.t) ->
      Fmt.pr
        "%s: keys=%d locals=%d emitters=%d loads=%d reductions=%d scans=%d \
         scan_updates=%d@."
        label counters.keys counters.locals counters.emitters counters.loads
        counters.reductions counters.scans counters.scan_updates

(* Same [seq]/[hidden_size]/[bidirectional] per family as the real corpus
   shapes below (same 6144 per-key boundary), [batch]/[input_size] shrunk
   like tests 2/3. Doesn't validate the real worst-case totals -- that's the
   gated test below -- but keeps the default run under a second. *)
let%expect_test
    "lstm real-scale resource counters: both corpus shapes (fast fixture)" =
  measure ~label:"family1 (16,16,384,96)" ~batch:1 ~seq:16 ~input_size:4
    ~hidden_size:96 ~bidirectional:true ~batch_first:true ();
  measure ~label:"family2 (32,32,192,48)" ~batch:1 ~seq:32 ~input_size:4
    ~hidden_size:48 ~bidirectional:true ~batch_first:true ();
  [%expect
    {|
    family1 (16,16,384,96): keys=3 locals=19584 emitters=3456 loads=6839424 reductions=6451200 scans=6 scan_updates=18432
    family2 (32,32,192,48): keys=3 locals=19008 emitters=3264 loads=3742272 reductions=3354624 scans=6 scan_updates=18432 |}]

(* Real corpus shapes, full [batch]/[input_size]. [@tags "disabled"] is
   ppx_inline_test's own gate: dropped by default, so plain `dune runtest`
   skips it. Re-run with `-require-tag disabled` on the built runner:
     _build/default/test/native/.native_test.inline-tests/inline-test-runner.exe \
       inline-test-runner native_test -require-tag disabled \
       -partition lstm_scale_test.ml -source-tree-root ../.. -diff-cmd -
   (from _build/default/test/native). *)
let%expect_test
    ("lstm real-scale resource counters: both corpus shapes (full, gated)"
     [@tags "disabled"]) =
  measure ~label:"family1 (16,16,384,96)" ~batch:16 ~seq:16 ~input_size:384
    ~hidden_size:96 ~bidirectional:true ~batch_first:true ();
  measure ~label:"family2 (32,32,192,48)" ~batch:32 ~seq:32 ~input_size:192
    ~hidden_size:48 ~bidirectional:true ~batch_first:true ();
  [%expect
    {|
    family1 (16,16,384,96): keys=48 locals=313344 emitters=55296 loads=893896704 reductions=495452160 scans=96 scan_updates=294912
    family2 (32,32,192,48): keys=96 locals=608256 emitters=104448 loads=895961088 reductions=495452160 scans=192 scan_updates=589824 |}]

(* "Verify default admission and rejection under tighter limits" (project
   step 16): the per-key update count this op needs at real corpus scale is
   independent of [batch]/[input_size] (only [directions*seq*2*hidden_size]
   -- confirmed above: family1's 6144 = 2*16*192, family2's 6144 = 2*32*96,
   neither involving batch or input width), so a cheap [batch=1, input_size=4]
   fixture at the SAME [seq]/[hidden_size]/[bidirectional] reaches the exact
   same per-key boundary as the real corpus shapes, without the real shapes'
   multi-minute cost. Default limits admit it (`max_scan_updates_per_key` =
   8192 > 6144); a limits value tightened to 6000 -- just under the real
   6144 -- rejects it with a typed `` `Scan `` error, not an exception. *)
let tightened_limits =
  Err.or_raise ~pp_error:Kernel.Limits.pp_error
    (Kernel.Limits.create ~max_size:4096 ~max_depth:128 ~max_values:4096
       ~max_dep_depth:1024 ~max_inputs:1024 ~max_outputs:1024
       ~max_extent:0x7FFF_FFFFL ~max_numel:0x7FFF_FFFFL ~max_local_slots:8192
       ~max_scan_state:8192 ~max_scan_updates_per_key:6000L
       ~max_scan_updates_total:16_000_000L)

let%expect_test
    "lstm: default limits admit the real per-key count, a tighter one rejects \
     it" =
  measure ~label:"default limits (per-key=6144)" ~batch:1 ~seq:16 ~input_size:4
    ~hidden_size:96 ~bidirectional:true ~batch_first:true ();
  measure ~limits:tightened_limits ~label:"tightened limits (cap=6000 < 6144)"
    ~batch:1 ~seq:16 ~input_size:4 ~hidden_size:96 ~bidirectional:true
    ~batch_first:true ();
  [%expect
    {|
    default limits (per-key=6144): keys=3 locals=19584 emitters=3456 loads=6839424 reductions=6451200 scans=6 scan_updates=18432
    tightened limits (cap=6000 < 6144): rejected: region key's scan updates exceed limit 6000 |}]

(* "Measure duplicate trace execution when all LSTM outputs are live"
   (project step 19's own first bullet, explicitly independent of step 18's
   still-open ground-IR question). Discarding [h_n]/[c_n] in the SOURCE
   graph ([states_live:false]) does NOT reduce [Eval_direct]'s cost at all:
   [Discard] only marks an edge unread, it does not suppress computation
   (.ai/native_multi_output_design.md §2, "Direct eval still materialises
   the discarded producer's edge"), and [Eval_direct] iterates every node's
   FULL [Node.outputs] arity unconditionally, with no graph-reachability-
   based elision. This run confirms that structural claim empirically
   rather than only citing it: the two counter lines below are IDENTICAL.
   That identity IS this bullet's measurement -- today, requesting only
   `output` costs exactly the same as requesting all three, which is the
   concrete "duplicate trace execution" step 19 proposes to eliminate by
   sharing one computation across selected outputs instead of building
   three independent copies regardless of what is asked for. *)
let%expect_test "lstm: discarding unread state outputs saves nothing today" =
  measure ~label:"all three live" ~batch:1 ~seq:16 ~input_size:4 ~hidden_size:96
    ~bidirectional:true ~batch_first:true ();
  measure ~states_live:false ~label:"h_n/c_n discarded" ~batch:1 ~seq:16
    ~input_size:4 ~hidden_size:96 ~bidirectional:true ~batch_first:true ();
  [%expect
    {|
    all three live: keys=3 locals=19584 emitters=3456 loads=6839424 reductions=6451200 scans=6 scan_updates=18432
    h_n/c_n discarded: keys=3 locals=19584 emitters=3456 loads=6839424 reductions=6451200 scans=6 scan_updates=18432 |}]
