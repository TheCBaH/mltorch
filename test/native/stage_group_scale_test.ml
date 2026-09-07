(* Project step 19, Section C milestone 2: [Stage_program.ground] now shares
   one evaluation of an Lstm node's shared locals across its three sibling
   stages per canonical batch key (via maximal [Region_group.Ref.Grouped]
   runs, see [Stage_program.runs_of_stages]/[execute_run]), instead of each
   stage independently rebuilding/re-evaluating them --
   test/native/lstm_scale_test.ml already proved the same reduction for
   [Eval_direct.run]. Builds a stacked (layers=2), bidirectional, batch=2
   graph so [scans = keys*layers*directions = 2*2*2 = 8] is non-trivial,
   grounds it through [Eval_symbolic.run] + [Stage_program.ground], and
   checks:
   - numerical agreement with [Eval_direct.run] on the identical graph/inputs
     (Direct's own correctness is already independently established against a
     hand-derived oracle in lstm_graph_layers_test.ml/lstm_graph_test.ml);
   - [keys = batch] (not 3x) and [scans = keys*layers*directions] (not 3x) --
     the same shared-vs-tripled distinction lstm_scale_test.ml's own exit
     criterion already proved for Direct, now for the Stage path; matches the
     design record's §8 formula [shared scan_updates = B*L*R*T*(2*K)] exactly
     (2*2*2*2*(2*2) = 64). *)

let filled shape v = Tensor.materialize shape (fun _ -> v)
let mat_shape ~rows ~cols = Vec6.shape ~n:rows ~t:1 ~d:1 ~h:1 ~w:1 ~c:cols
let vec_shape ~n = Vec6.shape ~n ~t:1 ~d:1 ~h:1 ~w:1 ~c:1
let k = 2 (* hidden_size *)
let isz = 2 (* input width *)
let seq = 2
let batch = 2
let num_layers = 2
let directions = 2
let bias_shape = vec_shape ~n:(4 * k)

let state_shape =
  Vec6.shape ~n:1 ~t:1 ~d:1 ~h:(num_layers * directions) ~w:batch ~c:k

let seq_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:seq ~w:batch ~c:isz

let params : Lstm.Lstm.params =
  { hidden_size = k; input_size = isz; batch_first = false }

let direction_inputs ~layer_input_size =
  Graph_builder.(
    let* wih =
      input
        ~shape:(mat_shape ~rows:(4 * k) ~cols:layer_input_size)
        ~name:"wih" ()
    in
    let* whh = input ~shape:(mat_shape ~rows:(4 * k) ~cols:k) ~name:"whh" () in
    let* bih = input ~shape:bias_shape ~name:"bih" () in
    let* bhh = input ~shape:bias_shape ~name:"bhh" () in
    return
      {
        Lstm.Lstm.Direction.weight_ih = wih;
        weight_hh = whh;
        bias = Some (bih, bhh);
      })

let layer_inputs ~layer_input_size =
  Graph_builder.(
    let* forward = direction_inputs ~layer_input_size in
    let* reverse = direction_inputs ~layer_input_size in
    return { Lstm.Lstm.Layer.forward; reverse = Some reverse })

let graph () =
  Graph_builder.(
    let* input_id = input ~shape:seq_shape ~name:"input" () in
    let* layer0 = layer_inputs ~layer_input_size:isz in
    let* layer1 = layer_inputs ~layer_input_size:(directions * k) in
    let* h0_id = input ~shape:state_shape ~name:"h0" () in
    let* c0_id = input ~shape:state_shape ~name:"c0" () in
    lstm params ~input:input_id ~layers:[ layer0; layer1 ] ~h0:h0_id ~c0:c0_id
      ())

let build_graph () =
  Err.or_raise ~pp_error:Graph_builder.pp_error
    (Graph_builder.build ~name:"stage_group_scale"
       ~outputs:(fun (out, hn, cn) -> [ out; hn; cn ])
       (graph ()))

let inputs g =
  (* Declaration order matches [g.Graph_ir.Graph.inputs]: input, then layer
     0's forward (wih,whh,bih,bhh), reverse (wih,whh,bih,bhh), layer 1's same
     four groups (with layer 1's own wider wih), then h0, c0 -- positional,
     mirroring lstm_graph_test.ml's own header note. *)
  let direction_tensors ~layer_input_size =
    [
      filled (mat_shape ~rows:(4 * k) ~cols:layer_input_size) 0.01;
      filled (mat_shape ~rows:(4 * k) ~cols:k) 0.01;
      filled bias_shape 0.01;
      filled bias_shape 0.01;
    ]
  in
  let tensors =
    [ filled seq_shape 0.01 ]
    @ direction_tensors ~layer_input_size:isz
    @ direction_tensors ~layer_input_size:isz
    @ direction_tensors ~layer_input_size:(directions * k)
    @ direction_tensors ~layer_input_size:(directions * k)
    @ [ filled state_shape 0.01; filled state_shape 0.01 ]
  in
  List.combine g.Graph_ir.Graph.inputs tensors

let lstm_node_outputs (g : Graph_ir.graph) =
  match
    List.find_opt
      (fun (n : Graph_ir.node) ->
        match n.Graph_ir.Node.op with Graph_ir.Lstm _ -> true | _ -> false)
      g.Graph_ir.Graph.nodes
  with
  | Some n -> n.Graph_ir.Node.outputs
  | None -> invalid_arg "stage_group_scale_test: no lstm node in the graph"

let max_abs_diff (Tensor.Tensor a) (Tensor.Tensor b) =
  Vec6.fold_coords a.Tensor.shape ~init:0. ~f:(fun acc coord ->
      let i = (Vec6.offset a.Tensor.shape coord :> int) in
      let c = Dim.to_int (Vec6.get coord Axis.C) in
      Float.max acc
        (Float.abs
           (Payload.get_float a.Tensor.payload ~c ~i
           -. Payload.get_float b.Tensor.payload ~c ~i)))

let%expect_test
    "Stage_program.ground shares one recurrence per batch key, agreeing with \
     Eval_direct" =
  let g = build_graph () in
  let bindings = inputs g in
  let direct_env =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.run g ~inputs:bindings)
  in
  let prog = Eval_symbolic.run g in
  let bind_map =
    List.fold_left
      (fun m (id, t) -> Tensor_id.Map.add id t m)
      Tensor_id.Map.empty bindings
  in
  let outs = lstm_node_outputs g in
  let region_counters =
    let counters = Region_execution.counters () in
    List.fold_left
      (fun m id -> Tensor_id.Map.add id counters m)
      Tensor_id.Map.empty outs
  in
  let staged =
    Err.or_raise ~pp_error:Stage_program.pp_error
      (Stage_program.ground prog ~region_counters ~bind:(fun id ->
           Tensor_id.Map.find id bind_map))
  in
  let counters = Tensor_id.Map.find (List.hd outs) region_counters in
  List.iter
    (fun oid ->
      let direct = Tensor_id.Map.find oid direct_env in
      let stage = Tensor_id.Map.find oid staged in
      Fmt.pr "output %a: max_abs_diff=%g@." Tensor_id.pp oid
        (max_abs_diff direct stage))
    outs;
  Fmt.pr "keys=%d scans=%d scan_updates=%d@." counters.Region_execution.keys
    counters.Region_execution.scans counters.Region_execution.scan_updates;
  [%expect
    {|
    output t19: max_abs_diff=0
    output t20: max_abs_diff=0
    output t21: max_abs_diff=0
    keys=2 scans=8 scan_updates=64 |}]

(* Non-vacuous per CLAUDE.md: temporarily replacing [runs_of_stages]'s body
   with a version that maps every [Grouped] stage to its own singleton
   [Group_run] (never merging consecutive siblings into one run) and re-running
   this test reproduced exactly the OLD, tripled counters --
   [keys=6 scans=24 scan_updates=192], 3x this test's real, shared numbers
   above -- confirming this test would catch a regression to independent
   per-stage execution. Reverted; not left in the tree as a runtime toggle
   since [runs_of_stages] has no legitimate reason to expose one. *)

(* Project step 19 / Section D, acceptance matrix §7 item 4 ("Scheduling"):
   "reject duplicate ordinals". No real symbolic builder ever produces this --
   both always assign one stage per emitter ordinal -- so this hand-corrupts a
   real, otherwise-valid [Stage_program.t]: a SECOND stage, under a fresh id
   distinct from every real one, referencing the SAME group's ordinal 0 a real
   stage already claims, spliced in immediately after it (still one contiguous
   run over the same physical [Region_group.t]). Before this session's fix,
   [Region_execution.materialize_group]'s own [tensors]/[List.assoc] pairing
   would have silently bound one of the two ordinal-0 stages to an
   UNWRITTEN, freshly-allocated tensor instead of failing -- see the commit
   fixing this for the direct before/after evidence. *)
let%expect_test
    "Stage_program.ground rejects a group run with a repeated emitter ordinal" =
  let g = build_graph () in
  let prog = Eval_symbolic.run g in
  let out0 = List.hd (lstm_node_outputs g) in
  let stage0 =
    List.find
      (fun (st : Stage_program.Stage.t) ->
        Tensor_id.equal st.Stage_program.Stage.id out0)
      prog.Stage_program.stages
  in
  let fake_id = Tensor_id.of_int 999_999 in
  let fake_sg =
    { stage0.Stage_program.Stage.sg with Tensor_sig.id = fake_id }
  in
  let fake_stage =
    { stage0 with Stage_program.Stage.id = fake_id; sg = fake_sg }
  in
  let corrupted =
    {
      prog with
      Stage_program.stages =
        List.concat_map
          (fun (st : Stage_program.Stage.t) ->
            if Tensor_id.equal st.Stage_program.Stage.id out0 then
              [ st; fake_stage ]
            else [ st ])
          prog.Stage_program.stages;
    }
  in
  let bind_map =
    List.fold_left
      (fun m (id, t) -> Tensor_id.Map.add id t m)
      Tensor_id.Map.empty (inputs g)
  in
  (match
     Stage_program.ground corrupted ~bind:(fun id ->
         Tensor_id.Map.find id bind_map)
   with
  | Error e -> Fmt.pr "%a@." Stage_program.pp_error (Err.Error.kind e)
  | Ok _ -> Fmt.pr "unexpectedly accepted@.");
  [%expect {| group run repeats emitter ordinal 0 |}]
