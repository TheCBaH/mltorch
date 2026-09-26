(* Project step 19, Section C milestone 2 (final piece): [Kernel_eval]'s eager
   materialize path (`run`/`run_plan`, via `execute`) now shares one
   evaluation of an Lstm node's shared locals across its three sibling
   values per canonical batch key (via maximal [Region_group.Ref.Grouped]
   runs, see [Kernel_eval.execute]/[materialize_group]), instead of each
   value independently rebuilding/re-projecting them --
   test/native/lstm_scale_test.ml and test/native/stage_group_scale_test.ml
   already proved the same reduction for [Eval_direct.run] and
   [Stage_program.ground] respectively. Same fixture shape as
   stage_group_scale_test.ml (stacked layers=2, bidirectional, batch=2, so
   [scans = keys*layers*directions] is non-trivial), duplicated rather than
   shared across test files per this suite's own convention (each test file
   is a separate compilation unit). *)

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
    (Graph_builder.build ~name:"kernel_group_scale"
       ~outputs:(fun (out, hn, cn) -> [ out; hn; cn ])
       (graph ()))

let inputs g =
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
  | None -> invalid_arg "kernel_group_scale_test: no lstm node in the graph"

let max_abs_diff (Tensor.Tensor a) (Tensor.Tensor b) =
  Vec6.fold_coords a.Tensor.shape ~init:0. ~f:(fun acc coord ->
      let i = (Vec6.offset a.Tensor.shape coord :> int) in
      let c = Vec6.get coord Axis.C in
      Float.max acc
        (Float.abs
           (Payload.get_float a.Tensor.payload ~c ~i
           -. Payload.get_float b.Tensor.payload ~c ~i)))

let%expect_test
    "Kernel_eval.run shares one recurrence per batch key, agreeing with \
     Eval_direct" =
  let g = build_graph () in
  let bindings = inputs g in
  let direct_env =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.run g ~inputs:bindings)
  in
  let prog = Eval_symbolic.run g in
  let kernel =
    Err.or_raise ~pp_error:Kernel_adapt.pp_error
      (Kernel_adapt.of_stage_program prog)
  in
  let outs = lstm_node_outputs g in
  let region_counters =
    let counters = Region_execution.counters () in
    List.fold_left
      (fun m id -> Tensor_id.Map.add id counters m)
      Tensor_id.Map.empty outs
  in
  let bind_map =
    List.fold_left
      (fun m (id, t) -> Tensor_id.Map.add id t m)
      Tensor_id.Map.empty bindings
  in
  let result =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.run ~region_counters kernel ~bind:(fun id ->
           Tensor_id.Map.find_opt id bind_map))
  in
  let counters = Tensor_id.Map.find (List.hd outs) region_counters in
  List.iter
    (fun oid ->
      let direct = Tensor_id.Map.find oid direct_env in
      let kern = Tensor_id.Map.find oid result in
      Fmt.pr "output %a: max_abs_diff=%g@." Tensor_id.pp oid
        (max_abs_diff direct kern))
    outs;
  Fmt.pr "keys=%d scans=%d scan_updates=%d@." counters.Region_execution.keys
    counters.Region_execution.scans counters.Region_execution.scan_updates;
  [%expect
    {|
    output t19: max_abs_diff=0
    output t20: max_abs_diff=0
    output t21: max_abs_diff=0
    keys=2 scans=8 scan_updates=64 |}]

(* Non-vacuous per CLAUDE.md: temporarily replacing [Kernel_eval.execute]'s
   [Region_group.Run.Group] arm with one that ignores the shared run and
   instead calls [materialize] once per member (the old, per-value path) and
   re-running this test reproduced exactly the OLD, tripled counters --
   [keys=6 scans=24 scan_updates=192], 3x this test's real, shared numbers
   above -- confirming this test would catch a regression to independent
   per-value execution. Reverted; not left in the tree as a runtime toggle. *)
