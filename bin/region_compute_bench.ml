open Graph_ir

let samples = 20
let warmup = 3

let median values =
  let values = Array.of_list values in
  Array.sort Float.compare values;
  values.(Array.length values / 2)

let words () =
  let stat = Gc.quick_stat () in
  stat.Gc.minor_words +. stat.Gc.major_words

let shape ~regions ~extent = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:regions ~c:extent

let input shape =
  Tensor.materialize shape (fun coord ->
      float_of_int
        ((100 * Dim.to_int (Vec6.get coord Axis.W))
        + Dim.to_int (Vec6.get coord Axis.C)
        + 1))

let graph name shape f =
  Err.or_raise ~pp_error:Graph_builder.pp_error
    Graph_builder.(
      build ~name ~outputs:(fun output -> [ output ])
      @@
      let* x = input ~shape ~name:"x" () in
      f x)

let graph4 shape f =
  let shape =
    Err.or_raise ~pp_error:Native4d.Shape4.pp_error
      (Native4d.Shape4.of_vec6 shape)
  in
  Err.or_raise ~pp_error:Native4d.Builder.pp_error
    Native4d.Builder.(
      build ~outputs:(fun output -> [ output ])
      @@
      let* x = input ~shape () in
      f x)

let kernel graph =
  Err.or_raise ~pp_error:Kernel_adapt.pp_error (Region_kernel.of_graph graph)

let run kernel graph input =
  Kernel_eval.run kernel ~bind:(fun id ->
      if List.mem id graph.Graph.inputs then Some input else None)
  |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  |> ignore

let run_direct graph input =
  Eval_direct.run graph ~inputs:[ (List.hd graph.Graph.inputs, input) ]
  |> Err.or_raise ~pp_error:Eval_direct.pp_error
  |> ignore

let run_direct4 graph input =
  Native4d.Eval_direct4.run graph
    ~inputs:[ (List.hd graph.Native4d.Graph.Graph.inputs, input) ]
  |> Err.or_raise ~pp_error:Native4d.Eval_direct4.pp_error
  |> ignore

let measure kernel graph input =
  for _ = 1 to warmup do
    run kernel graph input
  done;
  List.init samples (fun _ ->
      Gc.full_major ();
      let before = words () and started = Unix.gettimeofday () in
      run kernel graph input;
      ((Unix.gettimeofday () -. started) *. 1000., words () -. before))
  |> List.split

let measure_direct graph input =
  for _ = 1 to warmup do
    run_direct graph input
  done;
  List.init samples (fun _ ->
      Gc.full_major ();
      let before = words () and started = Unix.gettimeofday () in
      run_direct graph input;
      ((Unix.gettimeofday () -. started) *. 1000., words () -. before))
  |> List.split

let measure_direct4 graph input =
  for _ = 1 to warmup do
    run_direct4 graph input
  done;
  List.init samples (fun _ ->
      Gc.full_major ();
      let before = words () and started = Unix.gettimeofday () in
      run_direct4 graph input;
      ((Unix.gettimeofday () -. started) *. 1000., words () -. before))
  |> List.split

let counters kernel graph input =
  let counters = Region_execution.counters () in
  let by_value =
    List.fold_left
      (fun map (value : Kernel.Value.t) ->
        Tensor_id.Map.add value.id counters map)
      Tensor_id.Map.empty kernel.Kernel.values
  in
  Kernel_eval.run ~region_counters:by_value kernel ~bind:(fun id ->
      if List.mem id graph.Graph.inputs then Some input else None)
  |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  |> ignore;
  counters

let direct_counters graph input =
  let counters = Region_execution.counters () in
  let output = List.hd graph.Graph.outputs in
  Eval_direct.run
    ~region_counters:(Tensor_id.Map.singleton output counters)
    graph
    ~inputs:[ (List.hd graph.Graph.inputs, input) ]
  |> Err.or_raise ~pp_error:Eval_direct.pp_error
  |> ignore;
  counters

let direct4_counters graph input =
  let counters = Region_execution.counters () in
  let output = List.hd graph.Native4d.Graph.Graph.outputs in
  Native4d.Eval_direct4.run
    ~region_counters:(Tensor_id.Map.singleton output counters)
    graph
    ~inputs:[ (List.hd graph.Native4d.Graph.Graph.inputs, input) ]
  |> Err.or_raise ~pp_error:Native4d.Eval_direct4.pp_error
  |> ignore;
  counters

let report name ~regions ~extent graph graph4 =
  let input = input (shape ~regions ~extent) in
  let kernel = kernel graph in
  let kernel_time, kernel_words = measure kernel graph input in
  let direct_time, direct_words = measure_direct graph input in
  let direct4_time, direct4_words = measure_direct4 graph4 input in
  let counters = counters kernel graph input
  and direct_counters = direct_counters graph input
  and direct4_counters = direct4_counters graph4 input in
  Printf.printf
    "%s regions=%d extent=%d samples=%d kernel_region_ms=%.3f direct_ms=%.3f \
     direct4_ms=%.3f kernel_region_words=%.3f direct_words=%.3f \
     direct4_words=%.3f keys=%d locals=%d emitters=%d loads=%d reductions=%d \
     direct_keys=%d direct_locals=%d direct_emitters=%d direct_loads=%d \
     direct_reductions=%d direct4_keys=%d direct4_locals=%d \
     direct4_emitters=%d direct4_loads=%d direct4_reductions=%d\n\
     %!"
    name regions extent samples (median kernel_time) (median direct_time)
    (median direct4_time) (median kernel_words) (median direct_words)
    (median direct4_words) counters.keys counters.locals counters.emitters
    counters.loads counters.reductions direct_counters.keys
    direct_counters.locals direct_counters.emitters direct_counters.loads
    direct_counters.reductions direct4_counters.keys direct4_counters.locals
    direct4_counters.emitters direct4_counters.loads direct4_counters.reductions

let () =
  List.iter
    (fun (regions, extent) ->
      let shape = shape ~regions ~extent in
      report "rms" ~regions ~extent
        (graph "rms" shape (fun x ->
             Graph_builder.rms_norm
               { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-5 }
               ~x ()))
        (graph4 shape (fun x ->
             Native4d.Builder.rms_norm
               {
                 Native4d.Ops4.Rms_norm.dims = [ Native4d.Axis4.C ];
                 eps = 1e-5;
               }
               ~x ()));
      report "layer" ~regions ~extent
        (graph "layer" shape (fun x ->
             Graph_builder.layer_norm
               { Norm.LayerNorm.dims = [ Axis.C ]; eps = 1e-5 }
               ~x ()))
        (graph4 shape (fun x ->
             Native4d.Builder.layer_norm4
               {
                 Native4d.Ops4.Layer_norm.dims = [ Native4d.Axis4.C ];
                 eps = 1e-5;
               }
               ~x ()));
      report "softmax" ~regions ~extent
        (graph "softmax" shape (fun x ->
             Graph_builder.softmax { Reduce.Softmax.axis = Axis.C } x))
        (graph4 shape (fun x ->
             Native4d.Builder.softmax4
               { Native4d.Ops4.Softmax4.axis = Native4d.Axis4.C }
               x)))
    (* Fix C and sweep W (the number of whole-C region keys), including
       R > K, then retain the original extent sweep at R = 4. *)
    [ (1, 32); (4, 32); (16, 32); (64, 32); (4, 8); (4, 64) ]
