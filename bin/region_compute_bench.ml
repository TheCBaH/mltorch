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

let kernels graph =
  let pixel =
    Err.or_raise ~pp_error:Kernel_adapt.pp_error
      (Kernel_adapt.of_stage_program (Eval_symbolic.run graph))
  and region =
    Err.or_raise ~pp_error:Kernel_adapt.pp_error (Region_kernel.of_graph graph)
  in
  (pixel, region)

let run kernel graph input =
  Kernel_eval.run kernel ~bind:(fun id ->
      if List.mem id graph.Graph.inputs then Some input else None)
  |> Err.or_raise ~pp_error:Kernel_eval.pp_error
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

let report name ~regions ~extent graph =
  let input = input (shape ~regions ~extent) in
  let pixel, region = kernels graph in
  let pixel_time, pixel_words = measure pixel graph input in
  let region_time, region_words = measure region graph input in
  let counters = counters region graph input in
  Printf.printf
    "%s regions=%d extent=%d samples=%d pixel_ms=%.3f region_ms=%.3f \
     pixel_words=%.3f region_words=%.3f keys=%d locals=%d emitters=%d loads=%d \
     reductions=%d\n\
     %!"
    name regions extent samples (median pixel_time) (median region_time)
    (median pixel_words) (median region_words) counters.keys counters.locals
    counters.emitters counters.loads counters.reductions

let () =
  List.iter
    (fun (regions, extent) ->
      let shape = shape ~regions ~extent in
      report "rms" ~regions ~extent
        (graph "rms" shape (fun x ->
             Graph_builder.rms_norm
               { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-5 }
               ~x ()));
      report "layer" ~regions ~extent
        (graph "layer" shape (fun x ->
             Graph_builder.layer_norm
               { Norm.LayerNorm.dims = [ Axis.C ]; eps = 1e-5 }
               ~x ()));
      report "softmax" ~regions ~extent
        (graph "softmax" shape (fun x ->
             Graph_builder.softmax { Reduce.Softmax.axis = Axis.C } x)))
    (* Fix C and sweep W (the number of whole-C region keys), including
       R > K, then retain the original extent sweep at R = 4. *)
    [ (1, 32); (4, 32); (16, 32); (64, 32); (4, 8); (4, 64) ]
