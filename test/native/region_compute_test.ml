open Graph_ir

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:3

let input =
  Tensor.materialize shape (fun coord ->
      float_of_int
        ((10 * Dim.to_int (Vec6.get coord Axis.W))
        + Dim.to_int (Vec6.get coord Axis.C)
        + 1))

let graph name f =
  Err.or_raise ~pp_error:Graph_builder.pp_error
    Graph_builder.(
      build ~name ~outputs:(fun output -> [ output ])
      @@
      let* x = input ~shape ~name:"x" () in
      f x)

let run name graph =
  let output_id = List.hd graph.Graph.outputs in
  let kernel =
    Err.or_raise ~pp_error:Kernel_adapt.pp_error (Region_kernel.of_graph graph)
  in
  let bind id = if List.mem id graph.Graph.inputs then Some input else None in
  let actual =
    Tensor_id.Map.find output_id
      (Err.or_raise ~pp_error:Kernel_eval.pp_error
         (Kernel_eval.run kernel ~bind))
  in
  let expected =
    match name with
    | "rms" ->
        let module C = Norm.RmsNorm.Legacy_pixel (Direct) in
        let weight =
          Tensor.materialize
            (Norm_shared.normalized_shape ~x_shape:shape ~dims:Axis.[ C ])
            (fun _ -> 1.)
        in
        Schedule.evaluate shape
          (C.pixel
             { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-5 }
             ~x_shape:shape ~x:input ~weight)
    | "layer" ->
        let module C = Norm.LayerNorm.Legacy_pixel (Direct) in
        let affine =
          Norm_shared.normalized_shape ~x_shape:shape ~dims:Axis.[ C ]
        in
        let weight = Tensor.materialize affine (fun _ -> 1.)
        and bias = Tensor.materialize affine (fun _ -> 0.) in
        Schedule.evaluate shape
          (C.pixel
             { Norm.LayerNorm.dims = [ Axis.C ]; eps = 1e-5 }
             ~x_shape:shape ~x:input ~weight ~bias)
    | "softmax" ->
        let module C = Reduce.Softmax.Legacy_pixel (Direct) in
        Schedule.evaluate shape
          (C.pixel { Reduce.Softmax.axis = Axis.C } ~x_shape:shape ~x:input)
    | _ -> assert false
  in
  let region =
    List.for_all
      (fun (value : Kernel.Value.t) ->
        Region_program.pixel_expression value.computation = None)
      kernel.Kernel.values
  in
  Fmt.pr "%s: region=%b legacy_bits=%b@." name region
    (Tensor.equal_bits actual expected)

let count name graph =
  let kernel =
    Err.or_raise ~pp_error:Kernel_adapt.pp_error (Region_kernel.of_graph graph)
  in
  let counters = Region_execution.counters () in
  let counter_map =
    List.fold_left
      (fun map (value : Kernel.Value.t) ->
        Tensor_id.Map.add value.id counters map)
      Tensor_id.Map.empty kernel.Kernel.values
  in
  let bind id = if List.mem id graph.Graph.inputs then Some input else None in
  ignore
    (Err.or_raise ~pp_error:Kernel_eval.pp_error
       (Kernel_eval.run ~region_counters:counter_map kernel ~bind));
  Fmt.pr "%s: keys=%d locals=%d emitters=%d loads=%d reductions=%d@." name
    counters.keys counters.locals counters.emitters counters.loads
    counters.reductions

let count_direct name graph =
  let output = List.hd graph.Graph.outputs in
  let counters = Region_execution.counters () in
  ignore
    (Err.or_raise ~pp_error:Eval_direct.pp_error
       (Eval_direct.run
          ~region_counters:(Tensor_id.Map.singleton output counters)
          ~inputs:[ (List.hd graph.Graph.inputs, input) ]
          graph));
  Fmt.pr "%s: keys=%d locals=%d emitters=%d loads=%d reductions=%d@." name
    counters.keys counters.locals counters.emitters counters.loads
    counters.reductions

let%expect_test "Regionized normalization and softmax reconstruct and execute" =
  let rms =
    graph "rms" (fun x ->
        Graph_builder.rms_norm
          { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-5 }
          ~x ())
  in
  let layer =
    graph "layer" (fun x ->
        Graph_builder.layer_norm
          { Norm.LayerNorm.dims = [ Axis.C ]; eps = 1e-5 }
          ~x ())
  in
  let softmax =
    graph "softmax" (fun x ->
        Graph_builder.softmax { Reduce.Softmax.axis = Axis.C } x)
  in
  List.iter
    (fun (name, graph) -> run name graph)
    [ ("rms", rms); ("layer", layer); ("softmax", softmax) ];
  [%expect
    {|
    rms: region=true legacy_bits=true
    layer: region=true legacy_bits=true
    softmax: region=true legacy_bits=true |}]

let%expect_test "Symbolic stages carry the authoritative Region program" =
  let graph =
    graph "symbolic_region" (fun x ->
        Graph_builder.rms_norm
          { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-5 }
          ~x ())
  in
  let symbolic = Eval_symbolic.run graph in
  let stage = List.hd symbolic.Stage_program.stages in
  let kernel =
    Err.or_raise ~pp_error:Kernel_adapt.pp_error
      (Kernel_adapt.of_stage_program symbolic)
  in
  let carried = stage.Stage_program.Stage.computation in
  let received = (List.hd kernel.Kernel.values).Kernel.Value.computation in
  Fmt.pr "region=%b same_object=%b@."
    (Region_program.pixel_expression carried = None)
    (carried == received);
  [%expect {| region=true same_object=true |}]

let%expect_test "Authored Regions reconstruct their legacy scalar oracles" =
  let affine = Norm_shared.normalized_shape ~x_shape:shape ~dims:Axis.[ C ] in
  let rms =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      Graph_builder.(
        build ~name:"reconstruct_rms" ~outputs:(fun output -> [ output ])
        @@
        let* x = input ~shape ~name:"x" () in
        let* weight = input ~shape:affine ~name:"weight" () in
        rms_norm { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-5 } ~x ~weight ())
  in
  let layer =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      Graph_builder.(
        build ~name:"reconstruct_layer" ~outputs:(fun output -> [ output ])
        @@
        let* x = input ~shape ~name:"x" () in
        let* weight = input ~shape:affine ~name:"weight" () in
        let* bias = input ~shape:affine ~name:"bias" () in
        layer_norm
          { Norm.LayerNorm.dims = [ Axis.C ]; eps = 1e-5 }
          ~x ~weight ~bias ())
  in
  let softmax =
    graph "reconstruct_softmax" (fun x ->
        Graph_builder.softmax { Reduce.Softmax.axis = Axis.C } x)
  in
  let reconstruct graph pixel =
    let symbolic = Eval_symbolic.run graph in
    let program = (List.hd symbolic.Stage_program.stages).computation in
    Err.or_raise ~pp_error:Region_program.pp_error
      (Region_program.reconstructs ~max_size:Kernel.Limits.default.max_size
         ~max_depth:Kernel.Limits.default.max_depth ~pixel program)
  in
  let rms_pixel =
    let module C = Norm.RmsNorm.Legacy_pixel (Symbolic) in
    let x, weight =
      match rms.Graph.inputs with
      | [ x; weight ] ->
          ( Tensor_id.Map.find x rms.Graph.tensors,
            Tensor_id.Map.find weight rms.Graph.tensors )
      | _ -> assert false
    in
    Expr.Builder.run
      (C.pixel
         { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-5 }
         ~x_shape:shape ~x ~weight Symbolic.out_vec)
  in
  let layer_pixel =
    let module C = Norm.LayerNorm.Legacy_pixel (Symbolic) in
    let x, weight, bias =
      match layer.Graph.inputs with
      | [ x; weight; bias ] ->
          ( Tensor_id.Map.find x layer.Graph.tensors,
            Tensor_id.Map.find weight layer.Graph.tensors,
            Tensor_id.Map.find bias layer.Graph.tensors )
      | _ -> assert false
    in
    Expr.Builder.run
      (C.pixel
         { Norm.LayerNorm.dims = [ Axis.C ]; eps = 1e-5 }
         ~x_shape:shape ~x ~weight ~bias Symbolic.out_vec)
  in
  let softmax_pixel =
    let module C = Reduce.Softmax.Legacy_pixel (Symbolic) in
    let x = List.hd softmax.Graph.inputs in
    let x = Tensor_id.Map.find x softmax.Graph.tensors in
    Expr.Builder.run
      (C.pixel
         { Reduce.Softmax.axis = Axis.C }
         ~x_shape:shape ~x Symbolic.out_vec)
  in
  Fmt.pr "rms=%b layer=%b softmax=%b@."
    (reconstruct rms rms_pixel)
    (reconstruct layer layer_pixel)
    (reconstruct softmax softmax_pixel);
  [%expect {| rms=true layer=true softmax=true |}]

let%expect_test "transform grounding reads an authored Stage structurally" =
  let graph =
    graph "ground_region" (fun x ->
        Graph_builder.rms_norm
          { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-5 }
          ~x ())
  in
  let symbolic = Eval_symbolic.run graph in
  let stage = List.hd symbolic.Stage_program.stages in
  let program = { symbolic with stages = [ stage ] } in
  let env = Ground_eval.Env.of_program program ~side:`Src in
  let result =
    Ground_eval.at env stage.id (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0)
  in
  let reads_input =
    Result.fold
      ~ok:(fun value ->
        not (Ground_expr.Cell.Set.is_empty (Ground_expr.cells value)))
      ~error:(fun _ -> false)
      result
  in
  Fmt.pr "reads_input=%b@." reads_input;
  [%expect {| reads_input=true |}]

let%expect_test "Regionized reductions are linear in each output region" =
  let rms =
    graph "rms_count" (fun x ->
        Graph_builder.rms_norm
          { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-5 }
          ~x ())
  in
  let layer =
    graph "layer_count" (fun x ->
        Graph_builder.layer_norm
          { Norm.LayerNorm.dims = [ Axis.C ]; eps = 1e-5 }
          ~x ())
  in
  let softmax =
    graph "softmax_count" (fun x ->
        Graph_builder.softmax { Reduce.Softmax.axis = Axis.C } x)
  in
  List.iter
    (fun (name, graph) -> count name graph)
    [ ("rms", rms); ("layer", layer); ("softmax", softmax) ];
  [%expect
    {|
    rms: keys=2 locals=4 emitters=6 loads=24 reductions=6
    layer: keys=2 locals=8 emitters=6 loads=36 reductions=12
    softmax: keys=2 locals=4 emitters=6 loads=18 reductions=12 |}]

let%expect_test "Native Direct materializes authored Regions once per key" =
  let rms =
    graph "rms_direct" (fun x ->
        Graph_builder.rms_norm
          { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-5 }
          ~x ())
  in
  let layer =
    graph "layer_direct" (fun x ->
        Graph_builder.layer_norm
          { Norm.LayerNorm.dims = [ Axis.C ]; eps = 1e-5 }
          ~x ())
  in
  let softmax =
    graph "softmax_direct" (fun x ->
        Graph_builder.softmax { Reduce.Softmax.axis = Axis.C } x)
  in
  List.iter
    (fun (name, graph) -> count_direct name graph)
    [ ("rms", rms); ("layer", layer); ("softmax", softmax) ];
  let tight =
    Err.or_raise ~pp_error:Kernel.Limits.pp_error
      (Kernel.Limits.create ~max_size:1 ~max_depth:128 ~max_values:16
         ~max_dep_depth:16 ~max_inputs:16 ~max_outputs:16
         ~max_extent:0x7FFF_FFFFL ~max_numel:0x7FFF_FFFFL)
  in
  let tight_result =
    Eval_direct.run ~limits:tight
      ~inputs:[ (List.hd rms.Graph.inputs, input) ]
      rms
  in
  let tight =
    match tight_result with
    | Ok _ -> "ok"
    | Error error ->
        Format.asprintf "%a" Eval_direct.pp_error (Err.Error.kind error)
  in
  Fmt.pr "tight=%s@." tight;
  [%expect
    {|
    rms: keys=2 locals=4 emitters=6 loads=24 reductions=6
    layer: keys=2 locals=8 emitters=6 loads=36 reductions=12
    softmax: keys=2 locals=4 emitters=6 loads=18 reductions=12
    tight=invalid region program |}]

(* This is deliberately a structural trace matrix rather than another dense
   tensor comparison.  [Region_trace] independently checks every program's
   output ownership, which detects a duplicate store paired with an unvisited
   cell even when a final tensor comparison could not.  [T] and [W] are extent
   one so the same numel is exercised under distinct partitions. *)
let%expect_test "Native authored Region trace matrix" =
  let matrix_shape = Vec6.shape ~n:2 ~t:1 ~d:2 ~h:2 ~w:1 ~c:3 in
  let build name f =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      Graph_builder.(
        build ~name ~outputs:(fun output -> [ output ])
        @@
        let* x = input ~shape:matrix_shape ~name:"x" () in
        f x)
  in
  let rms name dims =
    build name (fun x ->
        Graph_builder.rms_norm { Norm.RmsNorm.dims; eps = 1e-5 } ~x ())
  in
  let layer name dims ~weight ~bias =
    let params : Norm.LayerNorm.params = { dims; eps = 1e-5 } in
    let affine_shape = Norm.normalized_shape ~x_shape:matrix_shape ~dims in
    Err.or_raise ~pp_error:Graph_builder.pp_error
      Graph_builder.(
        build ~name ~outputs:(fun output -> [ output ])
        @@
        let* x = input ~shape:matrix_shape ~name:"x" () in
        match (weight, bias) with
        | false, false -> layer_norm params ~x ()
        | true, false ->
            let* weight = input ~shape:affine_shape ~name:"weight" () in
            layer_norm params ~x ~weight ()
        | false, true ->
            let* bias = input ~shape:affine_shape ~name:"bias" () in
            layer_norm params ~x ~bias ()
        | true, true ->
            let* weight = input ~shape:affine_shape ~name:"weight" () in
            let* bias = input ~shape:affine_shape ~name:"bias" () in
            layer_norm params ~x ~weight ~bias ())
  in
  let softmax name axis =
    build name (fun x -> Graph_builder.softmax { Reduce.Softmax.axis } x)
  in
  let trace name graph =
    let symbolic = Eval_symbolic.run graph in
    let stage = List.hd symbolic.Stage_program.stages in
    let program = Stage_program.Stage.computation stage in
    let trace =
      Err.or_raise ~pp_error:Region_trace.pp_error
        (Region_trace.collect program ~output_shape:stage.sg.shape)
    in
    let coverage = trace.Region_trace.coverage in
    assert (coverage.visited = coverage.total);
    assert (coverage.duplicates = 0);
    assert (coverage.missing = 0);
    Fmt.pr "%s keys=%d total=%d@." name coverage.keys coverage.total
  in
  let axes =
    [
      ("n", Axis.N);
      ("t", Axis.T);
      ("d", Axis.D);
      ("h", Axis.H);
      ("w", Axis.W);
      ("c", Axis.C);
    ]
  in
  List.iter
    (fun (name, axis) -> trace ("rms_" ^ name) (rms ("rms_" ^ name) [ axis ]))
    axes;
  List.iter
    (fun (name, dims) -> trace ("rms_" ^ name) (rms ("rms_" ^ name) dims))
    [ ("n_c", [ Axis.N; Axis.C ]); ("t_w", [ Axis.T; Axis.W ]) ];
  List.iter
    (fun (name, dims) ->
      List.iter
        (fun (affine, weight, bias) ->
          trace
            ("layer_" ^ name ^ "_" ^ affine)
            (layer ("layer_" ^ name ^ "_" ^ affine) dims ~weight ~bias))
        [
          ("none", false, false);
          ("weight", true, false);
          ("bias", false, true);
          ("both", true, true);
        ])
    [
      ("n", [ Axis.N ]);
      ("t", [ Axis.T ]);
      ("d", [ Axis.D ]);
      ("h", [ Axis.H ]);
      ("w", [ Axis.W ]);
      ("c", [ Axis.C ]);
      ("n_c", [ Axis.N; Axis.C ]);
      ("t_w", [ Axis.T; Axis.W ]);
    ];
  List.iter
    (fun (name, axis) ->
      trace ("softmax_" ^ name) (softmax ("softmax_" ^ name) axis))
    axes;
  [%expect
    {|
    rms_n keys=12 total=24
    rms_t keys=24 total=24
    rms_d keys=12 total=24
    rms_h keys=12 total=24
    rms_w keys=24 total=24
    rms_c keys=8 total=24
    rms_n_c keys=4 total=24
    rms_t_w keys=24 total=24
    layer_n_none keys=12 total=24
    layer_n_weight keys=12 total=24
    layer_n_bias keys=12 total=24
    layer_n_both keys=12 total=24
    layer_t_none keys=24 total=24
    layer_t_weight keys=24 total=24
    layer_t_bias keys=24 total=24
    layer_t_both keys=24 total=24
    layer_d_none keys=12 total=24
    layer_d_weight keys=12 total=24
    layer_d_bias keys=12 total=24
    layer_d_both keys=12 total=24
    layer_h_none keys=12 total=24
    layer_h_weight keys=12 total=24
    layer_h_bias keys=12 total=24
    layer_h_both keys=12 total=24
    layer_w_none keys=24 total=24
    layer_w_weight keys=24 total=24
    layer_w_bias keys=24 total=24
    layer_w_both keys=24 total=24
    layer_c_none keys=8 total=24
    layer_c_weight keys=8 total=24
    layer_c_bias keys=8 total=24
    layer_c_both keys=8 total=24
    layer_n_c_none keys=4 total=24
    layer_n_c_weight keys=4 total=24
    layer_n_c_bias keys=4 total=24
    layer_n_c_both keys=4 total=24
    layer_t_w_none keys=24 total=24
    layer_t_w_weight keys=24 total=24
    layer_t_w_bias keys=24 total=24
    layer_t_w_both keys=24 total=24
    softmax_n keys=12 total=24
    softmax_t keys=24 total=24
    softmax_d keys=12 total=24
    softmax_h keys=12 total=24
    softmax_w keys=24 total=24
    softmax_c keys=8 total=24 |}]

let%expect_test "Region construction honors the graph-boundary contract" =
  let signature id shape =
    Tensor_sig.create ~id:(Tensor_id.of_int id) ~name:"" ~shape
      ~fmt:(Payload.Fmt Payload.F32) ()
  in
  let x = signature 10 shape in
  let op =
    Graph_ir.Rms_norm
      {
        Norm.RmsNorm.params = { dims = [ Axis.C ]; eps = 1e-5 };
        x = x.Tensor_sig.id;
        weight = None;
      }
  in
  let build ~limits ~output ~output_shape ~operand =
    Region_computation.program ~limits ~op ~output ~output_shape ~operand
      ~fill:(fun _role _value shape -> signature 11 shape)
  in
  let show = function
    | Ok _ -> "ok"
    | Error error ->
        Format.asprintf "%a" Region_computation.pp_error (Err.Error.kind error)
  in
  let tight =
    Err.or_raise ~pp_error:Kernel.Limits.pp_error
      (Kernel.Limits.create ~max_size:1 ~max_depth:128 ~max_values:16
         ~max_dep_depth:16 ~max_inputs:16 ~max_outputs:16
         ~max_extent:0x7FFF_FFFFL ~max_numel:0x7FFF_FFFFL)
  in
  let expanded =
    Err.or_raise ~pp_error:Kernel.Limits.pp_error
      (Kernel.Limits.create ~max_size:256 ~max_depth:128 ~max_values:256
         ~max_dep_depth:256 ~max_inputs:256 ~max_outputs:256
         ~max_extent:0x7FFF_FFFFL ~max_numel:0x7FFF_FFFFL)
  in
  let present id =
    if Tensor_id.equal id x.Tensor_sig.id then Some x else None
  in
  Fmt.pr "default=%s expanded=%s tight=%s ordinal=%s shape=%s missing=%s@."
    (show
       (build ~limits:Kernel.Limits.default ~output:0 ~output_shape:shape
          ~operand:present))
    (show
       (build ~limits:expanded ~output:0 ~output_shape:shape ~operand:present))
    (show (build ~limits:tight ~output:0 ~output_shape:shape ~operand:present))
    (show
       (build ~limits:Kernel.Limits.default ~output:1 ~output_shape:shape
          ~operand:present))
    (show
       (build ~limits:Kernel.Limits.default ~output:0
          ~output_shape:(Vec6.set shape Axis.C (Dim.extent 4))
          ~operand:present))
    (show
       (build ~limits:Kernel.Limits.default ~output:0 ~output_shape:shape
          ~operand:(fun _ -> None)));
  [%expect
    {|
    default=ok expanded=ok tight=invalid region program ordinal=unsupported output ordinal 1 shape=output shape does not match the input missing=missing operand t10 |}]

let%expect_test "Region executor traverses keys, locals, and emitters once" =
  let partition =
    Err.or_raise ~pp_error:Region_partition.pp_error
      (Region_partition.of_whole_axes [ Axis.C ])
  in
  let program =
    Err.or_raise ~pp_error:Region_program.pp_error
      (Region_program.Builder.run
         (Region_program.Builder.scalar
            (Expr.Value.load
               (Expr_bridge.source_of_id (Tensor_id.of_int 0))
               (Expr.Coord.make ~n:(Expr.Index.output Axis.N)
                  ~t:(Expr.Index.output Axis.T) ~d:(Expr.Index.output Axis.D)
                  ~h:(Expr.Index.output Axis.H) ~w:(Expr.Index.output Axis.W)
                  ~c:Expr.Index.zero))
            (fun local ->
              Region_program.Builder.finish ~max_size:32 ~max_depth:16
                ~partition
                ~output:
                  (Expr.Value.add local
                     (Expr.Value.load
                        (Expr_bridge.source_of_id (Tensor_id.of_int 0))
                        (Expr_bridge.coord_of_vec6 Symbolic.out_vec))))))
  in
  let lowered =
    match Region_execution.lower program with
    | Region_execution.Pixel_loop _ -> assert false
    | Region_execution.Region_loop lowered -> lowered
  in
  let counters = Region_execution.counters () in
  let env =
    {
      Expr.Eval.Env.load =
        (fun _ coord ->
          Err.return
            (Tensor.read input
               (Vec6.map Dim.index (Expr_bridge.vec6_of_coord coord))));
      load_index = (fun _ _ -> assert false);
    }
  in
  let output =
    Err.or_raise ~pp_error:Region_eval.pp_error
      (Region_execution.materialize ~counters lowered ~output_shape:shape ~env)
  in
  Fmt.pr
    "cells=%g,%g,%g,%g,%g,%g keys=%d locals=%d emitters=%d loads=%d \
     reductions=%d@."
    (Tensor.read output Vec6.origin)
    (Tensor.read output (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:1))
    (Tensor.read output (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:2))
    (Tensor.read output (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:1 ~c:0))
    (Tensor.read output (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:1 ~c:1))
    (Tensor.read output (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:1 ~c:2))
    counters.keys counters.locals counters.emitters counters.loads
    counters.reductions;
  [%expect
    {|
    cells=2,3,4,22,23,24 keys=2 locals=2 emitters=6 loads=8 reductions=0 |}]

(* [fresh_synthetic_ids] used to be recomputed inside [eval_node], folding
   [g.Graph.tensors] once per node — quadratic in graph size for a Region-free
   graph that never uses the result. A relu chain has no Region op, so any
   time spent here is pure per-node overhead: at 4x the nodes, a linear
   evaluator moves at roughly 4x the time, a quadratic one at roughly 16x. *)
let relu_chain n =
  Err.or_raise ~pp_error:Graph_builder.pp_error
    Graph_builder.(
      build ~name:"chain" ~outputs:(fun output -> [ output ])
      @@
      let* x = input ~shape ~name:"x" () in
      let rec go i x =
        if i = 0 then return x
        else
          let* x = relu x in
          go (i - 1) x
      in
      go n x)

let time_direct graph =
  let run () =
    ignore
      (Err.or_raise ~pp_error:Eval_direct.pp_error
         (Eval_direct.run ~inputs:[ (List.hd graph.Graph.inputs, input) ] graph))
  in
  for _ = 1 to 3 do
    run ()
  done;
  let samples =
    List.init 5 (fun _ ->
        let started = Sys.time () in
        run ();
        Sys.time () -. started)
  in
  let sorted = List.sort Float.compare samples in
  List.nth sorted (List.length sorted / 2)

let%expect_test
    "Native Direct is linear, not quadratic, in Region-free graph size" =
  let small = time_direct (relu_chain 100) in
  let large = time_direct (relu_chain 400) in
  (* Growing the node count 4x should move the time by roughly 4x under a
     linear evaluator and roughly 16x under a quadratic one; 10x sits between
     the two with headroom for measurement noise. *)
  let linear = large < small *. 10. in
  Fmt.pr "scaling: %s@." (if linear then "linear" else "quadratic");
  [%expect {| scaling: linear |}]
