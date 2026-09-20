(* [Me_kernel]: the value-graph export must carry the WHOLE Region program,
   not the emitter alone -- a LayerNorm kernel value has locals the emitter
   only references, and exporting the emitter by itself leaves dangling
   [?#N] placeholders where those locals belong. See .ai/region_compute_design.md
   and .ai/model_explorer_design.md. *)

module ME = Model_explorer

let limits = Me_limits.Limits.untrusted
let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:3

let layer_graph =
  Err.or_raise ~pp_error:Graph_builder.pp_error
    Graph_builder.(
      build ~name:"layer" ~outputs:(fun output -> [ output ])
      @@
      let* x = input ~shape ~name:"x" () in
      layer_norm { Norm.LayerNorm.dims = [ Axis.C ]; eps = 1e-5 } ~x ())

let layer_kernel =
  Err.or_raise ~pp_error:Kernel_adapt.pp_error
    (Region_kernel.of_graph layer_graph)

let canonical_origin =
  let producer = List.hd layer_graph.Graph_ir.Graph.nodes in
  let output = List.hd producer.Graph_ir.Node.outputs in
  fun id ->
    if Graph_ir.Tensor_id.equal id output then
      Some
        Me_kernel.Origin.
          { node = producer.id; output_slot = 0; namespace = "LayerNorm#g0" }
    else None

let exported =
  Err.or_raise ~pp_error:Me_kernel.pp_error
    (Me_kernel.kernel ~limits ~id:"k" ~origin:canonical_origin layer_kernel)

let stage_exported =
  Err.or_raise ~pp_error:Me_kernel.pp_error
    (Me_kernel.stage_program ~limits ~id:"s" ~origin:canonical_origin
       (Eval_symbolic.run layer_graph))

let value_node =
  let id =
    Core.Pretty.to_string Graph_ir.Tensor_id.pp
      (List.hd layer_kernel.Kernel.values).Kernel.Value.id
  in
  let has_value (n : ME.GraphNode.t) =
    List.exists
      (fun (m : ME.MetadataItem.t) ->
        List.exists
          (fun (kv : ME.KeyValue.t) ->
            kv.ME.KeyValue.key = "value" && kv.ME.KeyValue.value = id)
          m.ME.MetadataItem.attrs)
      (Option.value ~default:[] n.ME.GraphNode.outputsMetadata)
  in
  List.find has_value exported.ME.Graph.nodes

let attr key =
  List.find_map
    (fun (a : ME.NodeAttribute.t) ->
      if a.ME.NodeAttribute.key = key then
        match a.ME.NodeAttribute.value with
        | ME.NodeAttributeValue.Str s -> Some s
        | _ -> None
      else None)
    (Option.value ~default:[] value_node.ME.GraphNode.attrs)

let%expect_test
    "the exported LayerNorm body carries every local, not just the emitter" =
  let body = Option.get (attr "body") in
  Fmt.pr "no_dangling_locals=%b size=%s depth=%s@."
    (not (String.exists (fun c -> c = '?') body))
    (Option.get (attr "size"))
    (Option.get (attr "depth"));
  [%expect {| no_dangling_locals=true size=70 depth=6 |}]

let%expect_test "a symbolic stage preserves its canonical Native origin" =
  let stage =
    List.find
      (fun (n : ME.GraphNode.t) -> n.ME.GraphNode.label = "stage")
      stage_exported.ME.Graph.nodes
  in
  let origin key =
    List.find_map
      (fun (a : ME.NodeAttribute.t) ->
        if a.ME.NodeAttribute.key = key then
          match a.ME.NodeAttribute.value with
          | ME.NodeAttributeValue.Str value -> Some value
          | _ -> None
        else None)
      (Option.value ~default:[] stage.ME.GraphNode.attrs)
  in
  Fmt.pr "%s output %s@."
    (Option.get (origin "canonical_native_node"))
    (Option.get (origin "canonical_output_slot"));
  [%expect {| n0 output 0 |}]

let%expect_test "the matching kernel value stays in the canonical group" =
  let value =
    List.find
      (fun (n : ME.GraphNode.t) -> n.ME.GraphNode.label = "round_f32")
      exported.ME.Graph.nodes
  in
  Fmt.pr "%s@." value.ME.GraphNode.namespace;
  [%expect {| LayerNorm#g0 |}]

let%expect_test "the matching kernel value preserves its canonical origin" =
  Fmt.pr "%s output %s@."
    (Option.get (attr "canonical_native_node"))
    (Option.get (attr "canonical_output_slot"));
  [%expect {| n0 output 0 |}]

(* An int64 stage (here [To_copy(Long)]) is a value node too. Without one, the
   float stage that reads it names a source with no producer and the whole
   export fails -- a regression the ~100-model support sweep caught in
   mvitv2_tiny and volo_d1_224. *)
let long_chain =
  Err.or_raise ~pp_error:Graph_builder.pp_error
    Graph_builder.(
      build ~name:"long_chain" ~outputs:(fun output -> [ output ])
      @@
      let* x = input ~shape ~name:"x" () in
      let* l = to_copy Pointwise.To_copy.Long x in
      to_copy Pointwise.To_copy.Float l)

let body_of (n : ME.GraphNode.t) =
  List.find_map
    (fun (a : ME.NodeAttribute.t) ->
      match (a.ME.NodeAttribute.key, a.ME.NodeAttribute.value) with
      | "body", ME.NodeAttributeValue.Str s -> Some s
      | _ -> None)
    (Option.value ~default:[] n.ME.GraphNode.attrs)

let%expect_test "an int64 stage is exported as a value node, on both graphs" =
  let program = Eval_symbolic.run long_chain in
  Fmt.pr "stages: %d, stages_i64: %d@."
    (List.length program.Stage_program.stages)
    (List.length program.Stage_program.stages_i64);
  let stage_graph =
    Err.or_raise ~pp_error:Me_kernel.pp_error
      (Me_kernel.stage_program ~limits ~id:"s" ~origin:(fun _ -> None) program)
  in
  let kernel =
    Err.or_raise ~pp_error:Kernel_adapt.pp_error
      (Kernel_adapt.of_stage_program program)
  in
  let kernel_graph =
    Err.or_raise ~pp_error:Me_kernel.pp_error
      (Me_kernel.kernel ~limits ~id:"k" kernel)
  in
  let bodies g = List.filter_map body_of g.ME.Graph.nodes in
  Fmt.pr "stage bodies: %a@."
    Fmt.(list ~sep:(any " | ") string)
    (bodies stage_graph);
  Fmt.pr "kernel bodies: %a@."
    Fmt.(list ~sep:(any " | ") string)
    (bodies kernel_graph);
  [%expect
    {|
    stages: 1, stages_i64: 1
    stage bodies: region [N=singleton T=singleton D=singleton H=singleton W=singleton C=singleton]
      emit i64_to_float(t1[N,T,D,H,W,C]) | float_to_i64(t0[N,T,D,H,W,C])
    kernel bodies: region [N=singleton T=singleton D=singleton H=singleton W=singleton C=singleton]
      emit i64_to_float(t1[N,T,D,H,W,C]) | float_to_i64(t0[N,T,D,H,W,C]) |}]
