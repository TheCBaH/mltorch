(* [Me_kernel]: the value-graph export must carry the WHOLE Region program,
   not the emitter alone -- a LayerNorm kernel value has locals the emitter
   only references, and exporting the emitter by itself leaves dangling
   [?#N] placeholders where those locals belong. See .ai/region_compute_design.md
   and .ai/model_explorer_design.md. *)

module ME = Model_explorer

let limits = Me_limits.Limits.untrusted
let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:3

let layer_kernel =
  let graph =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      Graph_builder.(
        build ~name:"layer" ~outputs:(fun output -> [ output ])
        @@
        let* x = input ~shape ~name:"x" () in
        layer_norm { Norm.LayerNorm.dims = [ Axis.C ]; eps = 1e-5 } ~x ())
  in
  Err.or_raise ~pp_error:Kernel_adapt.pp_error (Region_kernel.of_graph graph)

let exported =
  Err.or_raise ~pp_error:Me_kernel.pp_error
    (Me_kernel.kernel ~limits ~id:"k" layer_kernel)

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
