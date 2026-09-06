(* PT2 graph orchestration: [lower]/[lower_archive] allocate sources, bind
   lowered outputs, and attach provenance. Per-operator lowering lives in the
   compute and shape dispatch families; captured-tensor materialisation is in
   [Native_interp_tensor]. *)

open Pytorch_types
open Schema_runtime
module Tensor_id = Graph_ir.Tensor_id
module Node_id = Graph_ir.Node_id
open Native_interp_decode

let lower program =
  Err.Escape.with_escape @@ fun esc ->
  Err.Escape.or_throw esc
  @@
  let graph : Pytorch_types.Graph.t =
    program.ExportedProgram.graph_module.graph
  in
  let sign = program.ExportedProgram.graph_module.GraphModule.signature in
  let source_specs =
    List.map
      (function
        | InputSpec.User_input { UserInputSpec.arg = Argument.Tensor a } ->
            (`Input, a.TensorArgument.name, None)
        | InputSpec.Parameter p ->
            (`Constant, p.arg.TensorArgument.name, Some p.parameter_name)
        | InputSpec.Buffer p ->
            (`Constant, p.arg.TensorArgument.name, Some p.buffer_name)
        | InputSpec.Tensor_constant p ->
            (`Constant, p.arg.TensorArgument.name, Some p.tensor_constant_name)
        | _ -> Err.Escape.throw esc (`Unsupported_input `Non_tensor))
      sign.GraphSignature.input_specs
  in
  (* Every SSA name any node READS, plus the graph's own outputs -- the
     complement of "dead". Computed once and lazily, so a graph containing no
     [native_layer_norm] never pays for it, and one containing 49 of them pays
     once rather than 49 times: the alternative, scanning the remaining nodes
     from inside the arm, is quadratic in the node count on exactly the models
     this row exists for.

     A node's own outputs are not reads, so they are not collected; only
     [inputs] and [graph.outputs] are. *)
  let reads =
    lazy
      (let add acc (a : Argument.t) =
         match a with
         | Argument.Tensor t -> String_map.add t.TensorArgument.name () acc
         | Argument.Optional_tensor (OptionalTensorArgument.Tensor t) ->
             String_map.add t.TensorArgument.name () acc
         | Argument.Tensors ts ->
             List.fold_left
               (fun acc (t : TensorArgument.t) ->
                 String_map.add t.TensorArgument.name () acc)
               acc ts
         | Argument.Optional_tensors ts ->
             List.fold_left
               (fun acc (t : OptionalTensorArgument.t) ->
                 match t with
                 | OptionalTensorArgument.Tensor (t : TensorArgument.t) ->
                     String_map.add t.name () acc
                 | OptionalTensorArgument.None _ -> acc)
               acc ts
         | _ -> acc
       in
       List.fold_left
         (fun acc (n : Node.t) ->
           List.fold_left
             (fun acc (i : NamedArgument.t) -> add acc i.NamedArgument.arg)
             acc n.Node.inputs)
         (List.fold_left add String_map.empty graph.outputs)
         graph.nodes)
  in
  let constant_names =
    List.fold_left
      (fun acc (kind, name, _) ->
        match kind with
        | `Constant -> String_map.add name () acc
        | `Input -> acc)
      String_map.empty source_specs
  in
  let tensor_origins = ref Tensor_id.Map.empty in
  let captured_targets = ref Tensor_id.Map.empty in
  let node_outputs = ref [] in
  let body =
    let open Graph_builder in
    let* env =
      List.fold_left
        (fun acc (kind, name, target) ->
          let* env = acc in
          let shape = tensor_shape esc graph name in
          let* id =
            match kind with
            | `Input -> input ~shape ()
            | `Constant -> constant ~shape ~fmt:(tensor_fmt graph name) ()
          in
          tensor_origins :=
            Tensor_id.Map.add id
              (Pt2_native_graph.Source
                 {
                   graph_path = [];
                   ssa_name = name;
                   meta = String_map.find_opt name graph.tensor_values;
                 })
              !tensor_origins;
          Option.iter
            (fun x ->
              captured_targets := Tensor_id.Map.add id x !captured_targets)
            target;
          return (String_map.add name id env))
        (return String_map.empty) source_specs
    in
    let ctx : Native_interp_lower_context.t =
      { esc; graph; reads; constant_names }
    in
    let lower_op env node =
      match
        List.find_map
          (fun dispatch -> dispatch ~ctx ~env node)
          [
            Native_interp_lower_compute.dispatch;
            Native_interp_lower_recurrent.dispatch;
            Native_interp_lower_reduce.dispatch;
            Native_interp_lower_shape.dispatch;
          ]
      with
      | Some body -> body
      | None -> Err.Escape.throw esc (`Unsupported_operator node.target)
    in
    let lower_node index env node =
      let names = materialized_output_names esc node in
      let bind ids =
        (* THE arity check, here rather than in any operator arm, and against the
           ids the builder actually produced rather than against anything
           derived. Two reasons, both learned the hard way:

           [output_names] flattens [Argument.Tensors] for EVERY node, not just
           the ops that return one — so a serialized `relu` carrying an
           `as_tensors` output reaches this point with two names and one id.
           Before flattening, that shape was refused as
           [`Non_tensor_node_output]; an arm-local check would restore the hole
           for every op it does not cover.

           And a check against metadata-derived counts is not the same property:
           a node whose declared [tensor_values] shape disagrees with the shape
           Native infers passes it and still arrives here mismatched.

           [add_env]'s [Invalid_argument] stays an invariant about this module,
           and this is what keeps a malformed export from reaching it — as does
           [List.iter2] just below, which would otherwise raise first. *)
        if List.compare_lengths names ids <> 0 then
          malformed esc
            (`Output_arity
               {
                 op = node.Pytorch_types.Node.target;
                 serialized = List.length names;
                 derived = List.length ids;
               });
        List.iter2
          (fun name id ->
            tensor_origins :=
              Tensor_id.Map.add id
                (Pt2_native_graph.Source
                   {
                     graph_path = [];
                     ssa_name = name;
                     meta = String_map.find_opt name graph.tensor_values;
                   })
                !tensor_origins)
          names ids;
        node_outputs := (index, ids) :: !node_outputs;
        add_env env names ids
      in
      if is_nontrivial_node node then
        let* ids = group ~label:node.target (lower_op env node) in
        return (bind ids)
      else
        let* ids = lower_op env node in
        return (bind ids)
    in
    let* env =
      List.fold_left
        (fun acc (index, node) ->
          let* env = acc in
          lower_node index env node)
        (return env)
        (List.mapi (fun i n -> (i, n)) graph.nodes)
    in
    (* The second place a `Tensor[]` has to flatten, through the same bounded
       helper: a graph may return the whole list an unbind produced. The budget
       is shared across the graph's outputs, so the bound is on the AGGREGATE
       interface width rather than on each list separately. *)
    let outputs =
      List.map (env_find esc env)
        (flatten_outputs esc
           ~on_bad_kind:(fun () -> malformed esc `Non_tensor_graph_output)
           graph.outputs)
    in
    return outputs
  in
  match
    Graph_builder.build ~name:"pt2" ~outputs:Fun.id body
    |> Err.map_error ~pos:__POS__ (fun e -> `Build e)
  with
  | Error _ as e -> e
  | Ok native_graph ->
      let node_origins =
        List.fold_left
          (fun acc (source_index, ids) ->
            let origin_node = List.nth graph.nodes source_index in
            List.fold_left
              (fun acc (node : Graph_ir.node) ->
                if
                  List.exists
                    (fun id -> List.mem id node.Graph_ir.Node.outputs)
                    ids
                then
                  Node_id.Map.add node.Graph_ir.Node.id
                    [
                      {
                        Pt2_native_graph.Node_origin.graph_path = [];
                        index = source_index;
                        target = origin_node.target;
                        name = origin_node.name;
                        metadata = origin_node.metadata;
                      };
                    ]
                    acc
                else acc)
              acc native_graph.Graph_ir.Graph.nodes)
          Node_id.Map.empty !node_outputs
      in
      Pt2_native_graph.make ~graph:native_graph ~tensor_origins:!tensor_origins
        ~node_origins ~captured_targets:!captured_targets
      |> Err.map_error ~pos:__POS__ (fun e -> `Provenance e)

let lower_archive archive = lower (Pt2_archive.program archive)
let tensor_of_pt2 = Native_interp_tensor.tensor_of_pt2
