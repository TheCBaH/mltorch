(* The two value-graph stages. See the .mli. *)

module ME = Model_explorer

type error =
  [ Me_ids.error | Me_limits.over_limit_error | `Unknown_producer of int ]

let pp_error fmt : [< error ] -> unit = function
  | #Me_ids.error as e -> Me_ids.pp_error fmt e
  | `Over_limit o -> Me_limits.Over_limit.pp fmt o
  | `Unknown_producer t -> Fmt.pf fmt "source t%d has no producer" t

let over_limit = Me_limits.check ~scope:Me_limits.Scope.Value_graph

let attr key value =
  ME.NodeAttribute.create ~key ~value:(ME.NodeAttributeValue.Str value)

let kv key value = ME.KeyValue.create ~key ~value
let tid_name id = Core.Pretty.to_string Graph_ir.Tensor_id.pp id

let shape (sg : Tensor_sig.t) =
  let text, _ =
    Me_build.bounded ~max:256
      (fun fmt s -> Vec6.pp_shape fmt s.Tensor_sig.shape)
      sg
  in
  text

(* One node per value, one pinned boundary per input and per output.

   [id] is the VALUE's tensor id in both dialects, so a node keeps the same id
   across the stage program and the kernel it was adapted into — which is what
   lets the two be compared without an explicit mapping, the same argument the
   Native comparison rests on. *)
let build ~limits ~id ~inputs ~outputs ~values =
  let open Err.Syntax in
  let node_count =
    List.length values + List.length inputs + List.length outputs
  in
  let* () =
    over_limit Me_limits.Field.Nodes node_count
      ~ceiling:limits.Me_limits.Limits.max_nodes_per_graph
  in
  let producer = Hashtbl.create 128 in
  List.iter
    (fun (vid, _, _, _) ->
      Hashtbl.replace producer
        (Graph_ir.Tensor_id.to_int vid)
        (Me_ids.boundary `In vid))
    inputs;
  List.iter
    (fun (vid, _, _, _) ->
      Hashtbl.replace producer
        (Graph_ir.Tensor_id.to_int vid)
        (Me_ids.value_node vid))
    values;
  let resolve src =
    let k = Expr.Source.to_int src in
    match Hashtbl.find_opt producer k with
    | Some n -> Err.return n
    | None -> Err.fail (`Unknown_producer k)
  in
  let boundary kind (vid, label, sg, attrs) =
    ME.GraphNode.create ~id:(Me_ids.boundary kind vid) ~label ~namespace:""
      ~incomingEdges:[]
      ~outputsMetadata:
        [
          ME.MetadataItem.create ~id:"0"
            ~attrs:([ kv "value" (tid_name vid); kv "shape" (shape sg) ] @ attrs);
        ]
      ()
  in
  let input_nodes = List.map (boundary `In) inputs in
  let* value_nodes =
    Err.List.map
      (fun (vid, label, sg, program) ->
        let+ incoming =
          Err.List.map
            (fun src ->
              let+ from = resolve src in
              ME.IncomingEdge.create ~sourceNodeId:from ~sourceNodeOutputId:"0"
                ~targetNodeInputId:(Core.Pretty.to_string Expr.Source.pp src)
                ())
            (Expr.Source.Set.elements (Region_program.Fold.sources program))
        in
        (* The WHOLE program, stopped at the cap rather than built whole and
           cut: an expression tree is exactly the value where those two
           differ. [Region_program.pp] names every local, so this carries no
           dangling [?#N] the way [Expr.Pp.value] on the emitter alone would
           for a program the emitter references locals into. *)
        let text, capped =
          Me_build.bounded ~max:limits.Me_limits.Limits.max_attr_chars
            Region_program.pp program
        in
        ME.GraphNode.create ~id:(Me_ids.value_node vid) ~label ~namespace:""
          ~incomingEdges:incoming
          ~outputsMetadata:
            [
              ME.MetadataItem.create ~id:"0"
                ~attrs:[ kv "value" (tid_name vid); kv "shape" (shape sg) ];
            ]
          ~attrs:
            ([
               attr "size" (string_of_int (Region_program.Fold.size program));
               attr "depth"
                 (string_of_int (Region_program.Fold.max_depth program));
               attr "body" text;
             ]
            @ if capped then [ attr "body_truncated" "true" ] else [])
          ())
      values
  in
  let* output_nodes =
    Err.List.map
      (fun (vid, label, sg, attrs) ->
        let+ from =
          resolve (Expr.Source.create (Graph_ir.Tensor_id.to_int vid))
        in
        ME.GraphNode.create ~id:(Me_ids.boundary `Out vid) ~label ~namespace:""
          ~incomingEdges:
            [
              ME.IncomingEdge.create ~sourceNodeId:from ~sourceNodeOutputId:"0"
                ~targetNodeInputId:"0" ();
            ]
          ~outputsMetadata:
            [
              ME.MetadataItem.create ~id:"0"
                ~attrs:
                  ([ kv "value" (tid_name vid); kv "shape" (shape sg) ] @ attrs);
            ]
          ())
      outputs
  in
  let all = input_nodes @ value_nodes @ output_nodes in
  let edges =
    List.fold_left
      (fun acc (n : ME.GraphNode.t) ->
        acc
        + List.length (Option.value n.ME.GraphNode.incomingEdges ~default:[]))
      0 all
  in
  let+ () =
    over_limit Me_limits.Field.Edges edges
      ~ceiling:limits.Me_limits.Limits.max_edges_per_graph
  in
  ME.Graph.create ~id ~nodes:all ()

let stage_program ~limits ~id (p : Stage_program.t) =
  let by_id =
    List.fold_left
      (fun m (s : Stage_program.Stage.t) ->
        Graph_ir.Tensor_id.Map.add s.Stage_program.Stage.id s m)
      Graph_ir.Tensor_id.Map.empty p.Stage_program.stages
  in
  build ~limits ~id
    ~inputs:
      (* [consts] are boundaries too, and omitting them is the one thing that
         made a real model fail here: a synthetic constant-filled operand is a
         source a stage loads from and no stage produces. Its fill value goes on
         the node, because it appears nowhere else. *)
      (List.map
         (fun (vid, sg) ->
           ( vid,
             (match
                Graph_ir.Tensor_id.Map.find_opt vid p.Stage_program.input_kinds
              with
             | Some Graph_ir.Input.Constant -> "constant"
             | Some Graph_ir.Input.Input | None -> "input"),
             sg,
             [] ))
         p.Stage_program.inputs
      @ List.map
          (fun ((sg : Tensor_sig.t), v) ->
            ( sg.Tensor_sig.id,
              "filled",
              sg,
              [ kv "fill" (Printf.sprintf "%g" v) ] ))
          p.Stage_program.consts)
    ~values:
      (List.map
         (fun (s : Stage_program.Stage.t) ->
           ( s.Stage_program.Stage.id,
             "stage",
             s.Stage_program.Stage.sg,
             s.Stage_program.Stage.computation ))
         p.Stage_program.stages)
    ~outputs:
      (List.filter_map
         (fun vid ->
           Option.map
             (fun (s : Stage_program.Stage.t) ->
               (vid, "output", s.Stage_program.Stage.sg, []))
             (Graph_ir.Tensor_id.Map.find_opt vid by_id))
         p.Stage_program.outputs)

let kernel ~limits ~id (k : Kernel.t) =
  build ~limits ~id
    ~inputs:
      (List.map
         (fun (i : Kernel.Input.t) ->
           ( i.Kernel.Input.id,
             (* The BINDING is what the adaptation decided and appears nowhere
                in the graph shape, so it is on the boundary node itself. *)
             Core.Pretty.to_string Kernel.Binding.pp i.Kernel.Input.binding,
             i.Kernel.Input.sg,
             [] ))
         k.Kernel.inputs)
    ~values:
      (List.map
         (fun (v : Kernel.Value.t) ->
           ( v.Kernel.Value.id,
             Kernel.Result_conversion.name v.Kernel.Value.result,
             v.Kernel.Value.sg,
             v.Kernel.Value.computation ))
         k.Kernel.values)
    ~outputs:
      (List.map
         (fun (o : Kernel.Output.t) ->
           (o.Kernel.Output.value, "output", o.Kernel.Output.sg, []))
         k.Kernel.outputs)
