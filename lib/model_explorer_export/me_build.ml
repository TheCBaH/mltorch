(* Project a native graph into a Model Explorer graph. See the .mli. *)

module ME = Model_explorer

module type SIDE = sig
  type op

  val op_name : op -> string
  val operands : op -> Graph_ir.Tensor_id.t list
  val pp_op : Format.formatter -> op -> unit
end

type error =
  [ Me_ids.error | Me_limits.over_limit_error | `Unknown_producer of int ]

let pp_error fmt : [< error ] -> unit = function
  | #Me_ids.error as e -> Me_ids.pp_error fmt e
  | `Over_limit o -> Me_limits.Over_limit.pp fmt o
  | `Unknown_producer t -> Fmt.pf fmt "tensor t%d has no producer" t

let over_limit = Me_limits.check ~scope:Me_limits.Scope.Graph

(* Stop the printer AT the cap.

   [Format.make_formatter] takes the output function, so refusing the write that
   would cross the ceiling and unwinding from inside it is what makes this a
   bound on the work rather than on the result. [Format.asprintf] followed by
   [String.sub] has already paid for everything it then throws away, and for a
   deeply nested op that is the whole cost. *)
exception Capped

let bounded ~max pp v =
  let buf = Buffer.create (Stdlib.min 256 max) in
  let out s pos len =
    let room = max - Buffer.length buf in
    if len > room then begin
      Buffer.add_substring buf s pos room;
      raise Capped
    end
    else Buffer.add_substring buf s pos len
  in
  let fmt = Format.make_formatter out (fun () -> ()) in
  let capped =
    match
      Format.pp_print_flush fmt ();
      pp fmt v;
      Format.pp_print_flush fmt ()
    with
    | () -> false
    | exception Capped -> true
  in
  (Buffer.contents buf, capped)

(* The [Group] tree is authoritative, so the namespace is read off it rather
   than off any importer string. Each level goes through
   [Me_ids.namespace_component], which is what stops a label containing a
   slash from silently becoming two levels of hierarchy — the renderer splits
   [namespace] on [/], so an unencoded label would move a node.

   Outside the functor because it does not mention the dialect at ALL, and
   because a second consumer needs it: a verification rollup keys
   [groupNodeAttributes] by namespace, and computing that key a second way is
   how two answers about the same hierarchy come to disagree. *)
let namespaces ~limits (root : Graph_ir.Group.t) =
  let open Err.Syntax in
  let table = Hashtbl.create 64 in
  let rec walk prefix (g : Graph_ir.Group.t) =
    let* component =
      Me_ids.namespace_component ~limits ?label:g.Graph_ir.Group.label
        (Graph_ir.Group_id.to_int g.Graph_ir.Group.id)
    in
    let path = prefix @ [ component ] in
    Err.List.iter
      (fun (item : Graph_ir.Group.item) ->
        match item with
        | Graph_ir.Group.Node id ->
            Hashtbl.replace table id (String.concat "/" path);
            Err.return ()
        | Graph_ir.Group.Group sub -> walk path sub)
      g.Graph_ir.Group.items
  in
  (* The root's own component is dropped: every node would otherwise sit one
     level deeper than the hierarchy says, and the extra level names the
     graph, which the graph id already does. *)
  let+ () =
    Err.List.iter
      (fun (item : Graph_ir.Group.item) ->
        match item with
        | Graph_ir.Group.Node id ->
            Hashtbl.replace table id "";
            Err.return ()
        | Graph_ir.Group.Group sub -> walk [] sub)
      root.Graph_ir.Group.items
  in
  table

let namespace_of ~limits root =
  let open Err.Syntax in
  let+ table = namespaces ~limits root in
  fun id -> Option.value (Hashtbl.find_opt table id) ~default:""

module Make (S : SIDE) = struct
  (* Node attributes and output-metadata attributes are different types
     upstream: the first carries a [NodeAttributeValue.t] (which can also be a
     node-id list), the second a plain string. Two constructors rather than one
     coerced, so a caller cannot put a node-id list where the renderer expects
     text. *)
  let attr key value =
    ME.NodeAttribute.create ~key ~value:(ME.NodeAttributeValue.Str value)

  let kv key value = ME.KeyValue.create ~key ~value

  let shape_attr (sig_ : Tensor_sig.t) =
    let text, _ =
      bounded ~max:256 (fun fmt s -> Vec6.pp_shape fmt s.Tensor_sig.shape) sig_
    in
    text

  let dtype_attr (sig_ : Tensor_sig.t) =
    let (Payload.Fmt f) = sig_.Tensor_sig.fmt in
    Payload.fmt_name f

  (* Every property a tensor signature can state, not only its shape: dtype
     and, where quantized, the quantization parameters. *)
  let tensor_sig_kv (sig_ : Tensor_sig.t) =
    [ kv "shape" (shape_attr sig_); kv "dtype" (dtype_attr sig_) ]
    @
    match sig_.Tensor_sig.quant with
    | None -> []
    | Some q -> [ kv "quant" (Core.Pretty.to_string Quant.pp q) ]

  (* A constant's symbolic definition, from Const-SSA. [Leaf] renders as what
     it is (captured/literal/materialized); [Apply] recurses through its
     operands so a folded constant shows its WHOLE derivation, e.g.
     [permute(captured "p_conv1_weight")], not just its own last step. The
     plan is acyclic by construction ([Const_ssa.validate]), so this
     terminates. *)
  let pp_const_leaf fmt : Const_ssa.leaf -> unit = function
    | Const_ssa.Captured c -> Fmt.pf fmt "captured %a" Const_ssa.Capture.pp c
    | Const_ssa.Literal _ -> Fmt.string fmt "literal"
    | Const_ssa.Opaque_materialized _ -> Fmt.string fmt "materialized"

  let rec pp_const_ref plan fmt (id : Graph_ir.Tensor_id.t) =
    match Const_ssa.find plan (Const_ssa.Value_id.of_tensor_id id) with
    | None -> Graph_ir.Tensor_id.pp fmt id
    | Some (Const_ssa.Leaf { leaf; _ }) -> pp_const_leaf fmt leaf
    | Some (Const_ssa.Apply { op; _ }) ->
        Graph_ir.pp_op_with ~pp_ref:(pp_const_ref plan) fmt op

  (* Only [Apply] counts as "subject to constant propagation": a bare
     captured or literal leaf was not derived from anything, so it has no
     transformation to show. [Constant_store.binding] is needed rather than
     [Const_ssa.find] on [t] directly, because packing only remaps a plan's
     EXPORTS — the main-graph tensor id can differ from the stable internal
     plan id the definition was recorded under. *)
  let constant_transform_attrs ~limits constant_store t =
    match constant_store with
    | None -> []
    | Some store -> (
        match Constant_store.binding store t with
        | None -> []
        | Some vid -> (
            let plan = Constant_store.plan store in
            match Const_ssa.find plan vid with
            | Some (Const_ssa.Apply { op; _ }) ->
                let text, capped =
                  bounded ~max:limits.Me_limits.Limits.max_attr_chars
                    (Graph_ir.pp_op_with ~pp_ref:(pp_const_ref plan))
                    op
                in
                attr "constant_transform" text
                ::
                (if capped then [ attr "constant_transform_truncated" "true" ]
                 else [])
            | Some (Const_ssa.Leaf _) | None -> []))

  let graph ~limits ~id ?labels ?group_attrs ?constant_store
      (g : S.op Graph_ir.Graph.t) =
    let open Err.Syntax in
    let label_of = match labels with Some f -> f | None -> fun _ -> "" in
    let nodes = g.Graph_ir.Graph.nodes in
    (* The ceiling comes first, before the walks that are linear in it. *)
    let node_count =
      List.length nodes
      + List.length g.Graph_ir.Graph.inputs
      + List.length g.Graph_ir.Graph.outputs
    in
    let* () =
      over_limit Me_limits.Field.Nodes node_count
        ~ceiling:limits.Me_limits.Limits.max_nodes_per_graph
    in
    let* ns = namespaces ~limits g.Graph_ir.Graph.root in
    (* Where each tensor comes from. Boundary tensors resolve to their pinned
       boundary node, so an edge from a graph input is an ordinary edge rather
       than a special case every consumer would have to know about. *)
    let producer = Hashtbl.create 128 in
    List.iter
      (fun t ->
        let kind = Graph_common.input_kind g t in
        Hashtbl.replace producer
          (Graph_ir.Tensor_id.to_int t)
          ( Me_ids.boundary
              (match kind with
              | Graph_ir.Input.Input -> `In
              | Graph_ir.Input.Constant -> `Const)
              t,
            0 ))
      g.Graph_ir.Graph.inputs;
    List.iter
      (fun (n : S.op Graph_ir.Node.t) ->
        List.iteri
          (fun slot t ->
            Hashtbl.replace producer
              (Graph_ir.Tensor_id.to_int t)
              (Me_ids.op_node n.Graph_ir.Node.id, slot))
          n.Graph_ir.Node.outputs)
      nodes;
    let resolve t =
      match Hashtbl.find_opt producer (Graph_ir.Tensor_id.to_int t) with
      | Some p -> Err.return p
      | None -> Err.fail (`Unknown_producer (Graph_ir.Tensor_id.to_int t))
    in
    let outputs_metadata (n : S.op Graph_ir.Node.t) =
      List.mapi
        (fun slot t ->
          let attrs =
            match
              Graph_ir.Tensor_id.Map.find_opt t g.Graph_ir.Graph.tensors
            with
            | None ->
                [
                  kv "tensor_id" (Core.Pretty.to_string Graph_ir.Tensor_id.pp t);
                ]
            | Some sg ->
                [
                  kv "tensor_id" (Core.Pretty.to_string Graph_ir.Tensor_id.pp t);
                  kv "shape" (shape_attr sg);
                ]
          in
          ME.MetadataItem.create ~id:(string_of_int slot) ~attrs)
        n.Graph_ir.Node.outputs
    in
    (* Boundary nodes, pinned so a comparison matches them like any other.
       Only a CONSTANT gets tensor-signature and Const-SSA attributes — an
       ordinary input's properties and an output's are unaffected. *)
    let boundary kind t =
      let sig_attrs, transform_attrs =
        match kind with
        | `In | `Out -> ([], [])
        | `Const ->
            let sig_attrs =
              match
                Graph_ir.Tensor_id.Map.find_opt t g.Graph_ir.Graph.tensors
              with
              | None -> []
              | Some sg -> tensor_sig_kv sg
            in
            (sig_attrs, constant_transform_attrs ~limits constant_store t)
      in
      ME.GraphNode.create ~id:(Me_ids.boundary kind t)
        ~label:
          (match kind with
          | `In -> "input"
          | `Const -> "constant"
          | `Out -> "output")
        ~namespace:"" ~incomingEdges:[]
        ~outputsMetadata:
          [
            ME.MetadataItem.create ~id:"0"
              ~attrs:
                (kv "tensor_id" (Core.Pretty.to_string Graph_ir.Tensor_id.pp t)
                :: sig_attrs);
          ]
        ?attrs:(if transform_attrs = [] then None else Some transform_attrs)
        ()
    in
    let input_nodes =
      List.map
        (fun t ->
          boundary
            (match Graph_common.input_kind g t with
            | Graph_ir.Input.Input -> `In
            | Graph_ir.Input.Constant -> `Const)
            t)
        g.Graph_ir.Graph.inputs
    in
    let* op_nodes =
      Err.List.map
        (fun (n : S.op Graph_ir.Node.t) ->
          let+ incoming =
            Err.List.map
              (fun t ->
                let+ src, slot = resolve t in
                ME.IncomingEdge.create ~sourceNodeId:src
                  ~sourceNodeOutputId:(string_of_int slot)
                  ~targetNodeInputId:
                    (Core.Pretty.to_string Graph_ir.Tensor_id.pp t)
                  ())
              (S.operands n.Graph_ir.Node.op)
          in
          let params, capped =
            bounded ~max:limits.Me_limits.Limits.max_attr_chars S.pp_op
              n.Graph_ir.Node.op
          in
          let extra = label_of n.Graph_ir.Node.id in
          ME.GraphNode.create
            ~id:(Me_ids.op_node n.Graph_ir.Node.id)
            ~label:(S.op_name n.Graph_ir.Node.op)
            ~namespace:
              (Option.value
                 (Hashtbl.find_opt ns n.Graph_ir.Node.id)
                 ~default:"")
            ~incomingEdges:incoming ~outputsMetadata:(outputs_metadata n)
            ~attrs:
              ([ attr "params" params ]
              @ (if capped then [ attr "params_truncated" "true" ] else [])
              @ if extra = "" then [] else [ attr "module" extra ])
            ())
        nodes
    in
    let* output_nodes =
      Err.List.map
        (fun t ->
          let+ src, slot = resolve t in
          ME.GraphNode.create ~id:(Me_ids.boundary `Out t) ~label:"output"
            ~namespace:""
            ~incomingEdges:
              [
                ME.IncomingEdge.create ~sourceNodeId:src
                  ~sourceNodeOutputId:(string_of_int slot)
                  ~targetNodeInputId:"0" ();
              ]
            ~outputsMetadata:[] ())
        g.Graph_ir.Graph.outputs
    in
    let all = input_nodes @ op_nodes @ output_nodes in
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
    (* [groupNodeAttributes] is keyed by NAMESPACE, which is why the rollup
       that fills it shares [namespaces] with this projection rather than
       recomputing the key. *)
    ME.Graph.create ~id ~nodes:all
      ?groupNodeAttributes:
        (Option.map
           (fun rows ->
             List.fold_left
               (fun acc (ns, attrs) ->
                 ME.String_map.add ns
                   (List.fold_left
                      (fun m (k, v) -> ME.String_map.add k v m)
                      ME.String_map.empty attrs)
                   acc)
               ME.String_map.empty rows)
           group_attrs)
      ()
end
