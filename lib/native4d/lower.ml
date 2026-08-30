(* See lower.mli.

   THE ID POLICY, which is the decision everything else hangs off. An edge whose
   value the conversion preserves keeps its SOURCE raw id; every created edge
   takes a fresh id above the source watermark. That is forced, not chosen:
   [Graph_map.create] rejects a source id that is neither mentioned nor deleted
   but absent from the destination, so without preservation every edge would
   need an explicit cluster and the implicit-identity bulk that keeps a map
   proportional to what changed would disappear.

   The hazard — one raw id, two different values — is what claim closure and the
   local verifier catch, and stage 6 adds the crossed-operand mutation that
   proves they do. *)

(* Bound BEFORE [open Graph_ir], which shadows [Graph] with the Native one. *)
module G4 = Graph
open Graph_ir
open Err.Syntax

type ('src, 'dst) t = {
  dst : 'dst Framework.Snapshot4.t;
  map : ('src, 'dst) Graph_map.t;
  constants : Tensor.packed Tensor_id.Map.t;
  constant_store : Constant_store.t;
}

type 'src packed = Pack : ('src, 'dst) t -> 'src packed

let graph r = Framework.Snapshot4.graph r.dst

type eval_error =
  [ Const_ssa_materialize.error | `Direct4 of Eval_direct4.error ]

let pp_eval_error ppf : [< eval_error ] -> unit = function
  | #Const_ssa_materialize.error as e -> Const_ssa_materialize.pp_error ppf e
  | `Direct4 e -> Eval_direct4.pp_error ppf e

let evaluate resolver r ~inputs =
  let open Err.Syntax in
  let* store, report =
    Const_ssa_materialize.materialize resolver r.constant_store
    |> Err.map_error (fun e -> (e :> eval_error))
  in
  let constants =
    Tensor_id.Map.union
      (fun _ _ planned -> Some planned)
      r.constants
      (Constant_store.materialized store)
  in
  let+ env =
    Eval_direct4.run (graph r)
      ~constants:(Tensor_id.Map.bindings constants)
      ~inputs
    |> Err.map_error (fun e -> `Direct4 e)
  in
  (env, report)

(* The walk accumulator, parameter-translation helpers and [lower_node]
   (the per-source-node dispatch) live in lower_engine.ml, split out under
   the tracked file-size ceiling; opened here so [convert] below can use
   [lower_node], [resolve] and the [acc] record fields unqualified, exactly
   as when they lived in this file. *)
open Lower_engine
(* ---- constants ------------------------------------------------------------ *)

(* Validated over the WHOLE supplied map before any legalization runs, against
   the same contract [Rewrite.origin] holds its own payloads to: shape and
   format must match the recorded signature, and the id must be an effective
   constant. Checking only what batch norm consumes would not do — every other
   payload is copied through into [result.constants] and evaluated much later,
   so a malformed conv weight would cross the boundary intact and fail far from
   the mistake. *)
let fmt_name (Payload.Fmt f) = Payload.fmt_name f

let check_constants ~view constants =
  Err.List.iter
    (fun (id, Tensor.Tensor payload) ->
      let* () =
        if Graph_view.is_constant view id then Err.return ()
        else Err.fail (`Bad_constant_payload id)
      in
      match Graph_view.sig_of view id with
      | None -> Err.fail (`Bad_constant_payload id)
      | Some sg ->
          let shape_ok =
            List.for_all
              (fun axis ->
                Dim.equal
                  (Vec6.get sg.Tensor_sig.shape axis)
                  (Vec6.get payload.Tensor.shape axis))
              Axis.all
          in
          if
            shape_ok
            && String.equal
                 (fmt_name sg.Tensor_sig.fmt)
                 (Payload.fmt_name payload.Tensor.payload.Payload.fmt)
          then Err.return ()
          else Err.fail (`Bad_constant_payload id))
    (Tensor_id.Map.bindings constants)

(* ---- the conversion ------------------------------------------------------- *)

let convert ?(constants = Tensor_id.Map.empty)
    ?(constant_store = Constant_store.empty) (src : 'src Snapshot.t) =
  let view = Snapshot.view src in
  let g = Snapshot.graph src in
  let* () = (Domain.check view :> (unit, Error.t) Err.t) in
  let* () = check_constants ~view constants in
  (* Fresh ids start above the source watermark, so a created edge can never
     collide with a preserved one. *)
  let watermark =
    Tensor_id.Map.fold
      (fun id _ acc -> max acc (Tensor_id.to_int id + 1))
      g.Graph.tensors 0
  in
  (* An unread constant is model-bound state, not interface, so it is OMITTED
     and recorded as a deletion — the same seam [Rewrite.apply] cuts along ("a
     constant nobody reads is gone"). An unused user INPUT is not omitted; it is
     part of the signature, and [Domain.check] already rejected a non-four-axis
     one. *)
  let read id =
    List.exists
      (fun (n : node) -> List.mem id (Graph_ir.operands n.Node.op))
      g.Graph.nodes
    || List.mem id g.Graph.outputs
  in
  let kept_inputs, dropped_inputs =
    List.partition
      (fun id -> Graph_ir.input_kind g id = Input.Input || read id)
      g.Graph.inputs
  in
  let acc0 =
    {
      nodes = [];
      tensors =
        Tensor_id.Map.filter
          (fun id _ -> not (List.mem id dropped_inputs))
          g.Graph.tensors;
      subst = Tensor_id.Map.empty;
      next_tid = watermark;
      next_nid =
        List.fold_left
          (fun acc (n : node) -> max acc (Node_id.to_int n.Node.id + 1))
          0 g.Graph.nodes;
      created = [];
      deleted = dropped_inputs;
      claims = [];
      node_pairs = [];
      provenance = [];
      constants;
      fresh_constants = [];
    }
  in
  let* acc = Err.List.fold_left (lower_node ~view) acc0 g.Graph.nodes in
  let dst_graph =
    {
      G4.Graph.nodes = List.rev acc.nodes;
      root =
        {
          Graph_common.Group.id = Graph_common.Group_id.of_int 0;
          label = None;
          items =
            List.map
              (fun (n : G4.node) -> Graph_common.Group.Node n.G4.Node.id)
              (List.rev acc.nodes);
        };
      tensors = acc.tensors;
      inputs = kept_inputs @ List.rev acc.fresh_constants;
      input_kinds =
        List.fold_left
          (fun m id -> Tensor_id.Map.add id Input.Constant m)
          (Tensor_id.Map.filter
             (fun id _ -> List.mem id kept_inputs)
             g.Graph.input_kinds)
          acc.fresh_constants;
      outputs = List.map (resolve acc) g.Graph.outputs;
    }
  in
  let* (Framework.Snapshot4.Pack dst) =
    Err.map_error (fun e -> `View e) (Framework.Snapshot4.create dst_graph)
  in
  let* constant_store =
    Constant_store.restrict_and_rename_exports constant_store (fun id ->
        if
          Graph_common.input_kind dst_graph id = Input.Constant
          && Tensor_id.Map.mem id dst_graph.G4.Graph.tensors
        then Some id
        else None)
    |> Err.map_error (fun e -> `Constant_store e)
  in
  (* Claims, PROPAGATED FORWARD. One claim per legalized node is not enough: the
     moment any legalization is weaker than Identical every edge downstream is
     too, and an edge left implicit reads as Identical — which
     [Graph_map.create]'s closure check rejects. [Rewrite.apply]'s steps 9-10 are
     the template, and the table is the DESTINATION dialect's. *)
  let explicit =
    List.fold_left
      (fun m (id, rel) -> Tensor_id.Map.add id rel m)
      Tensor_id.Map.empty acc.claims
  in
  let propagated =
    Output_transfer4.propagate ~explicit
      ~preserved:(fun id -> Snapshot.edge src id <> None)
      dst_graph
  in
  let all_claims =
    Tensor_id.Map.union (fun _ a _ -> Some a) explicit propagated
  in
  let edge_src id = Snapshot.edge src id in
  let edge_dst id = Framework.Snapshot4.edge dst id in
  let value_clusters =
    (* Weaker-than-Identical edges, stated outright. *)
    Tensor_id.Map.fold
      (fun id rel acc ->
        match (edge_src id, edge_dst id) with
        | Some s, Some d -> Correspondence.pair s d rel :: acc
        | _ -> acc)
      all_claims []
    (* Clone removal is {removed, kept} <-> {kept}, not {removed} <-> {kept}.
       The surviving edge is present in BOTH graphs, so mentioning it only as a
       destination makes it [Unpaired_dst]: an id named on one side while
       present in both must be named on the other. This is the same shape
       .ai/native_transform_design.md §3 gives for trimming an identity
       permute. *)
    @ List.filter_map
        (fun (removed, kept) ->
          match (edge_src removed, edge_src kept, edge_dst kept) with
          | Some removed, Some kept_src, Some kept_dst ->
              Some
                {
                  Correspondence.Cluster.src =
                    Correspondence.Set.of_list [ removed; kept_src ];
                  dst = Correspondence.Set.singleton kept_dst;
                  label = Correspondence.Identical;
                }
          | _ -> None)
        (Tensor_id.Map.bindings acc.subst)
    @ List.filter_map
        (fun id -> Option.map Correspondence.delete (edge_src id))
        acc.deleted
    @ List.filter_map
        (fun id -> Option.map Correspondence.create (edge_dst id))
        acc.created
  in
  (* One source node may become SEVERAL destination nodes — Mean keepdim=false
     becomes MeanKeepDims plus Reshape4, Bmm becomes Permute4 plus Conv2d — and
     [Node_map.fused] is the other direction (many sources, one destination).
     So the cluster is built directly: taking only the first destination would
     leave the second [Uncovered_dst]. *)
  let node_clusters =
    List.filter_map
      (fun (s, ds) ->
        let dst_ids =
          List.filter_map (fun d -> Framework.Snapshot4.node dst d) ds
        in
        match (Snapshot.node src s, dst_ids) with
        | Some s, (_ :: _ as ds) ->
            Some
              {
                Node_map.Cluster.src = Node_map.Set.singleton s;
                dst = Node_map.Set.of_list ds;
                label = ();
              }
        | _ -> None)
      acc.node_pairs
    (* A source node with no destination — [Clone], which contributes none — is
       a deletion, and saying nothing about it would leave it [Uncovered_src]. *)
    @ List.filter_map
        (fun (n : node) ->
          if
            List.exists (fun (s, _) -> Node_id.equal s n.Node.id) acc.node_pairs
          then None
          else Option.map Node_map.delete (Snapshot.node src n.Node.id))
        g.Graph.nodes
  in
  let provenance =
    List.fold_left
      (fun p (sources, target) ->
        match Framework.Snapshot4.edge dst target with
        | None -> p
        | Some target ->
            (* Sources that still exist in the source snapshot; one that does
               not simply contributes nothing. *)
            Provenance.add
              ~sources:
                (Correspondence.Set.of_list (List.filter_map edge_src sources))
              target p)
      Provenance.empty acc.provenance
  in
  let+ map =
    Err.map_error
      (fun e -> `Map e)
      (Framework.Map_from_native.create ~src ~dst ~values:value_clusters
         ~nodes:node_clusters ~provenance)
  in
  Pack { dst; map; constants = acc.constants; constant_store }
