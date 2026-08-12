(* Fusion as an overlay over an unchanged kernel. See the .mli. *)

module ME = Model_explorer

type error = Me_limits.over_limit_error

let pp_error fmt : [< error ] -> unit = function
  | `Over_limit o -> Me_limits.Over_limit.pp fmt o

let over_limit = Me_limits.check ~scope:Me_limits.Scope.Fusion

type t = {
  overlays : ME.EdgeOverlaysData.t;
  node_data : Me_session.Node_data_set.t;
  rejections : Me_limits.Diagnostic.t list;
}

(* [">= 2"], never a figure. The planner's counter saturates at two because the
   cross-body total is an [int] and the per-value limits admit a mathematical
   aggregate past 2^31 -- which wraps negative under js_of_ocaml and would read
   as unique-use. Legality only asks one versus more than one, so a count would
   claim precision the value does not have; the upstream field is named
   [at_least] for the same reason. *)
let pp_rejection fmt (r : Fusion_plan.Rejection.t) =
  match r with
  | Fusion_plan.Rejection.Multiple_uses { producer; at_least = _ } ->
      Fmt.pf fmt "%a has >= 2 uses" Graph_ir.Tensor_id.pp producer
  | _ -> Fusion_plan.Rejection.pp fmt r

let of_kernel ~limits ~graph k =
  let open Err.Syntax in
  let plan, decisions = Fusion_plan.plan k in
  let virtual_uses = plan.Fusion_plan.virtual_uses in
  let stores = plan.Fusion_plan.stores in
  let edges =
    Kernel.Use.Set.fold
      (fun (u : Kernel.Use.t) acc ->
        ME.Edge.create
          ~sourceNodeId:(Me_ids.value_node u.Kernel.Use.producer)
          ~targetNodeId:(Me_ids.value_node u.Kernel.Use.consumer)
          ~label:"virtual" ()
        :: acc)
      virtual_uses []
  in
  let* () =
    over_limit Me_limits.Field.Overlay_edges (List.length edges)
      ~ceiling:limits.Me_limits.Limits.max_overlay_edges_per_overlay
  in
  (* TWO facts, and the value scale is where they meet: a producer that is
     virtual for its consumer and still stored for someone else is neither
     "fused" nor "materialized", and the ordering says so. *)
  let virtual_producers =
    Kernel.Use.Set.fold
      (fun (u : Kernel.Use.t) acc ->
        Graph_ir.Tensor_id.Set.add u.Kernel.Use.producer acc)
      virtual_uses Graph_ir.Tensor_id.Set.empty
  in
  (* A rejection is per VALUE, so it belongs on the value's own datum rather
     than in the diagnostics list -- which is bounded at [max_diagnostics] for
     a reason, and a real model produces one rejection per unfused producer
     (seventy on resnet18 alone). What reaches the diagnostics is one SUMMARY,
     counted by reason, which is bounded by the reason vocabulary. *)
  let rejected = Hashtbl.create 64 in
  List.iter
    (fun (d : Fusion_plan.Decision.t) ->
      match d with
      | Fusion_plan.Decision.Virtualize _ -> ()
      | Fusion_plan.Decision.Reject { producer; reason } ->
          Hashtbl.replace rejected
            (Graph_ir.Tensor_id.to_int producer)
            (Core.Pretty.to_string pp_rejection reason))
    decisions;
  let results =
    List.map
      (fun (v : Kernel.Value.t) ->
        let id = v.Kernel.Value.id in
        let stored = Graph_ir.Tensor_id.Set.mem id stores in
        let virt = Graph_ir.Tensor_id.Set.mem id virtual_producers in
        let placement =
          match (stored, virt) with
          | true, false -> "stored"
          | true, true -> "virtual and stored"
          | false, true -> "virtual"
          (* Neither: a value the plan places nowhere. The planner does not
             produce one today, and the ordering still needs a place for it
             rather than folding it into "stored". *)
          | false, false -> "unplaced"
        in
        ( Me_ids.value_node id,
          {
            Me_session.Node_data_set.value =
              (match (stored, virt) with
              | true, false -> 0.
              | true, true -> 1.
              | false, true -> 2.
              | false, false -> 3.);
            label =
              Some
                (match
                   Hashtbl.find_opt rejected (Graph_ir.Tensor_id.to_int id)
                 with
                | None -> placement
                | Some why -> placement ^ " (" ^ why ^ ")");
          } ))
      k.Kernel.values
  in
  let+ () =
    over_limit Me_limits.Field.Node_data_results (List.length results)
      ~ceiling:limits.Me_limits.Limits.max_node_data_results_per_graph
  in
  (* The summary: how many producers stayed materialized, and how many edges
     the plan made virtual. Bounded by construction, unlike a diagnostic per
     rejection. *)
  let summary =
    Printf.sprintf "%d virtual edges, %d producers not fused"
      (Kernel.Use.Set.cardinal virtual_uses)
      (Hashtbl.length rejected)
  in
  {
    overlays =
      ME.EdgeOverlaysData.create ~name:"fusion" ~graphName:graph
        ~overlays:
          [
            ME.EdgeOverlay.create ~name:"virtual dependencies" ~edges
              ~edgeColor:"#8e24aa" ();
          ]
        ();
    node_data = { Me_session.Node_data_set.name = "fusion"; graph; results };
    rejections =
      [
        Me_limits.Diagnostic.create ~limits ~graph
          Me_limits.Diagnostic.Code.Unsupported_graph_shape summary;
      ];
  }
