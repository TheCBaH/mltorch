(* Running a lowered graph, and the transform/verify/evaluate pipeline built
   on top of it. Split from native_interp.ml. Depends on
   [Native_interp_lower] for [lower_archive]/[tensor_of_pt2], which is why it
   compiles after that module rather than before it. *)

module Tensor_id = Graph_ir.Tensor_id
open Native_interp_lower

type hooks =
  | Hooks : {
      on_start : Pt2_native_graph.t -> Graph_ir.node -> 'a;
      on_end : Pt2_native_graph.t -> Graph_ir.node -> 'a -> unit;
    }
      -> hooks

let run ?hooks archive ~input =
  let open Err.Syntax in
  let* lowered = lower_archive archive in
  let graph = lowered.Pt2_native_graph.graph in
  let eval_hooks =
    Option.map
      (fun (Hooks h) ->
        Eval_direct.Hooks
          {
            on_start = (fun node -> h.on_start lowered node);
            on_end = (fun node state -> h.on_end lowered node state);
          })
      hooks
  in
  let* input = tensor_of_pt2 input in
  let user_ids =
    List.filter
      (fun id -> Graph_ir.input_kind graph id = Graph_ir.Input.Input)
      graph.Graph_ir.Graph.inputs
  in
  let* inputs =
    match user_ids with
    | [ id ] -> Err.return [ (id, input) ]
    | ids ->
        Err.fail
          (`Unsupported_input (`Not_exactly_one_user_input (List.length ids)))
  in
  let used_constants =
    List.concat_map
      (fun node -> Graph_ir.operands node.Graph_ir.Node.op)
      graph.Graph_ir.Graph.nodes
  in
  let* constants =
    Err.List.map
      (fun (id, target) ->
        let* raw =
          Pt2_archive.load_captured_tensor archive target
          |> Err.map_error ~pos:__POS__ (fun e -> `Tensor_bridge (`Archive e))
        in
        let+ tensor = tensor_of_pt2 raw in
        (id, tensor))
      (List.filter
         (fun (id, _) -> List.mem id used_constants)
         (Tensor_id.Map.bindings lowered.captured_targets))
  in
  let* env =
    Eval_direct.run ?hooks:eval_hooks ~constants graph ~inputs
    |> Err.map_error ~pos:__POS__ (fun e -> `Eval e)
  in
  Err.List.map
    (fun id ->
      Tensor_id.Map.find_opt id env |> Err.of_option (`Output_not_evaluated id))
    graph.Graph_ir.Graph.outputs

(* ---- transforming, and running the result --------------------------------- *)

type transformed =
  | Transformed : {
      constants : Tensor.packed Tensor_id.Map.t;
      constant_store : Constant_store.t;
      derived : (Tensor_id.t * string list) list;
      graph : Graph_ir.graph;
      lens : 'b Pt2_native_graph.lens;
      nodes_before : int;
      audits : Pass.Audit_log.t;
      composed : Map_verify.Report.t option;
    }
      -> transformed

type loaded = { from_state : int; from_archive : int; from_plan : int }

let load_captured archive target =
  let open Err.Syntax in
  let* raw =
    Pt2_archive.load_captured_tensor archive target
    |> Err.map_error ~pos:__POS__ (fun e -> `Tensor_bridge (`Archive e))
  in
  tensor_of_pt2 raw

let capture_resolver archive capture =
  load_captured archive (Const_ssa.Capture.to_string capture)
  |> Err.map_error (fun _ -> `Missing_capture capture)

(* The PT2 names a destination constant derives from: its provenance sources,
   resolved in the sidecar the importer built. Asked only of an edge with no
   archive path of its own — a folded weight — which is exactly where "where did
   this come from" has no other answer. *)
let derivations lens sidecar (graph : Graph_ir.graph) =
  let open Err.Syntax in
  Err.List.fold_left
    (fun acc id ->
      if Graph_ir.input_kind graph id <> Graph_ir.Input.Constant then
        Err.return acc
      else
        let+ target =
          Pt2_native_graph.captured_target lens id
          |> Err.map_error ~pos:__POS__ (fun e -> `Lens e)
        in
        match target with
        | Some _ -> acc
        | None -> (
            let names =
              List.filter_map
                (fun src ->
                  match
                    Tensor_id.Map.find_opt src
                      sidecar.Pt2_native_graph.tensor_origins
                  with
                  | Some (Pt2_native_graph.Source o) ->
                      Some o.Pt2_native_graph.Tensor_origin.ssa_name
                  | Some Pt2_native_graph.Derived | None -> None)
                (Pt2_native_graph.provenance_sources lens id)
            in
            match names with [] -> acc | _ -> (id, names) :: acc))
    [] graph.Graph_ir.Graph.inputs

(* Everything downstream of lowering, over a graph the caller already has.

   Split out of [transform] because the archive is not a prerequisite for any of
   it: a caller holding a [Pt2_native_graph.t] — from a payload-free
   [model.json], or from a graph it built — can transform, verify and pack
   without an archive to read constants from, which [transform] would have
   demanded. [~constants] is that seed, as a MAP rather than the association
   list [Rewrite.origin] takes: an id appearing twice with different payloads is
   not a state this entry point should have to define an answer for. *)
let transform_lowered ?(constants = Tensor_id.Map.empty) ?verify ?verify_budget
    ?verify_probe ?max_verified_steps ?max_verify_clusters ?trace
    ?max_trace_entries ?max_audit_reports lowered ~passes =
  let open Err.Syntax in
  let source = lowered.Pt2_native_graph.graph in
  let seeded = Tensor_id.Map.bindings constants in
  let transform_error e = `Transform ((e : Rewrite.error) :> Pass.error) in
  let* constant_store =
    Tensor_id.Map.fold
      (fun id target acc ->
        let* store = acc in
        match Tensor_id.Map.find_opt id source.Graph_ir.Graph.tensors with
        | None -> Err.return store
        | Some tensor ->
            Constant_store.bind_captured store ~tensor
              (Const_ssa.Capture.of_string target)
            |> Err.map_error (fun e -> (e :> Rewrite.error))
            |> Err.map_error ~pos:__POS__ transform_error)
      lowered.Pt2_native_graph.captured_targets
      (Err.return Constant_store.empty)
  in
  let* (Rewrite.Origin origin) =
    Rewrite.origin ~constant_store ~constants:seeded source
    |> Err.map_error ~pos:__POS__ transform_error
  in
  let* {
         Pass.audits;
         trace = _;
         next_index = _;
         step = Rewrite.Step (rewritten, rewrite_map);
       } =
    Pass.run_reporting ?verify ?verify_budget ?verify_probe ?max_verified_steps
      ?trace ?max_trace_entries ?max_audit_reports origin passes
    |> Err.map_error ~pos:__POS__ (fun e -> `Transform e)
  in
  let* (Rewrite.Step (packed, pack_map)) =
    Rewrite.pack rewritten |> Err.map_error ~pos:__POS__ transform_error
  in
  let graph = Rewrite.graph packed in
  let composed_map = Graph_map.compose rewrite_map pack_map in
  let* lens =
    Pt2_native_graph.lens lowered ~src:origin composed_map ~dst:packed
    |> Err.map_error ~pos:__POS__ (fun e -> `Lens e)
  in
  (* The per-pass audits say what each rewrite established; this says what
     survived all of them, in the FINAL graph's ids — which is what lets a
     printed node carry its own verdict. A cluster here also names every origin
     edge that collapsed into one destination edge, so a node several passes
     rewrote reads as the one claim they add up to rather than as a pile of
     intermediate ones. *)
  let* composed =
    match verify with
    | None -> Err.return None
    | Some policy ->
        let* report =
          Map_verify.run ?budget:verify_budget ?probe:verify_probe
            ?max_clusters:max_verify_clusters composed_map
            ~src:(Rewrite.snapshot origin)
            ~src_constants:(Rewrite.constants origin)
            ~src_constant_store:(Rewrite.constant_store origin)
            ~dst:(Rewrite.snapshot packed)
            ~dst_constants:(Rewrite.constants packed)
            ~dst_constant_store:(Rewrite.constant_store packed)
          |> Err.map_error ~pos:__POS__ (fun e -> `Verify e)
        in
        (* The policy applies here too. Composition and terminal packing are the
           two steps no per-pass check covers — a refutation introduced by
           [Graph_map.compose] or [Rewrite.pack] appears in this report and
           nowhere else — so computing it and not judging it would print the
           failure inline and still exit successfully. *)
        if Map_verify.Policy.accepts policy report then Err.return (Some report)
        else
          Err.fail
            (`Transform
               (`Verification
                  {
                    Pass.Verification.pass = "compose+pack";
                    problem = Pass.Verification.Rejected report;
                  }))
  in
  let+ derived = derivations lens lowered graph in
  Transformed
    {
      composed;
      constants = Rewrite.constants packed;
      constant_store = Rewrite.constant_store packed;
      derived = List.rev derived;
      graph;
      lens;
      nodes_before = List.length source.Graph_ir.Graph.nodes;
      audits;
    }

(* Lower, optionally bind the payloads a node reads, then hand the rest to
   [transform_lowered]. The signature is unchanged, so every existing caller is
   unaffected by the split. *)
let preload archive lowered =
  let open Err.Syntax in
  let source = lowered.Pt2_native_graph.graph in
  let read_by_a_node =
    List.concat_map
      (fun (n : Graph_ir.node) -> Graph_ir.operands n.Graph_ir.Node.op)
      source.Graph_ir.Graph.nodes
    |> Tensor_id.Set.of_list
  in
  (* Only what a node reads: an archive holds buffers nothing evaluates —
     resnet18's int64 [num_batches_tracked] among them — and loading one would
     fail on a dtype the engine has no reason to support. *)
  let+ seeded =
    Err.List.map
      (fun (id, target) ->
        let+ payload = load_captured archive target in
        (id, payload))
      (List.filter
         (fun (id, _) -> Tensor_id.Set.mem id read_by_a_node)
         (Tensor_id.Map.bindings lowered.Pt2_native_graph.captured_targets))
  in
  Tensor_id.Map.of_seq (List.to_seq seeded)

let transform ?preload:(want_payloads = false) ?verify ?verify_budget
    ?verify_probe archive ~passes =
  let open Err.Syntax in
  let* lowered = lower_archive archive in
  let* constants =
    if want_payloads then preload archive lowered
    else Err.return Tensor_id.Map.empty
  in
  transform_lowered ~constants ?verify ?verify_budget ?verify_probe lowered
    ~passes

(* Payloads for the constants the graph actually reads, state before archive.
   An edge with neither is simply absent; [Eval_direct] is the one that decides
   whether that matters, and says which edge if it does. *)
let constants_for archive ~lens ~graph ~computed =
  let open Err.Syntax in
  let used =
    List.concat_map
      (fun (n : Graph_ir.node) -> Graph_ir.operands n.Graph_ir.Node.op)
      graph.Graph_ir.Graph.nodes
    |> Tensor_id.Set.of_list
  in
  let+ loaded =
    Err.List.fold_left
      (fun acc id ->
        if not (Tensor_id.Set.mem id used) then Err.return acc
        else
          match Tensor_id.Map.find_opt id computed with
          | Some payload -> Err.return ((id, payload, `State) :: acc)
          | None -> (
              let* target =
                Pt2_native_graph.captured_target lens id
                |> Err.map_error ~pos:__POS__ (fun e -> `Lens e)
              in
              match target with
              | None -> Err.return acc
              | Some target ->
                  let+ payload = load_captured archive target in
                  (id, payload, `Archive) :: acc))
      []
      (List.filter
         (fun id -> Graph_ir.input_kind graph id = Graph_ir.Input.Constant)
         graph.Graph_ir.Graph.inputs)
  in
  let count source =
    List.length (List.filter (fun (_, _, s) -> s = source) loaded)
  in
  ( List.rev_map (fun (id, payload, _) -> (id, payload)) loaded,
    { from_state = count `State; from_archive = count `Archive; from_plan = 0 }
  )

let evaluate archive (Transformed t) ~input =
  let open Err.Syntax in
  let* input = tensor_of_pt2 input in
  let* store, materialized =
    Const_ssa_materialize.materialize (capture_resolver archive)
      t.constant_store
    |> Err.map_error ~pos:__POS__ (fun e -> `Materialize e)
  in
  let* constants, loaded =
    constants_for archive ~lens:t.lens ~graph:t.graph
      ~computed:(Constant_store.materialized store)
  in
  let user_ids =
    List.filter
      (fun id -> Graph_ir.input_kind t.graph id = Graph_ir.Input.Input)
      t.graph.Graph_ir.Graph.inputs
  in
  let* inputs =
    match user_ids with
    | [ id ] -> Err.return [ (id, input) ]
    | ids ->
        Err.fail
          (`Unsupported_input (`Not_exactly_one_user_input (List.length ids)))
  in
  let* env =
    Eval_direct.run ~constants t.graph ~inputs
    |> Err.map_error ~pos:__POS__ (fun e -> `Eval e)
  in
  let+ outputs =
    Err.List.map
      (fun id ->
        Tensor_id.Map.find_opt id env
        |> Err.of_option (`Output_not_evaluated id))
      t.graph.Graph_ir.Graph.outputs
  in
  (outputs, { loaded with from_plan = materialized.applies })
