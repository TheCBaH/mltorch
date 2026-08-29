(* The two shape-construction functions: [unlowered_shape] for a model this
   repository decoded but could not lower, [lowered_shape] for the full
   pipeline past Canonical. Split out of me_export.ml; see
   me_export_types.ml for the [staged]/[shape] vocabulary and me_export.ml
   for the public entry points that call these. *)

open Me_export_types

let unlowered_shape ~stages ~source ~source_id ~source_view ~reason ~detail =
  let module C = Me_session.Capability in
  let wanted stage = List.mem stage stages in
  {
    graphs = [ source ];
    views = [ source_view ];
    comparisons = [];
    node_data_sets = [];
    diagnostics = [];
    (* No flow. With no Native state the spine would hold [s/pt2/000] and no
       transitions, and a one-node flow graph asserts a navigability that does
       not exist. *)
    flow = None;
    capabilities =
      List.map
        (fun k ->
          capability k
            (match k with
            | C.Graph_stage C.Source -> C.Available (C.Graph source_id)
            (* The key lowering itself decides. "Its prerequisite is
               unavailable" would be circular for the row that IS the lowering,
               and the classified reason is the one a user can act on.
               Unconditional on [wanted], matching [lowered_shape]'s treatment
               of the same two rows as an always-computed backbone rather than
               a stage a caller opts into. *)
            | C.Graph_stage C.Initial_native ->
                C.Unavailable { reason; detail = Some detail }
            | C.Feature C.Expression_detail -> C.Available C.Present
            | C.Feature C.Loop_ir | C.Feature C.Codegen ->
                C.Unavailable { reason = C.Not_implemented; detail = None }
            (* The four stages [lowered_shape] gates on [stages] too, checked
               here BEFORE the blanket "everything downstream is blocked" rule
               below: a stage nobody asked for is [Not_requested] whether or
               not lowering ever reached far enough to say anything else about
               it. *)
            | C.Graph_stage
                ((C.Native4d | C.Stage_program | C.Kernel | C.Fusion) as s)
              when not (wanted s) ->
                C.Not_requested
            (* Everything downstream, enumerated through the set [Me_classify]
               owns rather than restated here. *)
            | k when Me_classify.depends_on_lowering k ->
                C.Unavailable
                  { reason = C.Prerequisite_unavailable; detail = None }
            | _ -> C.Not_requested))
        C.all_keys;
    default_view = "v/source";
  }

let lowered_shape ~limits ~label ~source ~source_id ~source_view ~pt2_graph
    ~source_kind ~fold ~verify_symbolic ~archive ~stages lowered =
  let module C = Me_session.Capability in
  (* Which of the four stages past Canonical the CALLER asked for, directly or
     transitively: Fusion needs Kernel, Kernel needs the stage program. Source,
     Initial_native and Canonical are not gated here -- they are the backbone
     every comparison and every one of these four is built from, and cheap
     projections rather than the symbolic evaluation and kernel/fusion
     construction the review flagged as the expensive, skippable work. A stage
     NOT in this set is [Skipped] below rather than computed and then hidden:
     the point is not spending the work, not merely not showing the result. *)
  let wanted stage = List.mem stage stages in
  let needed_kernel = wanted C.Kernel || wanted C.Fusion in
  let needed_stage_program = wanted C.Stage_program || needed_kernel in
  let needed_native4d = wanted C.Native4d in
  (* Canonicalization is symbolic and archive-free. [fold] only controls
     whether an archive-backed caller preloads a materialization cache for a
     later explicit evaluation; it cannot select a different canonical graph. *)
  let* constants =
    match (fold, archive) with
    | false, _ | _, None -> Err.return Graph_ir.Tensor_id.Map.empty
    | true, Some archive ->
        wrap (fun e -> `Lowering e) (Native_interp.preload archive lowered)
  in
  let* transformed =
    wrap
      (fun e -> `Lowering e)
      (Native_interp.transform_lowered ~constants lowered
         ?verify:
           (Option.map
              (fun _ -> Map_verify.Policy.Reject_refuted)
              verify_symbolic)
         ?verify_budget:(Option.map Map_verify.Effort.budget verify_symbolic)
         ?verify_probe:(Option.map Map_verify.Effort.probe verify_symbolic)
         ?max_verified_steps:
           (Option.map Map_verify.Effort.max_verified_steps verify_symbolic)
         ?max_verify_clusters:
           (Option.map Map_verify.Effort.max_clusters verify_symbolic)
         ~passes:(passes ~fold))
  in
  let (Native_interp.Transformed t) = transformed in
  let initial_id = Me_ids.graph Me_ids.Layer.Native 0 in
  let canonical_id = Me_ids.graph Me_ids.Layer.Native 1 in
  let* initial =
    wrap
      (fun e -> `Project e)
      (Me_native.graph ~limits ~id:initial_id ~constant_store:t.constant_store
         lowered.Pt2_native_graph.graph)
  in
  (* The per-group rollup, keyed by the SAME namespaces the projection emits --
     which is why it comes from [Me_verify] rather than from [Map_verify]'s own
     label-only [Group_path], and why the projection hands out [namespace_of]
     rather than each side computing the key. *)
  let* group_attrs =
    match t.composed with
    | None -> Err.return None
    | Some report ->
        let* rows =
          wrap
            (fun e -> `Verification e)
            (Me_verify.by_namespace ~limits t.graph report)
        in
        Err.return (Some rows)
  in
  let* canonical =
    wrap
      (fun e -> `Project e)
      (Me_native.graph ~limits ~id:canonical_id ?group_attrs
         ~constant_store:t.constant_store t.graph)
  in
  (* The four-axis dialect, from CANONICAL Native -- the dataflow branches
     there, and this is one of the two branches. A graph outside the dialect's
     domain is the partiality [Me_classify.native4d] exists to classify, not a
     failure: it becomes a capability carrying the reason, exactly as an
     unsupported operator does one stage earlier. *)
  let native4d_id = Me_ids.graph Me_ids.Layer.Native4d 0 in
  let* native4d =
    if not needed_native4d then Err.return Skipped
    else
      let* snapshot = wrap (fun e -> `View e) (Snapshot.create t.graph) in
      let (Snapshot.Pack src) = snapshot in
      match
        Native4d.Lower.convert ~constants:t.constants
          ~constant_store:t.constant_store src
      with
      | Error e -> (
          match Me_classify.native4d (Err.Error.kind e) with
          | Me_classify.Unavailable reason ->
              Err.return
                (Refused
                   ( reason,
                     Core.Pretty.to_string Native4d.Error.pp (Err.Error.kind e)
                   ))
          | Me_classify.Fatal -> Err.fail (`Native4d (Err.Error.kind e)))
      | Ok (Native4d.Lower.Pack r) ->
          let+ g =
            wrap
              (fun e -> `Project e)
              (Me_native4d.graph ~limits ~id:native4d_id
                 ~constant_store:r.Native4d.Lower.constant_store
                 (Native4d.Lower.graph r))
          in
          Ready g
  in
  (* The OTHER branch from canonical Native: symbolic stages, then the kernel
     adapted from them. Both are value graphs, and a stage keeps the id of the
     kernel value adapted from it, so the two panes pair without a mapping. *)
  let stage_id = Me_ids.graph Me_ids.Layer.Symbolic 0 in
  let kernel_id = Me_ids.graph Me_ids.Layer.Kernel 0 in
  (* [Eval_symbolic.run] is the expensive symbolic evaluation the review
     flagged, not [Me_kernel.stage_program]'s projection of it -- the two are
     gated together since the projection has nothing to project without it. *)
  let* program_and_stage_graph =
    if not needed_stage_program then Err.return None
    else
      let program = Eval_symbolic.run t.graph in
      let+ stage_graph =
        wrap
          (fun e -> `Value_graph e)
          (Me_kernel.stage_program ~limits ~id:stage_id program)
      in
      Some (program, stage_graph)
  in
  let stage_graph = Option.map snd program_and_stage_graph in
  let* kernel_graph =
    if not needed_kernel then Err.return Skipped
    else
      (* [needed_kernel] implies [needed_stage_program] by construction, so
         [program_and_stage_graph] is always [Some] here. *)
      let program, _ = Option.get program_and_stage_graph in
      match Kernel_adapt.of_stage_program program with
      | Error e -> (
          match Me_classify.kernel (Err.Error.kind e) with
          | Me_classify.Unavailable reason ->
              Err.return
                (Refused
                   ( reason,
                     Core.Pretty.to_string Kernel_adapt.pp_error
                       (Err.Error.kind e) ))
          | Me_classify.Fatal -> Err.fail (`Kernel (Err.Error.kind e)))
      | Ok k ->
          (* The fusion PLAN comes with the kernel, because it is a view over
             that very kernel -- [Fusion_plan.t] carries the kernel it
             planned over so a plan cannot be applied to a different one, and
             separating them here would put that back at risk. *)
          let* g =
            wrap
              (fun e -> `Value_graph e)
              (Me_kernel.kernel ~limits ~id:kernel_id k)
          in
          let+ f =
            wrap
              (fun e -> `Fusion e)
              (Me_fusion.of_kernel ~limits ~graph:kernel_id k)
          in
          (* The overlay rides ON the kernel graph, because that is the graph
             it is an overlay over. A comparison's overlay slot would make it
             a fact about two panes rather than about one plan. *)
          Ready
            ( {
                g with
                Model_explorer.Graph.tasksData =
                  Some
                    (Model_explorer.TasksData.create
                       ~edgeOverlaysDataListLeftPane:[ f.Me_fusion.overlays ] ());
              },
              f )
  in
  (* The fusion node data and its rejection diagnostics reach the SESSION, while
     the overlay rides on the kernel graph above. Two facts about one plan, in
     the two places the wire format has for them. *)
  let fusion_data =
    match kernel_graph with
    | Skipped | Refused _ -> []
    | Ready (_, f) -> [ f.Me_fusion.node_data ]
  in
  let fusion_diagnostics =
    match kernel_graph with
    | Skipped | Refused _ -> []
    | Ready (_, f) -> f.Me_fusion.rejections
  in
  let* node_data_sets =
    match t.composed with
    | None -> Err.return []
    | Some report ->
        let* set =
          wrap
            (fun e -> `Verification e)
            (Me_verify.node_data ~limits ~graph:canonical_id t.graph report)
        in
        Err.return [ set ]
  in
  (* One comparison, with NO explicit mapping entries, and that is correct
     rather than approximate: entries for the changed nodes are what §7.3 adds
     from the pass lens, and until then the empty set says exactly what is
     known. Entries alone would be a LIE about the rest of the graph, which is
     what [match_node_id_fallback] exists to prevent -- stable node ids already
     pair every node a pass did not touch, and the id-identity rule means a
     changed value is a new id, so here equal ids ARE a correspondence claim and
     the comparison declares it.

     Declared rather than left to the renderer's default because the two facts
     are not separable downstream: a consumer that disabled the fallback on this
     comparison would give every node an empty mapped set, which the renderer
     reads as "changed" -- turning `show_diff_highlights` from a report of the
     handful of touched nodes into a claim that the pass rewrote everything. *)
  let comparison =
    {
      Me_session.Comparison.id = "c/canonical";
      label = "initial -> canonical";
      left = { Me_session.Pane_state.collection = label; graph = initial_id };
      right = { Me_session.Pane_state.collection = label; graph = canonical_id };
      sync =
        {
          Me_session.Sync_navigation.entries = [];
          show_diff_highlights = true;
          match_node_id_fallback = true;
        };
      overlays_left = [];
      overlays_right = [];
    }
  in
  (* The import comparison, which DOES carry entries: the two panes speak
     different id languages, so MATCH_NODE_ID pairs nothing across them and
     every correspondence has to be stated. They come from the sidecar as
     connected components, which is also what makes a decomposition and a fold
     one entry rather than several overlapping ones. *)
  let* navigation =
    wrap
      (fun e -> `Navigation e)
      (Me_pt2.of_origins ~limits
         ~source_nodes:(Me_source.node_ids pt2_graph)
         lowered.Pt2_native_graph.node_origins)
  in
  let import_comparison =
    {
      Me_session.Comparison.id = "c/import";
      label = "exported program -> initial";
      left = { Me_session.Pane_state.collection = label; graph = source_id };
      right = { Me_session.Pane_state.collection = label; graph = initial_id };
      sync = Me_pt2.sync navigation;
      overlays_left = [];
      overlays_right = [];
    }
  in
  let state id graph view label produced_by =
    {
      Me_flow.State.id;
      graph;
      view;
      layer = Me_ids.Layer.Native;
      label;
      produced_by;
    }
  in
  (* The DAG branches at canonical Native, and this is the branch that exists so
     far. When the conversion is outside the dialect there is no state to show:
     a spine node for a graph nobody can open asserts a navigability that is not
     there, which is the same rule the unlowered session follows. *)
  (* The symbolic and kernel states, present exactly when their graph is: a
     spine node for a graph nobody can open -- whether never built because
     nobody asked, or refused with a reason -- asserts a navigability that is
     not there, the same rule the four-axis branch follows. *)
  let symbolic_states =
    (match stage_graph with
      | None -> []
      | Some _ ->
          [
            {
              Me_flow.State.id = Me_ids.flow_state Me_ids.Layer.Symbolic 0;
              graph = stage_id;
              view = "v/stage_program";
              layer = Me_ids.Layer.Symbolic;
              label = "stage program";
              produced_by =
                Some (Me_ids.flow_transition Me_ids.Layer.Symbolic 0);
            };
          ])
    @
    match kernel_graph with
    | Skipped | Refused _ -> []
    | Ready _ ->
        [
          {
            Me_flow.State.id = Me_ids.flow_state Me_ids.Layer.Kernel 0;
            graph = kernel_id;
            view = "v/kernel";
            layer = Me_ids.Layer.Kernel;
            label = "kernel";
            produced_by = Some (Me_ids.flow_transition Me_ids.Layer.Kernel 0);
          };
        ]
  in
  let symbolic_transitions =
    (match stage_graph with
      | None -> []
      | Some _ ->
          [
            {
              Me_flow.Transition.id =
                Me_ids.flow_transition Me_ids.Layer.Symbolic 0;
              before = Me_ids.flow_state Me_ids.Layer.Native 1;
              after = Me_ids.flow_state Me_ids.Layer.Symbolic 0;
              kind = Me_flow.Transition.Adapt;
              comparison = None;
            };
          ])
    @
    match kernel_graph with
    | Skipped | Refused _ -> []
    | Ready _ ->
        [
          {
            Me_flow.Transition.id = Me_ids.flow_transition Me_ids.Layer.Kernel 0;
            before = Me_ids.flow_state Me_ids.Layer.Symbolic 0;
            after = Me_ids.flow_state Me_ids.Layer.Kernel 0;
            kind = Me_flow.Transition.Adapt;
            comparison = None;
          };
        ]
  in
  let native4d_state, native4d_transition =
    match native4d with
    | Skipped | Refused _ -> ([], [])
    | Ready _ ->
        ( [
            {
              Me_flow.State.id = Me_ids.flow_state Me_ids.Layer.Native4d 0;
              graph = native4d_id;
              view = "v/native4d";
              layer = Me_ids.Layer.Native4d;
              label = "native4d";
              produced_by =
                Some (Me_ids.flow_transition Me_ids.Layer.Native4d 0);
            };
          ],
          [
            {
              Me_flow.Transition.id =
                Me_ids.flow_transition Me_ids.Layer.Native4d 0;
              before = Me_ids.flow_state Me_ids.Layer.Native 1;
              after = Me_ids.flow_state Me_ids.Layer.Native4d 0;
              kind = Me_flow.Transition.Cross_dialect;
              comparison = None;
            };
          ] )
  in
  let flow =
    {
      Me_flow.states =
        [
          {
            Me_flow.State.id = Me_ids.flow_state Me_ids.Layer.Pt2 0;
            graph = source_id;
            view = source_view.Me_session.View.id;
            layer = Me_ids.Layer.Pt2;
            label = "exported program";
            produced_by = None;
          };
          state
            (Me_ids.flow_state Me_ids.Layer.Native 0)
            initial_id "v/initial" "initial"
            (Some (Me_ids.flow_transition Me_ids.Layer.Native 0));
          state
            (Me_ids.flow_state Me_ids.Layer.Native 1)
            canonical_id "v/canonical" "canonical"
            (Some (Me_ids.flow_transition Me_ids.Layer.Native 1));
        ]
        @ native4d_state @ symbolic_states;
      transitions =
        [
          {
            Me_flow.Transition.id = Me_ids.flow_transition Me_ids.Layer.Native 0;
            before = Me_ids.flow_state Me_ids.Layer.Pt2 0;
            after = Me_ids.flow_state Me_ids.Layer.Native 0;
            kind = Me_flow.Transition.Import;
            comparison = Some "c/import";
          };
          {
            Me_flow.Transition.id = Me_ids.flow_transition Me_ids.Layer.Native 1;
            before = Me_ids.flow_state Me_ids.Layer.Native 0;
            after = Me_ids.flow_state Me_ids.Layer.Native 1;
            kind = Me_flow.Transition.Pack;
            comparison = Some "c/canonical";
          };
        ]
        @ native4d_transition @ symbolic_transitions;
      graph = "g/flow";
    }
  in
  (* The spine, as the graph that renders it. A flow DESTINATION is three facts
     that must agree -- this graph, the [View.Flow] below, and the
     [feature:flow] capability -- and [Session.validate] is what checks they
     do. *)
  let* flow_graph =
    wrap (fun e -> `Flow_graph e) (Me_flow_graph.graph ~limits flow)
  in
  (* Two keys, not one. [composed] survives per-pass audit truncation, so
     mapping that truncation to a single [Verification -> Over_limit] would hide
     a result that is still available, while [Available] alone would conceal
     that the per-pass detail is incomplete. *)
  let* verification =
    match t.composed with
    | None -> Err.return C.Not_requested
    | Some report ->
        let* counts =
          wrap (fun e -> `Verification e) (Me_verify.summary report)
        in
        Err.return (C.Available (C.Verification_summary counts))
  in
  let pass_audits =
    if verify_symbolic = None then C.Not_requested
    else C.Available (C.Pass_audit_status (Me_verify.audit_status t.audits))
  in
  Err.return
    {
      graphs =
        [ source; initial; canonical ]
        @ (match stage_graph with None -> [] | Some g -> [ g ])
        @ (match native4d with Ready g -> [ g ] | Skipped | Refused _ -> [])
        @ (match kernel_graph with
          | Ready (g, _) -> [ g ]
          | Skipped | Refused _ -> [])
        @ [ flow_graph ];
      views =
        ([
           {
             Me_session.View.id = "v/canonical";
             label = "Canonical Native";
             kind = Me_session.View.Stage C.Canonical;
             collection = label;
             graph = canonical_id;
           };
           {
             Me_session.View.id = "v/initial";
             label = "Initial Native";
             kind = Me_session.View.Stage C.Initial_native;
             collection = label;
             graph = initial_id;
           };
           source_view;
         ]
        @
        match native4d with
        | Skipped | Refused _ -> []
        | Ready _ ->
            [
              {
                Me_session.View.id = "v/native4d";
                label = "Native4D";
                kind = Me_session.View.Stage C.Native4d;
                collection = label;
                graph = native4d_id;
              };
            ])
        @ (match stage_graph with
          | None -> []
          | Some _ ->
              [
                {
                  Me_session.View.id = "v/stage_program";
                  label = "Stage Program";
                  kind = Me_session.View.Stage C.Stage_program;
                  collection = label;
                  graph = stage_id;
                };
              ])
        @ (match kernel_graph with
          | Skipped | Refused _ -> []
          | Ready _ ->
              [
                {
                  Me_session.View.id = "v/kernel";
                  label = "Kernel";
                  kind = Me_session.View.Stage C.Kernel;
                  collection = label;
                  graph = kernel_id;
                };
              ])
        (* Never the default: the browser stays source-first and the CLI keeps
           canonical Native. *)
        @ [
            {
              Me_session.View.id = "v/flow";
              label = "Export flow";
              kind = Me_session.View.Flow;
              collection = label;
              graph = flow.Me_flow.graph;
            };
          ];
      comparisons = [ import_comparison; comparison ];
      node_data_sets = node_data_sets @ fusion_data;
      diagnostics = fusion_diagnostics;
      flow = Some flow;
      capabilities =
        List.map
          (fun k ->
            capability k
              (match k with
              | C.Graph_stage C.Initial_native ->
                  C.Available (C.Graph initial_id)
              | C.Graph_stage C.Canonical -> C.Available (C.Graph canonical_id)
              | C.Feature C.Fold
                when source_kind = Me_session.Model_summary.Json ->
                  (* A structural success carrying a reason, not a usage error:
                     the browser cannot surface one and the same path serves
                     both shells. *)
                  C.Unavailable
                    {
                      reason = Me_classify.requires_payloads_without_them;
                      detail = None;
                    }
              | C.Feature C.Fold when fold -> C.Available C.Present
              | C.Graph_stage C.Native4d ->
                  staged_status ~wanted:(wanted C.Native4d) native4d (fun _ ->
                      C.Graph native4d_id)
              (* [wanted] alone, not [stage_graph <> None]: a client that asked
                 only for [Kernel] gets a stage program computed as ITS
                 prerequisite, but did not ask to see it, and a future
                 implementation that reached [Kernel] without materialising
                 this graph should not turn today's byproduct into tomorrow's
                 broken promise. *)
              | C.Graph_stage C.Stage_program ->
                  if wanted C.Stage_program then C.Available (C.Graph stage_id)
                  else C.Not_requested
              | C.Graph_stage C.Kernel ->
                  staged_status ~wanted:(wanted C.Kernel) kernel_graph (fun _ ->
                      C.Graph kernel_id)
              (* Fusion is a stage OF the kernel layer, so it cannot be
                 available where the kernel is not -- and where the kernel is,
                 the plan is a view over an unchanged kernel that has not landed
                 yet. Two different unavailabilities, and saying which is the
                 point of having both reasons. *)
              (* A VIEW over the unchanged kernel graph, so it names that
                 graph rather than one of its own: a fused graph would be a
                 graph nothing produced, and a reader comparing it against the
                 kernel would be comparing a rendering with the thing it
                 renders. *)
              | C.Graph_stage C.Fusion -> (
                  if not (wanted C.Fusion) then C.Not_requested
                  else
                    (* [wanted C.Fusion] forces [needed_kernel], so
                       [kernel_graph] is never [Skipped] here -- kept for
                       totality, not because it is reachable. *)
                    match kernel_graph with
                    | Ready _ -> C.Available (C.Graph kernel_id)
                    | Refused _ | Skipped ->
                        C.Unavailable
                          { reason = C.Prerequisite_unavailable; detail = None }
                  )
              (* The spine is IN the document, so this is available and names
                 the graph it renders as -- [Not_requested] would say nobody
                 asked for something the session already carries. *)
              | C.Feature C.Flow -> C.Available (C.Graph "g/flow")
              | C.Feature C.Verification -> verification
              | C.Feature C.Pass_audits -> pass_audits
              | C.Feature C.Expression_detail -> C.Available C.Present
              | C.Feature C.Loop_ir | C.Feature C.Codegen ->
                  C.Unavailable { reason = C.Not_implemented; detail = None }
              (* The one unconditional row of the matrix: it claims that the
                 exported program parsed, and nothing more. *)
              | C.Graph_stage C.Source -> C.Available (C.Graph source_id)
              | _ -> C.Not_requested))
          C.all_keys;
      default_view = "v/canonical";
    }
