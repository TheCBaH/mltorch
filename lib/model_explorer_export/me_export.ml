(* Model bytes in, session document out. See the .mli. *)

type source_format = Model_json | Pt2_archive

(* CONTENT, never a declared extension: a mislabelled file is the ordinary case
   rather than the adversarial one, and the two loaders fail very differently on
   each other's input. *)
let detect ~bytes =
  if String.length bytes >= 4 && String.sub bytes 0 4 = "PK\003\004" then
    Err.return Pt2_archive
  else if String.length bytes >= 1 && bytes.[0] = '{' then Err.return Model_json
  else Err.fail `Unrecognised_format

(* Which ceiling applies is a fact about the CONTENT, the same rule [detect]
   uses. An unrecognised format falls back to [max_pt2_bytes]: [load] rejects
   it with [`Unrecognised_format] right after, so this check only needs to not
   itself be the reason a legitimately-typed file is missed. *)
let max_bytes_for ~limits bytes =
  match detect ~bytes with
  | Ok Model_json -> limits.Me_limits.Limits.max_json_bytes
  | Ok Pt2_archive | Error _ -> limits.Me_limits.Limits.max_pt2_bytes

type error =
  [ `Unrecognised_format
  | `Declared_format_disagrees
  | `Too_large of int64
  | `Document_too_large
    (** the ENCODED document overran [max_session_bytes]/[max_detail_bytes]. No
        byte count: the writer aborts mid-stream once the limit is hit, so
        unlike [`Too_large] there is no final length to report. *)
  | `Model_json_decode of string
  | `Archive of Pt2_archive.error
  | `Lowering of Native_interp.error
  | `Native4d of Native4d.Error.t
  | `Kernel of Kernel_adapt.error
  | `Unsupported_detail_key
  | `Value_graph of Me_kernel.error
  | `Fusion of Me_fusion.error
  | `Source_view of Me_source.error
  | `View of Graph_view.error
  | `Project of Me_build.error
  | `Navigation of Me_pt2.error
  | `Flow_graph of Me_flow_graph.error
  | `Verification of Me_verify.error
  | `Identifier of Me_ids.error
  | `Document of Me_session.Session.error ]

let pp_error fmt : [< error ] -> unit = function
  | `Unrecognised_format ->
      Fmt.string fmt "not a zip archive and not a JSON object"
  | `Declared_format_disagrees ->
      Fmt.string fmt "the bytes are not the format the request declared"
  | `Too_large n -> Fmt.pf fmt "%Ld bytes is over the ceiling" n
  | `Document_too_large ->
      Fmt.string fmt "the encoded document is over the ceiling"
  | `Model_json_decode m -> Fmt.pf fmt "model.json: %s" m
  | `Archive e -> Pt2_archive.pp_error fmt e
  | `Lowering e -> Native_interp.pp_error fmt e
  | `Native4d e -> Native4d.Error.pp fmt e
  | `Kernel e -> Kernel_adapt.pp_error fmt e
  | `Unsupported_detail_key ->
      Fmt.string fmt "the key names no value this model produces"
  | `Value_graph e -> Me_kernel.pp_error fmt e
  | `Fusion e -> Me_fusion.pp_error fmt e
  | `Source_view e -> Me_source.pp_error fmt e
  | `View e -> Graph_view.pp_error fmt e
  | `Project e -> Me_build.pp_error fmt e
  | `Navigation e -> Me_pt2.pp_error fmt e
  | `Flow_graph e -> Me_flow_graph.pp_error fmt e
  | `Verification e -> Me_verify.pp_error fmt e
  | `Identifier e -> Me_ids.pp_error fmt e
  | `Document e -> Me_session.Session.pp_error fmt e

(* The same line [Me_classify] draws: a model that is not what it claimed is
   [Invalid_source], a ceiling is [Over_limit], and an invariant of ours is
   [Internal]. A ceiling reached inside the projection is not our defect -- a
   real model can be too big, and that is a bound doing its job. *)
let diagnostic_code : [< error ] -> Me_limits.Diagnostic.Code.t =
  let module Code = Me_limits.Diagnostic.Code in
  function
  | `Unrecognised_format | `Declared_format_disagrees | `Model_json_decode _
  | `Archive _ ->
      Code.Invalid_source
  | `Too_large _ | `Document_too_large -> Code.Over_limit
  | `Unsupported_detail_key -> Code.Unsupported_detail_key
  | `Lowering _ | `Native4d _ | `Kernel _ -> Code.Internal
  | `Fusion (`Over_limit _)
  | `Value_graph (`Over_limit _)
  | `Source_view (`Over_limit _)
  | `Project (`Over_limit _)
  | `Navigation (`Over_limit _)
  | `Flow_graph (`Over_limit _)
  | `Verification (`Over_limit _)
  | `Document (`Over_limit _) ->
      Code.Over_limit
  | `Value_graph _ | `Source_view _ | `View _ | `Project _ | `Navigation _
  | `Flow_graph _ | `Verification _ | `Identifier _ | `Document _ ->
      Code.Internal

(* The one place a component's error is widened into this module's. Named,
   rather than a [Err.map_error ~pos:__POS__] at each of the twelve crossings, so the
   wrapping constructor is visible at the call site and the detection backtrace
   is preserved by [map_error] rather than rebuilt. *)
let wrap tag r = Err.map_error ~pos:__POS__ tag r
let ( let* ) = Err.Syntax.( let* )
let ( let+ ) = Err.Syntax.( let+ )

(* The PRODUCER side of the session/detail byte ceilings: a writer that
   refuses to grow past [max_bytes] rather than a check on the string after
   [encode_string] already built the whole thing. Same shape as
   [Me_response.Wire.encode_bounded], which bounds the META wrapper -- this
   bounds the document itself, which is the larger of the two and the one an
   oversized model can actually reach. No byte count on failure: the writer
   aborts mid-stream once the limit is hit, so there is no final length to
   report, same as [Wire]'s [`Meta_too_large]. *)
let encode_bounded ~max_bytes jsont v =
  let buf = Buffer.create 4096 in
  let w =
    Bytesrw.Bytes.Writer.limit max_bytes
      (Bytesrw.Bytes.Writer.of_buffer buf)
      ~eod:true
  in
  match Jsont_bytesrw.encode jsont v ~eod:true w with
  | Ok () -> Err.return (Buffer.contents buf)
  | Error _ -> Err.fail `Document_too_large
  | exception Bytesrw.Bytes.Stream.Error _ -> Err.fail `Document_too_large

let passes ~fold = [ Pipeline.canonical ~fold ]
let capability key status = { Me_session.Capability.key; status }

(* The outcome of a stage past Canonical, one step richer than the
   [(_, reason * string) result] the classified stages already used: that
   type has no way to say "never attempted" distinctly from "attempted and
   refused", and conflating them would report [Not_requested] rows as
   [Unavailable] with a manufactured reason, or vice versa. [Skipped] and
   [Refused] both read as absent to a comparison or a flow state -- the same
   place the two-case version already collapsed [Error] to "leave it out" --
   only the capability row needs to tell them apart. *)
type ('a, 'reason) staged =
  | Skipped
  | Refused of 'reason * string
  | Ready of 'a

let staged_status ~wanted staged payload =
  match (wanted, staged) with
  | false, _ -> Me_session.Capability.Not_requested
  | true, Skipped -> Me_session.Capability.Not_requested
  | true, Refused (reason, detail) ->
      Me_session.Capability.Unavailable { reason; detail = Some detail }
  | true, Ready a -> Me_session.Capability.Available (payload a)

(* The exported program's own distinct op targets -- what the field NAMES, and
   the one figure both shapes below can produce, since a session whose lowering
   failed has no native graph whose nodes it could count instead. *)
let op_targets (g : Pytorch_types.Graph.t) =
  List.map
    (fun (n : Pytorch_types.Node.t) -> n.Pytorch_types.Node.target)
    g.Pytorch_types.Graph.nodes
  |> List.sort_uniq compare |> List.length

(* Everything that differs between "we lowered this" and "we could not", as ONE
   record rather than six threaded options. Those are the only two shapes a
   session takes here, and six independent options describe combinations that
   are not sessions -- a flow spine with no Native state, a comparison naming a
   graph the collection does not hold. *)
type shape = {
  graphs : Model_explorer.Graph.t list;
  views : Me_session.View.t list;
  comparisons : Me_session.Comparison.t list;
  node_data_sets : Me_session.Node_data_set.t list;
  diagnostics : Me_limits.Diagnostic.t list;
      (** what a STAGE had to say beyond its capability row. The vector says
          which rows are missing; a diagnostic carries the bounded detail, and a
          stage that succeeded can still have something to report -- fusion's
          rejections are the case. *)
  flow : Me_flow.t option;
  capabilities : Me_session.Capability.t list;
  default_view : string;
}

(* A model this repository cannot lower still has a source view: decoding
   succeeded, which is the whole of what that row claims. So it is a SUCCESSFUL
   session whose capabilities say what is missing and why -- not a usage error,
   because the browser shell cannot surface one and the same code path serves
   both. *)
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
  (* [--fold] with no payloads bound is a fold that DECLINES every node:
     [Fold_const] refuses a constant whose payload is not there. So a request
     that asks to fold an archive has to preload it, and one that asks to fold a
     payload-free model.json reports [Requires_payloads] instead -- the
     capability and the pipeline saying the same thing. *)
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
         ~passes:(passes ~fold))
  in
  let (Native_interp.Transformed t) = transformed in
  let initial_id = Me_ids.graph Me_ids.Layer.Native 0 in
  let canonical_id = Me_ids.graph Me_ids.Layer.Native 1 in
  let* initial =
    wrap
      (fun e -> `Project e)
      (Me_native.graph ~limits ~id:initial_id lowered.Pt2_native_graph.graph)
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
      (Me_native.graph ~limits ~id:canonical_id ?group_attrs t.graph)
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
      match Native4d.Lower.convert ~constants:t.constants src with
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

(* --- the two loaders ----------------------------------------------------- *)

module Options = struct
  type t = {
    stages : Me_session.Capability.graph_stage list;
        (** {!session} only -- {!detail} reaches the kernel through its own
            smaller pipeline and does not read this field. *)
    fold : bool;
    verify_symbolic : Map_verify.Effort.t option;
    name : string;
    source_bytes : int64;
    source_sha256 : string option;
  }
end

(* Both end at a [Pt2_native_graph.t], which is what [transform_lowered] takes:
   the archive is a prerequisite of binding payloads and of nothing else, so a
   payload-free [model.json] reaches the same pipeline.

   The PROGRAM comes back too, not only what was lowered from it -- the source
   view is a projection of the exported graph itself, and it is the one stage
   that survives a lowering failure. And the lowering comes back as a RESULT
   rather than short-circuiting: a model this repository cannot lower is an
   ordinary outcome the capability matrix has a row for. *)
let classify_lowering result =
  match result with
  | Ok lowered -> Err.return (Ok lowered)
  | Error e -> (
      let detail =
        Core.Pretty.to_string Native_interp.pp_error (Err.Error.kind e)
      in
      match Me_classify.lowering (Err.Error.kind e) with
      | Me_classify.Unavailable reason -> Err.return (Error (reason, detail))
      | Me_classify.Fatal -> Err.fail (`Lowering (Err.Error.kind e)))

let load ~limits ~bytes =
  let* format = detect ~bytes in
  match format with
  | Pt2_archive ->
      let* archive =
        wrap
          (fun e -> `Archive e)
          (Pt2_archive.of_string ~limits:limits.Me_limits.Limits.zip
             ~name:"model.pt2" bytes)
      in
      let* lowering = classify_lowering (Native_interp.lower_archive archive) in
      Err.return
        ( Pt2_archive.program archive,
          lowering,
          Me_session.Model_summary.Pt2,
          Some archive )
  | Model_json ->
      let* program =
        Err.of_option (`Model_json_decode "not a well-formed ExportedProgram")
          (Result.to_option
             (Jsont_bytesrw.decode_string Pytorch_types.ExportedProgram.jsont
                bytes))
      in
      let* lowering = classify_lowering (Native_interp.lower program) in
      Err.return (program, lowering, Me_session.Model_summary.Json, None)

(* --- the session --------------------------------------------------------- *)

let session ~limits ~(options : Options.t) ~bytes =
  let* () =
    let n = Int64.of_int (String.length bytes) in
    if Int64.compare n (max_bytes_for ~limits bytes) > 0 then
      Err.fail (`Too_large n)
    else Err.return ()
  in
  let* program, lowering, source_kind, archive = load ~limits ~bytes in
  let graph_module = program.Pytorch_types.ExportedProgram.graph_module in
  let pt2_graph = graph_module.Pytorch_types.GraphModule.graph in
  let* source =
    wrap (fun e -> `Source_view e) (Me_source.graph ~limits graph_module)
  in
  let source_id = source.Model_explorer.Graph.id in
  let* label =
    wrap
      (fun e -> `Identifier e)
      (Me_ids.collection ~limits options.Options.name)
  in
  let source_view =
    {
      Me_session.View.id = "v/source";
      label = "Exported Program";
      kind = Me_session.View.Stage Me_session.Capability.Source;
      collection = label;
      graph = source_id;
    }
  in
  let* shape =
    match lowering with
    | Error (reason, detail) ->
        Err.return
          (unlowered_shape ~stages:options.Options.stages ~source ~source_id
             ~source_view ~reason ~detail)
    | Ok lowered ->
        lowered_shape ~limits ~label ~source ~source_id ~source_view ~pt2_graph
          ~source_kind ~fold:options.Options.fold
          ~verify_symbolic:options.Options.verify_symbolic ~archive
          ~stages:options.Options.stages lowered
  in
  let collection =
    Model_explorer.GraphCollection.create ~label ~graphs:shape.graphs ()
  in
  (* A diagnostic per unavailable-with-a-reason capability. The vector already
     says WHICH rows are missing; a diagnostic is what carries the free-text
     detail, bounded, in the one type that crosses every boundary here. *)
  let diagnostics =
    List.filter_map
      (fun (c : Me_session.Capability.t) ->
        match c.Me_session.Capability.status with
        | Me_session.Capability.Unavailable { reason; detail = Some detail } ->
            Some
              (Me_limits.Diagnostic.create ~limits ~graph:source_id
                 (Me_classify.diagnostic_code reason)
                 detail)
        | _ -> None)
      shape.capabilities
    @ shape.diagnostics
  in
  let session =
    {
      Me_session.Session.schema_version = 1;
      producer = { Me_session.Producer.tool = "mltorch"; session_schema = 1 };
      model =
        {
          Me_session.Model_summary.name = options.Options.name;
          (* the DETECTED kind, not a declared one: [--fold] on a
             payload-free model.json is decided by what the bytes are *)
          source_kind;
          source_bytes = options.Options.source_bytes;
          source_sha256 = options.Options.source_sha256;
          pt2_graph_count = 1;
          op_targets = op_targets pt2_graph;
        };
      graph_collections = [ collection ];
      views = shape.views;
      comparisons = shape.comparisons;
      node_data_sets = shape.node_data_sets;
      flow = shape.flow;
      capabilities = shape.capabilities;
      diagnostics;
      default_view = shape.default_view;
    }
  in
  let* () =
    wrap (fun e -> `Document e) (Me_session.Session.validate ~limits session)
  in
  Err.return session

(* --- one value's expression ---------------------------------------------- *)

(* A SMALLER pipeline than [session]'s, on purpose. A detail needs the kernel
   and nothing else -- no source view, no comparisons, no flow -- so projecting
   all of that to reach one value would make the on-demand path pay for the
   eager one it was introduced to avoid.

   It re-lowers rather than caching, which is what "self-contained request"
   means: the worker holds no session between requests, so there is nothing that
   could be stale. *)
let detail ~limits ~(options : Options.t) ~key ~bytes =
  let* () =
    let n = Int64.of_int (String.length bytes) in
    if Int64.compare n (max_bytes_for ~limits bytes) > 0 then
      Err.fail (`Too_large n)
    else Err.return ()
  in
  let* _, lowering, _, archive = load ~limits ~bytes in
  let* lowered =
    match lowering with
    | Ok l -> Err.return l
    | Error (reason, _) ->
        (* The one row that cannot become a capability here: a detail request
           about a model that does not lower is asking for something that was
           never offered, and the session it would have been offered by already
           said so. *)
        ignore reason;
        Err.fail `Unsupported_detail_key
  in
  let* constants =
    match (options.Options.fold, archive) with
    | false, _ | _, None -> Err.return Graph_ir.Tensor_id.Map.empty
    | true, Some archive ->
        wrap (fun e -> `Lowering e) (Native_interp.preload archive lowered)
  in
  let* transformed =
    wrap
      (fun e -> `Lowering e)
      (Native_interp.transform_lowered ~constants lowered
         ~passes:(passes ~fold:options.Options.fold))
  in
  let (Native_interp.Transformed t) = transformed in
  let* k =
    wrap
      (fun e -> `Kernel e)
      (Kernel_adapt.of_stage_program (Eval_symbolic.run t.graph))
  in
  let* value =
    Err.of_option `Unsupported_detail_key
      (Kernel.value k key.Me_request.Detail_key.value)
  in
  let* graph =
    wrap (fun e -> `Value_graph e) (Me_detail.of_value ~limits ~key value)
  in
  let* collection =
    wrap
      (fun e -> `Identifier e)
      (Me_ids.collection ~limits options.Options.name)
  in
  Err.return
    {
      Me_detail.Delta.schema_version = 1;
      collection;
      graph;
      view =
        {
          Me_session.View.id = Me_request.Detail_key.id key;
          label = "expression";
          kind = Me_session.View.Stage Me_session.Capability.Kernel;
          collection;
          graph = Me_request.Detail_key.id key;
        };
      node_data = [];
      diagnostics = [];
    }

(* --- the worker entry point ---------------------------------------------- *)

let declared_format_disagrees (source : Me_request.Source.t)
    (s : Me_session.Session.t) =
  let detected =
    match s.Me_session.Session.model.Me_session.Model_summary.source_kind with
    | Me_session.Model_summary.Json -> `Model_json
    | Me_session.Model_summary.Pt2 -> `Pt2
  in
  detected <> source.Me_request.Source.format

let handle ~emit request ~bytes =
  let id = Me_request.Request.id request in
  let key = Me_request.Request.key request in
  let limits =
    Me_limits.Wire_limits.limits (Me_request.Request.limits request)
  in
  let source = Me_request.Request.source request in
  let request_options = Me_request.Request.options request in
  let progress phase done_ total =
    emit { Me_response.Progress.id; phase; done_; total }
  in
  let failed (e : error) : Me_response.Handle_result.t =
    Me_response.Handle_result.Failed
      {
        Me_response.Failed.id;
        key;
        error =
          Me_limits.Diagnostic.create ~limits (diagnostic_code e)
            (Core.Pretty.to_string pp_error e);
      }
  in
  progress Me_response.Phase.Decode 0L None;
  let options =
    {
      Options.stages = request_options.Me_request.Options.stages;
      fold = request_options.Me_request.Options.fold;
      verify_symbolic = request_options.Me_request.Options.verify_symbolic;
      name = source.Me_request.Source.name;
      source_bytes = source.Me_request.Source.bytes;
      source_sha256 = Me_request.Source.verified_sha256 source;
    }
  in
  match Me_request.Request.key request with
  | Some key -> (
      match detail ~limits ~options ~key ~bytes with
      | Error e -> failed (Err.Error.kind e)
      | Ok d -> (
          match
            encode_bounded ~max_bytes:limits.Me_limits.Limits.max_detail_bytes
              Me_detail.Delta.jsont d
          with
          | Error e -> failed (Err.Error.kind e)
          | Ok json ->
              Me_response.Handle_result.Delta
                { Me_response.Delta.id; key; json }))
  | None -> (
      match session ~limits ~options ~bytes with
      | Error e -> failed (Err.Error.kind e)
      | Ok s when declared_format_disagrees source s ->
          (* The request DECLARED a format; the bytes are content-validated
         regardless, and a disagreement is a fact about the source rather than
         about the model. *)
          failed `Declared_format_disagrees
      | Ok s -> (
          progress Me_response.Phase.Encode 0L None;
          match
            encode_bounded ~max_bytes:limits.Me_limits.Limits.max_session_bytes
              Me_session.Session.jsont s
          with
          | Error e -> failed (Err.Error.kind e)
          | Ok json ->
              Me_response.Handle_result.Session
                {
                  Me_response.Session.id;
                  limits = Me_request.Request.limits request;
                  json;
                }))
