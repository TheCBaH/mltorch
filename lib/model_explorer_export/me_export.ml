(* Model bytes in, session document out. See the .mli.

   This file is now the public entry points only: Options, load, session,
   detail, handle. Small independent helpers (format detection, the error
   vocabulary, encode_bounded, the staged/shape types) live in
   me_export_types.ml, and the two shape-construction functions in
   me_export_shape.ml.. *)

type source_format = Me_export_types.source_format = Model_json | Pt2_archive

let detect = Me_export_types.detect

type error = Me_export_types.error

let pp_error = Me_export_types.pp_error
let diagnostic_code = Me_export_types.diagnostic_code
let encode_bounded = Me_export_types.encode_bounded

open Me_export_types
open Me_export_shape

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
        Jsont_bytesrw.decode_string Pytorch_types.ExportedProgram.jsont bytes
        |> Err.import ~pos:__POS__ (fun e -> `Model_json_decode e)
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
  let stage_program = Eval_symbolic.run t.graph in
  let* graph =
    match Me_request.Detail_key.value key with
    | Some value ->
        let* kernel =
          wrap
            (fun e -> `Kernel e)
            (Kernel_adapt.of_stage_program stage_program)
        in
        let* value =
          Err.of_option `Unsupported_detail_key (Kernel.value kernel value)
        in
        wrap (fun e -> `Value_graph e) (Me_detail.of_value ~limits ~key value)
    | None -> (
        match Me_request.Detail_key.operator_node key with
        | None -> Err.fail `Unsupported_detail_key
        | Some node_id ->
            let node =
              List.find_opt
                (fun (node : Graph_ir.node) ->
                  Graph_ir.Node_id.equal node.id node_id)
                t.graph.Graph_ir.Graph.nodes
            in
            let* node = Err.of_option `Unsupported_detail_key node in
            let stage id =
              List.find_opt
                (fun (stage : Stage_program.Stage.t) ->
                  Graph_ir.Tensor_id.equal stage.id id)
                stage_program.Stage_program.stages
            in
            let* stages =
              Err.List.map
                (fun output ->
                  Err.of_option `Unsupported_detail_key (stage output))
                node.Graph_ir.Node.outputs
            in
            let kernel = Kernel_adapt.of_stage_program stage_program in
            let values, kernel_available =
              match kernel with
              | Ok kernel ->
                  ( List.map
                      (fun (stage : Stage_program.Stage.t) ->
                        Option.value
                          ~default:
                            {
                              Kernel.Value.id = stage.id;
                              sg = stage.sg;
                              computation = stage.computation;
                              result = Kernel.Result_conversion.Round_f32;
                            }
                          (Kernel.value kernel stage.id))
                      stages,
                    true )
              | Error _ ->
                  ( List.map
                      (fun (stage : Stage_program.Stage.t) ->
                        {
                          Kernel.Value.id = stage.id;
                          sg = stage.sg;
                          computation = stage.computation;
                          result = Kernel.Result_conversion.Round_f32;
                        })
                      stages,
                    false )
            in
            let* graph =
              wrap
                (fun e -> `Value_graph e)
                (Me_detail.of_operator ~limits ~key ~outputs:values)
            in
            let graph =
              if kernel_available then graph
              else
                {
                  graph with
                  Model_explorer.Graph.nodes =
                    List.map
                      (fun (node : Model_explorer.GraphNode.t) ->
                        if
                          String.length node.id > 3
                          && String.sub node.id (String.length node.id - 3) 3
                             = "/e0"
                        then
                          {
                            node with
                            Model_explorer.GraphNode.label =
                              "kernel boundary unavailable";
                          }
                        else node)
                      graph.nodes;
                }
            in
            Err.return graph)
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
          kind =
            Me_session.View.Detail
              {
                parent_graph = Me_request.Detail_key.parent_graph key;
                parent_node = Me_request.Detail_key.session_node key;
              };
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
