(* Payload-free model-support sweep, typed. The counterpart of
   test/pt2_model_support_cram.t over every model.json in the pinned producer
   submodule (modules/devcontainer.pytorch-image-models) -- no download
   needed, so it covers ~100 models where the cram (real .pt2 archives, gated
   on PT2_DATA) covers 6.

   Calls Me_export.session directly -- the same library visualize's CLI
   command calls -- rather than shelling out to native_graph.exe and
   re-parsing its JSON/stderr with jq and shell prefix-matching: the result is
   read off the typed Me_session.Capability.t/Me_export.error values the
   library already computes, so there is no message format for this tool and
   the CLI to drift apart on.

   One compact JSON object per model, sorted by name. Past Canonical the
   pipeline FORKS into two independent branches (see me_export.ml's "the
   OTHER branch from canonical Native"), which is why there are two
   self-contained (converts, stage, reason, blocker) triples rather than one:
   - native4d_*: the four-axis dialect conversion (stage:native4d).
   - kernel_*:   Stage Program -> Kernel -> Kernel Fusion (the Model Explorer
                 web UI's other three graph options). stage:stage_program has
                 no Unavailable case of its own in Me_export -- a failure
                 there is Fatal, not a gradeable capability -- and
                 stage:fusion always exactly mirrors stage:kernel's
                 Ready/Refused (Fusion needs the same [Kernel_adapt.t] the
                 kernel graph was built from, per me_export.ml's "the fusion
                 PLAN comes with the kernel"), so reading stage:kernel alone
                 is a complete, correct read of the whole branch's outcome
                 for a request that always asks for [all_stages] -- which
                 this tool always does.

   Each triple's `stage`/`reason`/`blocker` name WHERE that branch stopped and
   WHY, resolved to the deepest real cause -- a branch's own capability of
   reason `prerequisite_unavailable` (which carries no detail of its own)
   resolves to `stage:initial_native`'s own reason instead:
   - "source": model.json itself didn't decode (reason "model_json_decode").
   - "native": the payload-free Native import failed -- either recoverably
               (reason "unsupported_operator"/"unsupported_input"/
               "over_limit", a session still builds) or as a defect the
               importer flags outright (reason "malformed": an argument
               value, e.g. max_pool2d's ceil_mode or clone's memory_format,
               that the importer rejects rather than merely not implementing
               -- Me_classify.lowering's Fatal rows). BOTH branches resolve
               to this stage identically when Native import itself is the
               blocker, since neither can proceed without it.
   - "native4d" (native4d_* only): Native4D itself rejected the graph on its
               own terms (reason "outside_dialect_domain" or
               "requires_payloads").
   - "kernel" (kernel_* only): Kernel_adapt itself rejected the stage program
               (reason "unsupported_graph_shape" or "over_limit").
   All four fields of a triple are null on that branch's full success.

   Usage: pt2_json_model_support.exe <models_dir> <output.jsonl> *)

module C = Me_session.Capability

let reason_name : C.reason -> string = function
  | C.Unsupported_operator -> "unsupported_operator"
  | C.Unsupported_input -> "unsupported_input"
  | C.Unsupported_graph_shape -> "unsupported_graph_shape"
  | C.Outside_dialect_domain -> "outside_dialect_domain"
  | C.Over_limit -> "over_limit"
  | C.Requires_payloads -> "requires_payloads"
  | C.Prerequisite_unavailable -> "prerequisite_unavailable"
  | C.Not_implemented -> "not_implemented"

type branch = {
  converts : bool option;
  stage : string option;
  reason : string option;
  blocker : string option;
}

let branch_json (b : branch) prefix =
  let open Jsont.Json in
  [
    mem (name (prefix ^ "_converts")) (option bool b.converts);
    mem (name (prefix ^ "_stage")) (option string b.stage);
    mem (name (prefix ^ "_reason")) (option string b.reason);
    mem (name (prefix ^ "_blocker")) (option string b.blocker);
  ]

type row = {
  model : string;
  native_builds : bool;
  nodes : int option;
  native4d : branch;
  kernel : branch;
}

let row_json (r : row) =
  let open Jsont.Json in
  object'
    ([
       mem (name "model") (string r.model);
       mem (name "native_builds") (bool r.native_builds);
       mem (name "nodes") (option int r.nodes);
     ]
    @ branch_json r.native4d "native4d"
    @ branch_json r.kernel "kernel")

(* [caps] always has one entry per [C.all_keys] (Me_export.session computes
   every key unconditionally when [stages = all_stages], which this tool
   always passes) -- [Not_requested] below is unreachable in practice and
   only guards against that assumption ever changing silently. *)
let status_of caps key =
  match List.find_opt (fun (c : C.t) -> c.C.key = key) caps with
  | Some c -> c.C.status
  | None -> C.Not_requested

let node_count (session : Me_session.Session.t) =
  List.concat_map
    (fun (gc : Model_explorer.GraphCollection.t) ->
      gc.Model_explorer.GraphCollection.graphs)
    session.Me_session.Session.graph_collections
  |> List.find_opt (fun (g : Model_explorer.Graph.t) ->
      g.Model_explorer.Graph.id = "pt2/root")
  |> Option.map (fun (g : Model_explorer.Graph.t) ->
      List.length g.Model_explorer.Graph.nodes)

(* One branch's outcome, given its own DEEPEST capability status and
   [initial_native]'s -- the shared root every branch forks from, and the
   fallback whenever [own] is a bare [Prerequisite_unavailable] propagation
   rather than a reason of the branch's own. *)
let resolve_branch ~own_stage_name ~(own : C.status) ~(native : C.status) =
  match own with
  | C.Available _ ->
      { converts = Some true; stage = None; reason = None; blocker = None }
  | C.Unavailable { reason = C.Prerequisite_unavailable; _ } ->
      let reason, blocker =
        match native with
        | C.Unavailable { reason; detail } -> (Some (reason_name reason), detail)
        | C.Available _ | C.Not_requested -> (None, None)
      in
      { converts = None; stage = Some "native"; reason; blocker }
  | C.Unavailable { reason; detail } ->
      {
        converts = Some false;
        stage = Some own_stage_name;
        reason = Some (reason_name reason);
        blocker = detail;
      }
  | C.Not_requested ->
      { converts = None; stage = None; reason = None; blocker = None }

let row_of_session ~model (session : Me_session.Session.t) =
  let caps = session.Me_session.Session.capabilities in
  let native = status_of caps (C.Graph_stage C.Initial_native) in
  let native4d_status = status_of caps (C.Graph_stage C.Native4d) in
  let kernel_status = status_of caps (C.Graph_stage C.Kernel) in
  let native_builds = match native with C.Available _ -> true | _ -> false in
  {
    model;
    native_builds;
    nodes = node_count session;
    native4d =
      resolve_branch ~own_stage_name:"native4d" ~own:native4d_status ~native;
    kernel = resolve_branch ~own_stage_name:"kernel" ~own:kernel_status ~native;
  }

(* Every failure [Me_export.session] can return where the input was a plain
   model.json (never a .pt2 archive here, so the Archive/Navigation rows
   never fire): a decode failure is the "source" stage, a Fatal [Lowering] or
   [Native4d] error is that stage flagging the graph as a defect rather than
   merely unsupported (Me_classify's Fatal rows), and everything else is a
   resource ceiling or an internal invariant this corpus is not expected to
   hit -- bucketed as "process" so a real occurrence is visible rather than
   silently mis-attributed to a pipeline stage it didn't reach. Both branches
   share the SAME (stage, reason, blocker) here: neither can run at all
   without a session, so there is only one cause to report, not two. *)
let row_of_error ~model (e : Me_export.error) =
  let blocker = Format.asprintf "%a" Me_export.pp_error e in
  let stage, reason =
    match e with
    | `Model_json_decode _ -> ("source", "model_json_decode")
    | `Lowering _ -> ("native", "malformed")
    | `Native4d _ -> ("native4d", "malformed")
    | `Too_large _ | `Document_too_large | `Unrecognised_format
    | `Declared_format_disagrees | `Archive _ | `Kernel _
    | `Unsupported_detail_key | `Value_graph _ | `Fusion _ | `Source_view _
    | `View _ | `Project _ | `Navigation _ | `Flow_graph _ | `Verification _
    | `Identifier _ | `Document _ ->
        ("process", "process_error")
  in
  let shared =
    {
      converts = None;
      stage = Some stage;
      reason = Some reason;
      blocker = Some blocker;
    }
  in
  {
    model;
    native_builds = false;
    nodes = None;
    native4d = shared;
    kernel = shared;
  }

let read path = In_channel.with_open_bin path In_channel.input_all

let row_of_model models_dir model =
  let path =
    Filename.concat (Filename.concat models_dir model) "models/model.json"
  in
  let bytes = read path in
  let options =
    {
      Me_export.Options.stages = C.all_stages;
      fold = false;
      verify_symbolic = None;
      name = model;
      source_bytes = Int64.of_int (String.length bytes);
      source_sha256 = None;
    }
  in
  match
    Me_export.session ~limits:Me_limits.Limits.untrusted ~options ~bytes
  with
  | Ok session -> row_of_session ~model session
  | Error e -> row_of_error ~model (Err.Error.kind e)

let model_dirs models_dir =
  Sys.readdir models_dir |> Array.to_list
  |> List.filter (fun name ->
      Sys.is_directory (Filename.concat models_dir name))
  |> List.sort String.compare

let () =
  match Err_host.install_from_env () with
  | Ok (_ : Err.Config.t option) -> ()
  | Error e ->
      Fmt.epr "pt2_json_model_support: %a@." Err_host.pp_error
        (Err.Error.kind e);
      exit 124

let () =
  match Sys.argv with
  | [| _; models_dir; output |] ->
      let rows = List.map (row_of_model models_dir) (model_dirs models_dir) in
      Out_channel.with_open_bin output (fun oc ->
          List.iter
            (fun row ->
              match
                Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json
                  (row_json row)
              with
              | Ok line ->
                  Out_channel.output_string oc line;
                  Out_channel.output_char oc '\n'
              | Error msg -> failwith msg)
            rows)
  | _ ->
      prerr_endline "usage: pt2_json_model_support MODELS_DIR OUTPUT.jsonl";
      exit 2
