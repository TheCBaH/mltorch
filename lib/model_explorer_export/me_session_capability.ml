(* Capability (what a session row says about one stage/feature) and View
   (a graph/flow/compare tab), which depends on Capability.graph_stage.
   Split out of me_session.ml. *)

module Capability = struct
  type graph_stage =
    | Canonical
    | Fusion
    | Initial_native
    | Kernel
    | Native4d
    | Source
    | Stage_program

  type feature =
    | Codegen
    | Expression_detail
    | Flow
    | Fold
    | Loop_ir
    | Pass_audits
    | Verification

  type key = Feature of feature | Graph_stage of graph_stage

  let stage_name = function
    | Canonical -> "canonical"
    | Fusion -> "fusion"
    | Initial_native -> "initial_native"
    | Kernel -> "kernel"
    | Native4d -> "native4d"
    | Source -> "source"
    | Stage_program -> "stage_program"

  let feature_name = function
    | Codegen -> "codegen"
    | Expression_detail -> "expression_detail"
    | Flow -> "flow"
    | Fold -> "fold"
    | Loop_ir -> "loop_ir"
    | Pass_audits -> "pass_audits"
    | Verification -> "verification"

  let key_name = function
    | Feature f -> "feature:" ^ feature_name f
    | Graph_stage s -> "stage:" ^ stage_name s

  (* These successor relations define the stage and feature timelines used by
     the public [all_*] lists; their declarations remain alphabetical. *)
  let next_stage = function
    | Source -> Some Initial_native
    | Initial_native -> Some Canonical
    | Canonical -> Some Native4d
    | Native4d -> Some Stage_program
    | Stage_program -> Some Kernel
    | Kernel -> Some Fusion
    | Fusion -> None

  let next_feature = function
    | Flow -> Some Verification
    | Verification -> Some Pass_audits
    | Pass_audits -> Some Fold
    | Fold -> Some Expression_detail
    | Expression_detail -> Some Loop_ir
    | Loop_ir -> Some Codegen
    | Codegen -> None

  let chain next first =
    let rec walk acc c =
      match next c with
      | None -> List.rev (c :: acc)
      | Some n -> walk (c :: acc) n
    in
    walk [] first

  let all_stages = chain next_stage Source
  let all_features = chain next_feature Flow

  let all_keys =
    List.map (fun s -> Graph_stage s) all_stages
    @ List.map (fun f -> Feature f) all_features

  module Pass_audit_status = struct
    type t = {
      retained_reports : int64;
      omitted_reports : int64;
      omitted_counts : Pass.Outcome_counts.t;
    }

    let jsont =
      Jsont.Object.map ~kind:"pass_audit_status"
        (fun retained_reports omitted_reports omitted_counts ->
          { retained_reports; omitted_reports; omitted_counts })
      |> Jsont.Object.mem "retainedReports" Jsont.int64_as_string ~enc:(fun t ->
          t.retained_reports)
      |> Jsont.Object.mem "omittedReports" Jsont.int64_as_string ~enc:(fun t ->
          t.omitted_reports)
      (* Through [Pass.Outcome_counts.jsont] in BOTH directions, so a malformed
         binding cannot become an unchecked map on the way in. It surfaces as a
         Jsont decode message rather than the typed [`Invalid_counts], because a
         [Jsont.t] has no typed error channel. *)
      |> Jsont.Object.mem "omittedCounts" Pass.Outcome_counts.jsont
           ~enc:(fun t -> t.omitted_counts)
      |> Jsont.Object.finish
  end

  type payload =
    | Graph of string
    | Pass_audit_status of Pass_audit_status.t
    | Present
    | Verification_summary of Pass.Outcome_counts.t

  type reason =
    | Not_implemented
    | Outside_dialect_domain
    | Over_limit
    | Prerequisite_unavailable
    | Requires_payloads
    | Unsupported_graph_shape
    | Unsupported_input
    | Unsupported_operator

  type status =
    | Available of payload
    | Not_requested
    | Unavailable of { reason : reason; detail : string option }

  type t = { key : key; status : status }

  (* The table IS the specification, so it is one exhaustive match rather than
     a set of guards: a key added to [key] stops this compiling. *)
  let compatible c =
    match (c.key, c.status) with
    | Feature Codegen, Unavailable { reason = Not_implemented; _ }
    | Feature Loop_ir, Unavailable { reason = Not_implemented; _ } ->
        true
    (* Never available and never merely unrequested: there is nothing to
       request. *)
    | Feature Codegen, _ | Feature Loop_ir, _ -> false
    | _, Not_requested -> true
    | _, Unavailable _ -> true
    | Graph_stage _, Available (Graph _) -> true
    | Feature Flow, Available (Graph _) -> true
    | Feature Verification, Available (Verification_summary _) -> true
    | Feature Pass_audits, Available (Pass_audit_status _) -> true
    | Feature Fold, Available Present -> true
    | Feature Expression_detail, Available Present -> true
    | _, Available _ -> false

  let units =
    "Verification_summary counts verifier clusters in the composed report; \
     Pass_audit_status counts audit reports in retained_reports and \
     omitted_reports, and clusters in omitted_counts."

  (* --- wire --- *)

  let key_jsont =
    Jsont.enum ~kind:"capability_key"
      (List.map (fun k -> (key_name k, k)) all_keys)

  let reason_jsont =
    Jsont.enum ~kind:"capability_reason"
      [
        ("not_implemented", Not_implemented);
        ("outside_dialect_domain", Outside_dialect_domain);
        ("over_limit", Over_limit);
        ("prerequisite_unavailable", Prerequisite_unavailable);
        ("requires_payloads", Requires_payloads);
        ("unsupported_graph_shape", Unsupported_graph_shape);
        ("unsupported_input", Unsupported_input);
        ("unsupported_operator", Unsupported_operator);
      ]

  (* A tagged union rather than four optional members, so a payload that names
     no kind is not representable on the wire either. *)
  let payload_jsont =
    let kind_of = function
      | Graph _ -> "graph"
      | Pass_audit_status _ -> "pass_audit_status"
      | Present -> "present"
      | Verification_summary _ -> "verification_summary"
    in
    Jsont.Object.map ~kind:"capability_payload"
      (fun kind graph verification pass_audits ->
        match (kind, graph, verification, pass_audits) with
        | "pass_audit_status", None, None, Some p -> Pass_audit_status p
        | "graph", Some g, None, None -> Graph g
        | "present", None, None, None -> Present
        | "verification_summary", None, Some v, None -> Verification_summary v
        | _ ->
            Jsont.Error.msgf Jsont.Meta.none
              "capability payload %S does not carry its own field" kind)
    |> Jsont.Object.mem "kind" Jsont.string ~enc:kind_of
    |> Jsont.Object.opt_mem "graph" Jsont.string ~enc:(function
      | Graph g -> Some g
      | _ -> None)
    |> Jsont.Object.opt_mem "verificationSummary" Pass.Outcome_counts.jsont
         ~enc:(function
         | Verification_summary v -> Some v
         | _ -> None)
    |> Jsont.Object.opt_mem "passAuditStatus" Pass_audit_status.jsont
         ~enc:(function
         | Pass_audit_status p -> Some p
         | _ -> None)
    |> Jsont.Object.finish

  let status_jsont =
    Jsont.Object.map ~kind:"capability_status"
      (fun state payload reason detail ->
        match (state, payload, reason) with
        | "available", Some p, None -> Available p
        | "not_requested", None, None -> Not_requested
        | "unavailable", None, Some reason -> Unavailable { reason; detail }
        | _ ->
            Jsont.Error.msgf Jsont.Meta.none
              "capability status %S does not carry its own field" state)
    |> Jsont.Object.mem "state" Jsont.string ~enc:(function
      | Available _ -> "available"
      | Not_requested -> "not_requested"
      | Unavailable _ -> "unavailable")
    |> Jsont.Object.opt_mem "payload" payload_jsont ~enc:(function
      | Available p -> Some p
      | _ -> None)
    |> Jsont.Object.opt_mem "reason" reason_jsont ~enc:(function
      | Unavailable { reason; _ } -> Some reason
      | _ -> None)
    |> Jsont.Object.opt_mem "detail" Jsont.string ~enc:(function
      | Unavailable { detail; _ } -> detail
      | _ -> None)
    |> Jsont.Object.finish

  let jsont =
    Jsont.Object.map ~kind:"capability" (fun key status -> { key; status })
    |> Jsont.Object.mem "key" key_jsont ~enc:(fun c -> c.key)
    |> Jsont.Object.mem "status" status_jsont ~enc:(fun c -> c.status)
    |> Jsont.Object.finish
end

module View = struct
  type kind = Compare | Flow | Stage of Capability.graph_stage

  type t = {
    id : string;
    label : string;
    kind : kind;
    collection : string;
    graph : string;
  }

  let kind_jsont =
    Jsont.enum ~kind:"view_kind"
      (List.map
         (fun s -> ("stage:" ^ Capability.stage_name s, Stage s))
         Capability.all_stages
      @ [ ("compare", Compare); ("flow", Flow) ])

  let jsont =
    Jsont.Object.map ~kind:"view" (fun id label kind collection graph ->
        { id; label; kind; collection; graph })
    |> Jsont.Object.mem "id" Jsont.string ~enc:(fun v -> v.id)
    |> Jsont.Object.mem "label" Jsont.string ~enc:(fun v -> v.label)
    |> Jsont.Object.mem "kind" kind_jsont ~enc:(fun v -> v.kind)
    |> Jsont.Object.mem "collection" Jsont.string ~enc:(fun v -> v.collection)
    |> Jsont.Object.mem "graph" Jsont.string ~enc:(fun v -> v.graph)
    |> Jsont.Object.finish
end
