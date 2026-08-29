(* Capability (what a session row says about one stage/feature) and View
   (a graph/flow/compare tab), which depends on Capability.graph_stage.
   Split out of me_session.ml. *)

module Capability = struct
  type graph_stage =
    | Source
    | Initial_native
    | Canonical
    | Native4d
    | Stage_program
    | Kernel
    | Fusion

  type feature =
    | Flow
    | Verification
    | Pass_audits
    | Fold
    | Expression_detail
    | Loop_ir
    | Codegen

  type key = Graph_stage of graph_stage | Feature of feature

  let stage_name = function
    | Source -> "source"
    | Initial_native -> "initial_native"
    | Canonical -> "canonical"
    | Native4d -> "native4d"
    | Stage_program -> "stage_program"
    | Kernel -> "kernel"
    | Fusion -> "fusion"

  let feature_name = function
    | Flow -> "flow"
    | Verification -> "verification"
    | Pass_audits -> "pass_audits"
    | Fold -> "fold"
    | Expression_detail -> "expression_detail"
    | Loop_ir -> "loop_ir"
    | Codegen -> "codegen"

  let key_name = function
    | Graph_stage s -> "stage:" ^ stage_name s
    | Feature f -> "feature:" ^ feature_name f

  (* Successor chains again, for the reason [Diagnostic.Code] has one: a key
     added to the type and not to [all_keys] is a key [validate] would never
     miss, so completeness would silently stop meaning what it says. *)
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
    | Verification_summary of Pass.Outcome_counts.t
    | Pass_audit_status of Pass_audit_status.t
    | Present

  type reason =
    | Unsupported_operator
    | Unsupported_input
    | Unsupported_graph_shape
    | Outside_dialect_domain
    | Over_limit
    | Requires_payloads
    | Prerequisite_unavailable
    | Not_implemented

  type status =
    | Available of payload
    | Unavailable of { reason : reason; detail : string option }
    | Not_requested

  type t = { key : key; status : status }

  (* The table IS the specification, so it is one exhaustive match rather than
     a set of guards: a key added to [key] stops this compiling. *)
  let compatible c =
    match (c.key, c.status) with
    | Feature Loop_ir, Unavailable { reason = Not_implemented; _ }
    | Feature Codegen, Unavailable { reason = Not_implemented; _ } ->
        true
    (* Never available and never merely unrequested: there is nothing to
       request. *)
    | Feature Loop_ir, _ | Feature Codegen, _ -> false
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
        ("unsupported_operator", Unsupported_operator);
        ("unsupported_input", Unsupported_input);
        ("unsupported_graph_shape", Unsupported_graph_shape);
        ("outside_dialect_domain", Outside_dialect_domain);
        ("over_limit", Over_limit);
        ("requires_payloads", Requires_payloads);
        ("prerequisite_unavailable", Prerequisite_unavailable);
        ("not_implemented", Not_implemented);
      ]

  (* A tagged union rather than four optional members, so a payload that names
     no kind is not representable on the wire either. *)
  let payload_jsont =
    let kind_of = function
      | Graph _ -> "graph"
      | Verification_summary _ -> "verification_summary"
      | Pass_audit_status _ -> "pass_audit_status"
      | Present -> "present"
    in
    Jsont.Object.map ~kind:"capability_payload"
      (fun kind graph verification pass_audits ->
        match (kind, graph, verification, pass_audits) with
        | "graph", Some g, None, None -> Graph g
        | "verification_summary", None, Some v, None -> Verification_summary v
        | "pass_audit_status", None, None, Some p -> Pass_audit_status p
        | "present", None, None, None -> Present
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
        | "unavailable", None, Some reason -> Unavailable { reason; detail }
        | "not_requested", None, None -> Not_requested
        | _ ->
            Jsont.Error.msgf Jsont.Meta.none
              "capability status %S does not carry its own field" state)
    |> Jsont.Object.mem "state" Jsont.string ~enc:(function
      | Available _ -> "available"
      | Unavailable _ -> "unavailable"
      | Not_requested -> "not_requested")
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
  type kind = Stage of Capability.graph_stage | Flow | Compare

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
      @ [ ("flow", Flow); ("compare", Compare) ])

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
