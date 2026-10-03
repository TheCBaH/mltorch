(* The arena problems a model.json poses, payload-free, for each normalized
   dialect: decode, Native import, canonical normalization (no constants), then
   one dry run per dialect over that dialect's own graph. Native4D is converted
   from the same canonical Native graph and measured on its converted graph:
   legalization adds, removes and reshapes intermediates.

   A refusal is classified by the same total classifiers the model-support
   report uses ([Me_classify]), so "outside the dialect" and "a defect" mean the
   same here as there. Nothing here runs a tensor operation. *)

module Dialect = struct
  type t = Native | Native4d

  let all = [ Native; Native4d ]
  let name = function Native -> "native" | Native4d -> "native4d"

  let of_name = function
    | "native" -> Some Native
    | "native4d" -> Some Native4d
    | _ -> None
end

(* Where a branch stopped. *)
module Stage = struct
  type t =
    | Dry_run
    | Evaluate
    | Internal
    | Native4d_convert
    | Native_import
    | Normalize
    | Problem
    | Source

  let name = function
    | Dry_run -> "dry_run"
    | Evaluate -> "evaluate"
    | Internal -> "internal"
    | Native4d_convert -> "native4d_convert"
    | Native_import -> "native_import"
    | Normalize -> "normalize"
    | Problem -> "problem"
    | Source -> "source"
end

(* A branch that produced no problem, and why. [Refused]: a partiality the
   dialect declares, with the classifier's reason. [Failed]: a defect.
   [Prerequisite]: the shared import failed, so this branch never started. *)
module Outcome = struct
  type t =
    | Failed of { stage : Stage.t; diagnostic : string }
    | Prerequisite of { stage : Stage.t; diagnostic : string }
    | Refused of { stage : Stage.t; reason : string; diagnostic : string }
end

module Extracted = struct
  type t = {
    problem : Arena_problem.t;
    nodes : int;
    events : int;
    script_digest : string;
    convert_s : float;
    dry_run_s : float;
  }
end

let reason_name : Me_session.Capability.reason -> string = function
  | Not_implemented -> "not_implemented"
  | Outside_dialect_domain -> "outside_dialect_domain"
  | Over_limit -> "over_limit"
  | Prerequisite_unavailable -> "prerequisite_unavailable"
  | Requires_payloads -> "requires_payloads"
  | Unsupported_dtype -> "unsupported_dtype"
  | Unsupported_graph_shape -> "unsupported_graph_shape"
  | Unsupported_input -> "unsupported_input"
  | Unsupported_operator -> "unsupported_operator"

let script_digest script =
  Digest.to_hex (Digest.string (Fmt.str "%a" Alloc_script.pp script))

let only_empty = Release_schedule.Retain.Only Graph_ir.Tensor_id.Set.empty

let timed now f =
  let start = now () in
  let r = f () in
  (r, now () -. start)

(* The canonical Native graph, or why there is none. *)
let canonical ~now bytes =
  let str pp e = Fmt.str "%a" pp e in
  let t0 = now () in
  match
    Jsont_bytesrw.decode_string Pytorch_types.ExportedProgram.jsont bytes
  with
  | Error msg ->
      Error (Outcome.Failed { stage = Stage.Source; diagnostic = msg }, 0.)
  | Ok program -> (
      match Native_interp.lower program with
      | Error e -> (
          let kind = Err.Error.kind e in
          let diagnostic = str Native_interp.pp_error kind in
          match Me_classify.lowering kind with
          | Me_classify.Unavailable reason ->
              Error
                ( Outcome.Refused
                    {
                      stage = Stage.Native_import;
                      reason = reason_name reason;
                      diagnostic;
                    },
                  now () -. t0 )
          | Me_classify.Fatal ->
              Error
                ( Outcome.Failed { stage = Stage.Native_import; diagnostic },
                  now () -. t0 ))
      | Ok lowered -> (
          match
            Native_interp.transform_lowered lowered
              ~passes:[ Pipeline.canonical ~fold:false ]
          with
          | Error e ->
              Error
                ( Outcome.Failed
                    {
                      stage = Stage.Normalize;
                      diagnostic = str Native_interp.pp_error (Err.Error.kind e);
                    },
                  now () -. t0 )
          | Ok transformed -> Ok (transformed, now () -. t0)))

let problem_of ~convert_s ~dry_run_s ~nodes script =
  match Arena_problem.of_script script with
  | Error e ->
      let (`Arena_script id) = Err.Error.kind e in
      Error
        (Outcome.Failed
           {
             stage = Stage.Problem;
             diagnostic =
               Fmt.str "inconsistent script at %a" Graph_ir.Tensor_id.pp id;
           })
  | Ok problem ->
      Ok
        {
          Extracted.problem;
          nodes;
          events = List.length script;
          script_digest = script_digest script;
          convert_s;
          dry_run_s;
        }

let native ~now (Native_interp.Transformed t) =
  let script, dry_run_s =
    timed now (fun () -> Eval_direct.dry_run ~retain:only_empty t.graph)
  in
  match script with
  | Error e ->
      Error
        (Outcome.Failed
           {
             stage = Stage.Dry_run;
             diagnostic = Fmt.str "%a" Eval_direct.pp_error (Err.Error.kind e);
           })
  | Ok script ->
      problem_of ~convert_s:0. ~dry_run_s
        ~nodes:(List.length t.graph.Graph_common.Graph.nodes)
        script

(* No constants: [convert] is given the symbolic store only, so a legalization
   that needed a payload would refuse rather than read one. *)
let native4d ~now (Native_interp.Transformed t) =
  let converted, convert_s =
    timed now (fun () ->
        match Snapshot.create t.graph with
        | Error e ->
            Error
              (Outcome.Failed
                 {
                   stage = Stage.Native4d_convert;
                   diagnostic =
                     Fmt.str "%a" Graph_view.pp_error (Err.Error.kind e);
                 })
        | Ok (Snapshot.Pack src) -> (
            match
              Native4d.Lower.convert ~constant_store:t.constant_store src
            with
            | Error e -> (
                let kind = Err.Error.kind e in
                let diagnostic = Fmt.str "%a" Native4d.Error.pp kind in
                match Me_classify.native4d kind with
                | Me_classify.Unavailable reason ->
                    Error
                      (Outcome.Refused
                         {
                           stage = Stage.Native4d_convert;
                           reason = reason_name reason;
                           diagnostic;
                         })
                | Me_classify.Fatal ->
                    Error
                      (Outcome.Failed
                         { stage = Stage.Native4d_convert; diagnostic }))
            | Ok (Native4d.Lower.Pack r) -> Ok (Native4d.Lower.graph r)))
  in
  match converted with
  | Error o -> Error o
  | Ok dst -> (
      let script, dry_run_s =
        timed now (fun () ->
            Native4d.Eval_direct4.dry_run ~retain:only_empty dst)
      in
      match script with
      | Error e ->
          Error
            (Outcome.Failed
               {
                 stage = Stage.Dry_run;
                 diagnostic =
                   Fmt.str "%a" Native4d.Eval_direct4.pp_error
                     (Err.Error.kind e);
               })
      | Ok script ->
          problem_of ~convert_s ~dry_run_s
            ~nodes:(List.length dst.Graph_common.Graph.nodes)
            script)

let dialect ~now d transformed =
  match d with
  | Dialect.Native -> native ~now transformed
  | Dialect.Native4d -> native4d ~now transformed

(* A shared refusal applies to every branch. Only an import defect blocks a
   branch with a failed prerequisite. *)
let prerequisite = function
  | Outcome.Failed { stage; diagnostic }
  | Outcome.Prerequisite { stage; diagnostic } ->
      Outcome.Prerequisite { stage; diagnostic }
  | Outcome.Refused _ as o -> o
