open Inference_compare_schema
module W = Wire
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module H = Codecs.Payload_hasher
module Core = Inference_compare_core
module Plan = Inference_compare_plan

type error =
  | Duplicate_run of string
  | Io of File_io.error
  | Digest of string
  | Unportable_path of string

let pp_error ppf = function
  | Duplicate_run d -> Fmt.pf ppf "run directory %s already exists" d
  | Io e -> File_io.pp_error ppf e
  | Digest s -> Fmt.pf ppf "digest: %s" s
  | Unportable_path p ->
      Fmt.pf ppf
        "input path %s must be relative and stay inside the experiment \
         directory"
        p

type t = {
  dir : string;
  hashers : H.Registry.t;
  prepared : Session.prepared;
  mutable events : out_channel option;
  mutable event_budget : int option;
  mutable retained_bytes : int64;
  mutable entries : Retained.entry list;
  mutable omissions : Retained.omission list;
  mutable first_mismatch_done : bool;
  mutable finalized : bool;
  seen_payloads : (string, unit) Hashtbl.t;
}

let ( let* ) = Result.bind
let io r = Result.map_error (fun e -> Io e) r

let digest_hex hashers algorithm text =
  match H.digest_string hashers algorithm text with
  | Ok d -> Ok (Digest.Payload.hex d)
  | Error e ->
      Error (Digest (Format.asprintf "%a" H.pp_error (Err.Error.kind e)))

let payload_digest hashers algorithm text =
  let* hex = digest_hex hashers algorithm text in
  match Digest.Payload.make algorithm ~hex with
  | Ok d -> Ok d
  | Error _ -> Error (Digest "non-canonical digest")

let json_text j = Wire_text.to_string j ^ "\n"
let path t name = Filename.concat t.dir name

(* ---- manifest ------------------------------------------------------------ *)

let manifest_json hashers (p : Session.prepared) ~run_id =
  let l = p.validated.loaded in
  let alg = l.experiment.digest_algorithm in
  let pin name =
    let text = List.assoc name l.texts in
    let* d = payload_digest hashers alg text in
    Ok
      (W.obj
         [
           W.field "role" W.enc_string name;
           W.field "digest" Wire_digest.encode_payload d;
         ])
  in
  let rec pins acc = function
    | [] -> Ok (List.rev acc)
    | n :: rest ->
        let* j = pin n in
        pins (j :: acc) rest
  in
  let* inputs =
    pins []
      [ "experiment"; "left_graph"; "right_graph"; "mapping"; "selection" ]
  in
  let* case =
    let rec go acc = function
      | [] -> Ok (List.rev acc)
      | (k, bytes) :: rest ->
          let* d = payload_digest hashers alg bytes in
          go
            (W.obj
               [
                 W.field "key"
                   (fun k -> W.enc_string (Id.Canonical_value.to_string k))
                   k;
                 W.field "digest" Wire_digest.encode_payload d;
               ]
            :: acc)
            rest
    in
    go [] l.case_texts
  in
  let plan = Core.Sealed_plan.plan p.sealed in
  let engine side (spec : Engine_spec.t) (id : Engine_identity.t) =
    W.obj
      [
        W.field "side" Side.encode side;
        W.field "identity" Engine_identity.encode id;
        W.field "config" Engine_spec.encode spec;
      ]
  in
  Ok
    (W.obj
       [
         W.field "schema_version" W.enc_string "1";
         W.field "run_id" (fun r -> W.enc_string (Id.Run.to_string r)) run_id;
         W.field "case_id"
           (fun c -> W.enc_string (Id.Case.to_string c))
           l.experiment.case_id;
         W.field "experiment_id"
           (fun e -> W.enc_string (Id.Experiment.to_string e))
           l.experiment.experiment_id;
         W.field "evidence" W.enc_string "shared_frontier";
         W.field "digest_algorithm"
           (fun a -> W.enc_string (Id.Digest_algorithm.to_string a))
           alg;
         W.field "inputs" (W.enc_list Fun.id) inputs;
         W.field "left_graph" Graph_key.encode l.left_graph.key;
         W.field "right_graph" Graph_key.encode l.right_graph.key;
         W.field "case_payloads" (W.enc_list Fun.id) case;
         W.field "mapping_digest" Wire_digest.encode_mapping plan.mapping_digest;
         W.field "frontier_mapping_digest" Wire_digest.encode_frontier_mapping
           plan.frontier_mapping_digest;
         W.field "policy_selection_digest" Wire_digest.encode_policy_selection
           plan.policy_selection_digest;
         W.field "policy_bindings_digest" Wire_digest.encode_policy_bindings
           plan.policy_bindings_digest;
         W.field "plan_digest" Wire_digest.encode_plan plan.plan_digest;
         W.field "default_donor" Side.encode l.experiment.default_donor;
         W.field "overrides"
           (W.enc_list Donor_override.encode)
           l.experiment.overrides;
         W.field "limits" Resource_limits.encode l.experiment.limits;
         W.field "retain_units"
           (W.enc_list (fun u -> W.enc_string (Id.Unit.to_string u)))
           l.experiment.retain_units;
         W.field "engines" (W.enc_list Fun.id)
           [
             engine Left l.experiment.left.engine p.left_identity;
             engine Right l.experiment.right.engine p.right_identity;
           ];
       ])

(* ---- self-contained inputs ----------------------------------------------- *)

let safe_relative p =
  Filename.is_relative p && not (List.mem ".." (String.split_on_char '/' p))

(* Copies the experiment and every file it names under inputs/, at the same
   relative paths, so a run directory can be replayed on its own. The
   experiment text is kept byte for byte: its digest is a manifest pin. *)
let copy_inputs t =
  let l = t.prepared.validated.loaded in
  let e = l.experiment in
  let files =
    [
      ("experiment.json", List.assoc "experiment" l.texts);
      (e.left.graph, List.assoc "left_graph" l.texts);
      (e.right.graph, List.assoc "right_graph" l.texts);
      (e.mapping, List.assoc "mapping" l.texts);
      (e.selection, List.assoc "selection" l.texts);
    ]
    @ List.map
        (fun (c : Experiment_file.case_payload) ->
          (c.file, List.assoc_opt c.key l.case_texts |> Option.value ~default:""))
        e.case
  in
  match List.find_opt (fun (p, _) -> not (safe_relative p)) files with
  | Some (p, _) -> Error (Unportable_path p)
  | None ->
      let base = path t "inputs" in
      let rec go = function
        | [] -> Ok ()
        | (rel, text) :: rest ->
            let dest = Filename.concat base rel in
            let* () = io (File_io.mkdir_p (Filename.dirname dest)) in
            let* () = io (File_io.write_exclusive dest text) in
            go rest
      in
      go files

(* ---- creation ------------------------------------------------------------ *)

let create ~root ~hashers (prepared : Session.prepared) ~run_id =
  let dir = Filename.concat root ("run-" ^ Id.Run.to_string run_id) in
  let* () = io (File_io.mkdir_p root) in
  let* () =
    match File_io.mkdir_exclusive dir with
    | Ok () -> Ok ()
    | Error _ when Sys.file_exists dir -> Error (Duplicate_run dir)
    | Error e -> Error (Io e)
  in
  let t =
    {
      dir;
      hashers;
      prepared;
      events = None;
      event_budget = None;
      retained_bytes = 0L;
      entries = [];
      omissions = [];
      first_mismatch_done = false;
      finalized = false;
      seen_payloads = Hashtbl.create 16;
    }
  in
  let* manifest = manifest_json hashers prepared ~run_id in
  let* () =
    io (File_io.write_exclusive (path t "manifest.json") (json_text manifest))
  in
  let plan = Core.Sealed_plan.plan prepared.sealed in
  let* () =
    io
      (File_io.write_exclusive (path t "plan.json")
         (json_text (Execution_plan.encode plan)))
  in
  let* () =
    io
      (File_io.write_exclusive
         (path t "policy_bindings.json")
         (json_text
            (Policy_bindings.encode (Core.Sealed_plan.bindings prepared.sealed))))
  in
  let* () = io (File_io.mkdir_exclusive (path t "payloads")) in
  let* () = copy_inputs t in
  let* oc =
    io
      (File_io.guard (path t "events.jsonl") (fun () ->
           let fd =
             Unix.openfile (path t "events.jsonl")
               [ Unix.O_WRONLY; Unix.O_CREAT; Unix.O_EXCL ]
               0o644
           in
           Unix.out_channel_of_descr fd))
  in
  t.events <- Some oc;
  Ok t

let directory t = t.dir

(* ---- events -------------------------------------------------------------- *)

let emit t (e : Event.t) =
  match t.events with
  | None -> Error "event stream is closed"
  | Some oc -> (
      match t.event_budget with
      | Some 0 -> Error "event write failed (injected)"
      | b -> (
          t.event_budget <- Option.map pred b;
          try
            output_string oc (Wire_text.to_string (Event.encode e));
            output_char oc '\n';
            flush oc;
            Ok ()
          with Sys_error m -> Error m))

(* ---- retention ----------------------------------------------------------- *)

let alg t = t.prepared.validated.loaded.experiment.digest_algorithm
let limits t = t.prepared.validated.loaded.experiment.limits

let selected t (s : Core.Coordinator.snapshot) =
  let l = limits t in
  match s.stage with
  | After_failure -> l.retention <> No_retention
  | Before_commit -> (
      match l.retention with
      | No_retention -> false
      | All -> true
      | Selected_units ->
          List.exists (Id.Unit.equal s.unit)
            t.prepared.validated.loaded.experiment.retain_units
      | First_mismatch ->
          s.mismatching
          && (not t.first_mismatch_done)
          &&
          (t.first_mismatch_done <- true;
           true))

let store_payload t (v : L.t) =
  let bytes = L.contents v in
  let* d = payload_digest t.hashers (alg t) bytes in
  let key = Codecs.Payload_hasher.content_path d in
  if Hashtbl.mem t.seen_payloads key then Ok (d, 0L)
  else
    let size = Int64.of_int (String.length bytes) in
    let dir =
      path t
        (Filename.concat "payloads" (Id.Digest_algorithm.to_string (alg t)))
    in
    let* () = io (File_io.mkdir_p dir) in
    let* () =
      io
        (File_io.write_atomic
           (Filename.concat dir (Digest.Payload.hex d))
           bytes)
    in
    Hashtbl.replace t.seen_payloads key ();
    Ok (d, size)

let retain t (s : Core.Coordinator.snapshot) =
  if not (selected t s) then Ok ()
  else
    let l = limits t in
    let required = l.retention_requirement = Required in
    let items =
      List.map
        (fun (k, i, v) ->
          (Retained.Common_input, Id.Canonical_value.to_string k, None, i, v))
        s.common_inputs
      @ List.map
          (fun (side, tr, i, v) ->
            ( Retained.Raw_output,
              Id.Tensor_local.to_string tr.Tensor_ref.local_id,
              Some side,
              i,
              v ))
          s.raw
      @ List.map
          (fun (k, side, i, v) ->
            (Retained.Candidate, Id.Canonical_value.to_string k, Some side, i, v))
          s.candidates
    in
    let rec go = function
      | [] -> Ok ()
      | (role, subject, side, instance, v) :: rest -> (
          let omit reason =
            t.omissions <-
              t.omissions
              @ [ { Retained.unit = s.unit; role; subject; reason } ]
          in
          let bytes = Int64.of_int (String.length (L.contents v)) in
          let fits =
            Int64.compare bytes
              (Int64.sub l.max_retained_disk_bytes t.retained_bytes)
            <= 0
            ||
            match digest_hex t.hashers (alg t) (L.contents v) with
            | Ok hex -> (
                match Digest.Payload.make (alg t) ~hex with
                | Ok d ->
                    Hashtbl.mem t.seen_payloads
                      (Codecs.Payload_hasher.content_path d)
                | Error _ -> false)
            | Error _ -> false
          in
          if not fits then
            if required then Error "retention budget exhausted"
            else (
              omit Retained.Budget_exhausted;
              go rest)
          else
            match store_payload t v with
            | Error e ->
                if required then Error (Format.asprintf "%a" pp_error e)
                else (
                  omit Retained.Write_failed;
                  go rest)
            | Ok (d, added) ->
                t.retained_bytes <- Int64.add t.retained_bytes added;
                t.entries <-
                  t.entries
                  @ [
                      {
                        Retained.instance;
                        unit = s.unit;
                        role;
                        subject;
                        side;
                        spec = L.spec v;
                        payload_digest = d;
                      };
                    ];
                go rest)
    in
    go items

(* ---- finalization -------------------------------------------------------- *)

let verdict_name : Comparison_result.verdict -> string = function
  | Pass -> "pass"
  | Mismatch -> "mismatch"
  | Metadata_error -> "metadata_error"
  | Metric_unavailable -> "metric_unavailable"
  | Unsupported -> "unsupported"

let summary_json (r : Core.Coordinator.result) =
  W.obj
    [
      W.field "completion" Completion.encode r.completion;
      W.field "failure" (W.enc_option Failure.encode) r.failure;
      W.field "executed_units"
        (W.enc_list (fun u -> W.enc_string (Id.Unit.to_string u)))
        r.executed;
      W.field "comparisons"
        (W.enc_list (fun (c : Comparison_result.t) ->
             W.obj
               [
                 W.field "comparison"
                   (fun c -> W.enc_string (Id.Comparison.to_string c))
                   c.comparison_id;
                 W.field "verdict"
                   (fun v -> W.enc_string (verdict_name v))
                   c.verdict;
                 W.field "elements_checked" W.enc_int64 c.elements_checked;
                 W.field "elements_expected" W.enc_int64 c.elements_expected;
               ]))
        r.comparisons;
    ]

let close_events t =
  match t.events with
  | Some oc ->
      close_out_noerr oc;
      t.events <- None
  | None -> ()

let abandon t = close_events t

let finalize t (r : Core.Coordinator.result) =
  if t.finalized then Error (Digest "already finalized")
  else (
    t.finalized <- true;
    close_events t;
    let index = { Retained.entries = t.entries; omissions = t.omissions } in
    let* () =
      io
        (File_io.write_exclusive (path t "retained.json")
           (json_text (Retained.encode_index index)))
    in
    let* () =
      io
        (File_io.write_exclusive (path t "summary.json")
           (json_text (summary_json r)))
    in
    let files =
      [
        "manifest.json";
        "plan.json";
        "policy_bindings.json";
        "events.jsonl";
        "retained.json";
        "summary.json";
      ]
    in
    let* pins =
      let rec go acc = function
        | [] -> Ok (List.rev acc)
        | f :: rest ->
            let* text = io (File_io.read (path t f)) in
            let* d = payload_digest t.hashers (alg t) text in
            go
              (W.obj
                 [
                   W.field "name" W.enc_string f;
                   W.field "digest" Wire_digest.encode_payload d;
                 ]
              :: acc)
              rest
      in
      go [] files
    in
    let completion =
      W.obj
        [
          W.field "schema_version" W.enc_string "1";
          W.field "files" (W.enc_list Fun.id) pins;
          W.field "completion" Completion.encode r.completion;
        ]
    in
    io (File_io.write_atomic (path t "completion.json") (json_text completion)))

module Fault = struct
  let fail_events_after t n = t.event_budget <- Some n
end
