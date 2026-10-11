open Inference_compare_schema
module W = Wire
module Core = Inference_compare_core
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value

type error =
  | Evidence of Evidence_read.error
  | Load of Session.error
  | Identity of string
  | Missing_payload of { unit : Id.Unit.t; keys : string list; omitted : bool }
  | Unknown_unit of Id.Unit.t

let pp_error ppf = function
  | Evidence e -> Evidence_read.pp_error ppf e
  | Load e -> Session.pp_error ppf e
  | Identity s -> Fmt.pf ppf "incompatible identity: %s" s
  | Missing_payload { unit; keys; omitted } ->
      Fmt.pf ppf "unit %a: retained input(s) %s %s" Id.Unit.pp unit
        (String.concat ", " keys)
        (if omitted then "were omitted when the run was recorded"
         else "were never retained")
  | Unknown_unit u ->
      Fmt.pf ppf "unit %a is not in the recorded plan" Id.Unit.pp u

type report = {
  scope : [ `Unit of Id.Unit.t | `Full ];
  reexecuted : bool;
  identical : bool;
  differences : string list;
  notes : string list;
}

let ( let* ) = Result.bind
let hashers = Session.hashers

(* What an event contributes to a comparison between two runs: values and
   decisions, never instance ids or sequence numbers. *)
let signature (e : Event.t) =
  let u = match e.unit with Some u -> Id.Unit.to_string u | None -> "-" in
  let canon j = Wire_text.to_string j in
  match e.payload with
  | Inputs_verified vs ->
      Some
        (Printf.sprintf "%s inputs %s" u
           (String.concat ","
              (List.map
                 (fun (v : Verified_input.t) ->
                   Id.Canonical_value.to_string v.key
                   ^ "="
                   ^ Digest.Logical_value.to_string v.left_normalized_digest)
                 vs)))
  | Graph_completed { side; outputs } ->
      Some
        (Printf.sprintf "%s %s outputs %s" u (Side.to_string side)
           (String.concat ","
              (List.map
                 (fun (o : Computed_value.t) ->
                   canon (Tensor_spec.encode o.spec)
                   ^ "="
                   ^ Digest.Logical_value.to_string o.logical_digest)
                 outputs)))
  | Outputs_compared rs ->
      Some
        (Printf.sprintf "%s compared %s" u
           (String.concat ";"
              (List.map (fun r -> canon (Comparison_result.encode r)) rs)))
  | Frontier_committed vs ->
      Some
        (Printf.sprintf "%s committed %s" u
           (String.concat ","
              (List.map
                 (fun (c : Committed_value.t) ->
                   Id.Canonical_value.to_string c.key
                   ^ "<-" ^ Side.to_string c.donor)
                 vs)))
  | Failed f ->
      Some (Printf.sprintf "%s failed %s" u (canon (Failure.encode f)))

let signatures events = List.filter_map signature events

let diff_lists ~expected ~actual =
  let rec go i e a acc =
    match (e, a) with
    | [], [] -> List.rev acc
    | x :: er, y :: ar ->
        go (i + 1) er ar
          (if String.equal x y then acc
           else
             Printf.sprintf "event %d: recorded [%s] replayed [%s]" i x y :: acc)
    | x :: er, [] ->
        go (i + 1) er []
          (Printf.sprintf "event %d: recorded [%s] missing from the replay" i x
          :: acc)
    | [], y :: ar ->
        go (i + 1) [] ar
          (Printf.sprintf "event %d: replay produced extra [%s]" i y :: acc)
  in
  go 0 expected actual []

(* ---- reopening the recorded experiment ----------------------------------- *)

type opened = {
  evidence : Evidence_read.t;
  prepared : Session.prepared;
  events : Event.t list;
}

let engine_pins =
  W.list
    (W.record (fun o ->
         let ( let* ) = W.( let* ) in
         let* side = W.Obj.required o "side" Side.decode in
         let* identity = W.Obj.required o "identity" Engine_identity.decode in
         let* _config = W.Obj.required o "config" Engine_spec.decode in
         Ok (side, identity)))

let input_pins =
  W.list
    (W.record (fun o ->
         let ( let* ) = W.( let* ) in
         let* role = W.Obj.required o "role" W.string in
         let* digest = W.Obj.required o "digest" Wire_digest.payload in
         Ok (role, digest)))

let reopen run_dir =
  let* evidence =
    Result.map_error
      (fun e -> Evidence e)
      (Evidence_read.open_ ~hashers run_dir)
  in
  let manifest = evidence.manifest in
  let m name dec =
    Result.map_error
      (fun s -> Evidence (Evidence_read.Corrupt s))
      (Json_access.decode_member name dec manifest)
  in
  let* pins = m "inputs" input_pins in
  let* plan_digest = m "plan_digest" Wire_digest.plan in
  let* engines = m "engines" engine_pins in
  let* loaded =
    Result.map_error
      (fun e -> Load e)
      (Session.load (Filename.concat run_dir "inputs/experiment.json"))
  in
  (* the kept inputs are the ones the run was pinned to *)
  let* () =
    List.fold_left
      (fun acc (role, expected) ->
        let* () = acc in
        match List.assoc_opt role loaded.texts with
        | None -> Error (Identity ("manifest pins unknown input " ^ role))
        | Some text -> (
            match
              Codecs.Payload_hasher.digest_string hashers
                (Digest.Payload.algorithm expected)
                text
            with
            | Ok d when Digest.Payload.equal d expected -> Ok ()
            | _ -> Error (Identity (role ^ " differs from the recorded input"))))
      (Ok ()) pins
  in
  let* validated =
    Result.map_error (fun e -> Load e) (Session.validate loaded)
  in
  let* prepared =
    Result.map_error (fun e -> Load e) (Session.prepare validated)
  in
  let plan = Core.Sealed_plan.plan prepared.sealed in
  let* () =
    if Digest.Plan.equal plan.plan_digest plan_digest then Ok ()
    else
      Error
        (Identity
           "the plan recomputed from the kept inputs differs from the recorded \
            plan")
  in
  let* () =
    let stored = Filename.concat run_dir "plan.json" in
    let* text =
      Result.map_error
        (fun e -> Evidence (Evidence_read.Io e))
        (File_io.read stored)
    in
    match Wire_text.decode_string Execution_plan.decode text with
    | Error _ -> Error (Evidence (Evidence_read.Corrupt "plan.json"))
    | Ok decoded -> (
        match Core.Sealed_plan.verify_decoded prepared.sealed decoded with
        | Ok () -> Ok ()
        | Error _ ->
            Error (Identity "stored plan.json differs from the recomputed plan")
        )
  in
  let* () =
    List.fold_left
      (fun acc (side, recorded) ->
        let* () = acc in
        let now =
          match (side : Side.t) with
          | Left -> prepared.left_identity
          | Right -> prepared.right_identity
        in
        if Engine_identity.equal recorded now then Ok ()
        else
          Error
            (Identity
               (Printf.sprintf "%s engine identity changed"
                  (Side.to_string side))))
      (Ok ()) engines
  in
  let* events =
    Result.map_error (fun e -> Evidence e) (Evidence_read.events evidence)
  in
  Ok { evidence; prepared; events }

(* ---- execution ----------------------------------------------------------- *)

let execute (p : Session.prepared) ~mode =
  let events = ref [] in
  let cfg =
    Run.config p
      ~emit:(fun e ->
        events := e :: !events;
        Ok ())
      ~retain:(fun _ -> Ok ())
      ~mode ~ledger_limit:p.validated.loaded.experiment.limits.max_live_bytes
  in
  let result =
    Core.Coordinator.run cfg p.sealed ~left:p.left_endpoint
      ~right:p.right_endpoint
  in
  List.iter (fun (_, v) -> L.drop v) result.finals;
  (result, List.rev !events)

let unit_of_event (e : Event.t) = e.unit

let unit ~run_dir u =
  let* o = reopen run_dir in
  let plan =
    Inference_compare_plan.Correspondence_check.unit_plan
      (Inference_compare_plan.Structural.correspondence
         o.prepared.validated.structural)
      u
  in
  match plan with
  | None -> Error (Unknown_unit u)
  | Some plan ->
      let* index =
        Result.map_error
          (fun e -> Evidence e)
          (Evidence_read.retained o.evidence)
      in
      let wanted = List.map Id.Canonical_value.to_string plan.canonical_reads in
      let have =
        List.filter
          (fun (e : Retained.entry) ->
            Id.Unit.equal e.unit u && e.role = Common_input)
          index.entries
      in
      let missing =
        List.filter
          (fun k ->
            not (List.exists (fun (e : Retained.entry) -> e.subject = k) have))
          wanted
      in
      let* () =
        if missing = [] then Ok ()
        else
          Error
            (Missing_payload
               {
                 unit = u;
                 keys = missing;
                 omitted =
                   List.exists
                     (fun (m : Retained.omission) ->
                       Id.Unit.equal m.unit u && m.role = Common_input)
                     index.omissions;
               })
      in
      let ctx = o.prepared.validated.loaded.initial_ctx in
      let* inputs =
        List.fold_left
          (fun acc (e : Retained.entry) ->
            let* acc = acc in
            let* body =
              Result.map_error
                (fun e -> Evidence e)
                (Evidence_read.payload ~hashers o.evidence e.payload_digest)
            in
            match L.make ctx e.spec body with
            | Ok v ->
                Ok
                  (acc
                  @ [
                      ( Id.Canonical_value.of_string e.subject |> Result.get_ok,
                        v );
                    ])
            | Error _ ->
                Error
                  (Evidence
                     (Evidence_read.Corrupt
                        "retained payload does not fit its spec")))
          (Ok []) have
      in
      let _, events =
        execute o.prepared ~mode:(Single_unit { unit = u; inputs })
      in
      List.iter (fun (_, v) -> L.drop v) inputs;
      let recorded = List.filter (fun e -> unit_of_event e = Some u) o.events in
      let recorded =
        (* a recorded failure of this unit is part of what is compared *)
        recorded
      in
      let differences =
        diff_lists ~expected:(signatures recorded) ~actual:(signatures events)
      in
      Ok
        {
          scope = `Unit u;
          reexecuted = true;
          identical = differences = [];
          differences;
          notes = [ "re-executed from the retained common inputs of the unit" ];
        }

let full ~run_dir =
  let* o = reopen run_dir in
  let _, events = execute o.prepared ~mode:Full in
  let differences =
    diff_lists ~expected:(signatures o.events) ~actual:(signatures events)
  in
  Ok
    {
      scope = `Full;
      reexecuted = true;
      identical = differences = [];
      differences;
      notes =
        (if differences = [] then []
         else
           [
             "the replay differs from the record: the engines are not \
              reproducible, or the record is not what ran";
           ]);
    }

let differences ~recorded ~replayed =
  diff_lists ~expected:(signatures recorded) ~actual:(signatures replayed)
