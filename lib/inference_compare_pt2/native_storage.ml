(* The Native direct evaluator under the storage and scheduling configurations
   inference runs with, as an ordinary graph runner over a coordinator-supplied
   body. A body is planned when it is prepared: the schedule, the lifetimes, the
   pools and the admission against a budget are decided there, so a body that
   cannot be taken is refused before any unit runs. Running only allocates and
   executes what was planned, on the body's own inputs. *)

module Tid = Graph_ir.Tensor_id
module Retain = Release_schedule.Retain

(* What the prepared bodies came to, for a run's report. *)
module Notes = struct
  type t = {
    mutable arena_declined : string list;
    mutable arena_reused : int;
    mutable arena_used : int;
    mutable bodies : int;
    mutable release_only : int;
    mutable role_planned : int;
  }

  let create () =
    {
      arena_declined = [];
      arena_reused = 0;
      arena_used = 0;
      bodies = 0;
      release_only = 0;
      role_planned = 0;
    }

  let summary t =
    let declined =
      match t.arena_declined with
      | [] -> ""
      | reasons ->
          Printf.sprintf "; %d declined (%s)" (List.length reasons)
            (List.hd (List.rev reasons))
    in
    Printf.sprintf
      "%d body(ies): %d in an arena (%d reusing storage), %d role-planned, %d \
       release-only%s"
      t.bodies t.arena_used t.arena_reused t.role_planned t.release_only
      declined
end

type compiled = {
  graph : Graph_ir.graph;  (** The body, reordered when scheduled. *)
  mode : mode;
}

and mode =
  | Arena_plan of Arena_plan.t * Arena.Admission.t
  | Release_only
  | Role_plan of Storage_plan.t

let retain = Retain.Only Tid.Set.empty
let ( let* ) = Result.bind

let text pp r =
  Result.map_error (fun e -> Format.asprintf "%a" pp (Err.Error.kind e)) r

let arena_text r = text Arena_run.pp_error (Result.map_error (fun e -> e) r)

let schedule_config ~mode =
  {
    Arena_schedule.Config.mode;
    retain;
    alignment = Alignment_policy.standard;
    limits = Arena_schedule.Limits.constructive_only;
  }

let admission_of = function
  | Storage_config.Arena_required bytes ->
      let* size =
        Core.Storage_units.Byte_size.of_int64 bytes
        |> Result.map_error (fun _ -> "invalid arena budget")
      in
      Ok (Arena.Admission.Required size)
  | _ -> Ok Arena.Admission.Best_effort

(* The slots of a plan hold more than its pools when storage is reused. *)
let reuses plan =
  let slots =
    List.fold_left
      (fun acc (s : Arena_plan.Slot.t) ->
        Int64.add acc (Core.Storage_units.Byte_size.to_int64 s.bytes))
      0L (Arena_plan.slots plan)
  in
  Int64.compare slots
    (Core.Storage_units.Byte_size.to_int64
       (Arena_plan.stats plan).Arena_plan.Stats.pool_bytes)
  > 0

let plan_arena ~(notes : Notes.t) ~(config : Storage_config.t) g =
  let* admission = admission_of config.storage in
  let declined reason graph =
    notes.arena_declined <- reason :: notes.arena_declined;
    Ok { graph; mode = Release_only }
  in
  let planned graph plan =
    match Arena_run.check_plan ~admission plan with
    | Ok () ->
        if reuses plan then notes.arena_reused <- notes.arena_reused + 1;
        Ok { graph; mode = Arena_plan (plan, admission) }
    | Error e -> (
        match admission with
        | Arena.Admission.Best_effort ->
            declined
              (Format.asprintf "%a" Arena_run.pp_error (Err.Error.kind e))
              graph
        | Required _ ->
            Error (Format.asprintf "%a" Arena_run.pp_error (Err.Error.kind e)))
  in
  if config.scheduled then
    let* selection =
      Arena_schedule_plan.choose ~admission
        (schedule_config ~mode:Arena_schedule.Mode.Intermediate)
        g
      |> text Arena_schedule_plan.pp_error
    in
    match selection.plan with
    | Some (Arena_schedule_plan.Plan.Intermediate plan) ->
        planned selection.graph plan
    | Some (Roles _) -> Error "an intermediate request planned roles"
    | None ->
        declined
          (match selection.declined with
          | Some e -> Format.asprintf "%a" Arena_schedule_plan.pp_error e
          | None -> "declined")
          selection.graph
  else
    let planned_plan =
      let* script = Eval_direct.dry_run ~retain g |> arena_text in
      Arena_plan.create script |> arena_text
    in
    match (planned_plan, admission) with
    | Ok plan, _ -> planned g plan
    | Error m, Arena.Admission.Best_effort -> declined m g
    | (Error _ as e), Required _ -> e

let plan_release ~(config : Storage_config.t) g =
  if config.scheduled then
    let* graph, _ =
      Arena_schedule_plan.release_only
        (schedule_config ~mode:Arena_schedule.Mode.Intermediate)
        g
      |> text Arena_schedule_plan.pp_error
    in
    Ok { graph; mode = Release_only }
  else Ok { graph = g; mode = Release_only }

let plan_roles ~(config : Storage_config.t) layout g =
  let roles =
    {
      Storage_script.Config.layout;
      constants = Storage_script.Ownership.Copied;
      inputs = Storage_script.Ownership.Copied;
    }
  in
  if config.scheduled then
    let* selection =
      Arena_schedule_plan.choose
        (schedule_config ~mode:(Arena_schedule.Mode.Roles roles))
        g
      |> text Arena_schedule_plan.pp_error
    in
    match selection.plan with
    | Some (Arena_schedule_plan.Plan.Roles plan) ->
        Ok { graph = selection.graph; mode = Role_plan plan }
    | Some (Intermediate _) | None -> Error "a role request always plans roles"
  else
    let* script = Eval_direct.storage_script ~retain roles g |> arena_text in
    let* plan = Storage_plan.create script |> arena_text in
    Ok { graph = g; mode = Role_plan plan }

let compile ~(notes : Notes.t) ~(config : Storage_config.t) g =
  notes.Notes.bodies <- notes.bodies + 1;
  let* compiled =
    match config.storage with
    | Storage_config.Default -> Ok { graph = g; mode = Release_only }
    | Release -> plan_release ~config g
    | Arena_best_effort | Arena_required _ -> plan_arena ~notes ~config g
    | Separate -> plan_roles ~config Storage_script.Layout.Separate g
    | Shared -> plan_roles ~config Storage_script.Layout.Shared_execution g
  in
  (match compiled.mode with
  | Release_only -> notes.release_only <- notes.release_only + 1
  | Role_plan _ -> notes.role_planned <- notes.role_planned + 1
  | Arena_plan _ -> ());
  Ok compiled

let eval = function
  | Error e ->
      Error
        (Format.asprintf "Native: %a" Eval_direct.pp_error (Err.Error.kind e))
  | Ok env -> Ok env

let execute ~(notes : Notes.t) c ~inputs =
  match c.mode with
  | Release_only -> eval (Eval_direct.run ~retain c.graph ~inputs)
  | Arena_plan (plan, admission) -> (
      match Arena_run.acquire_plan ~admission plan |> arena_text with
      | Error m -> Error m
      | Ok (Arena_run.Release_only reason) ->
          notes.Notes.arena_declined <-
            Format.asprintf "%a" Arena_run.pp_error reason
            :: notes.arena_declined;
          eval (Eval_direct.run ~retain c.graph ~inputs)
      | Ok (Arena a) ->
          notes.arena_used <- notes.arena_used + 1;
          eval (Eval_direct.run ~arena:a ~retain c.graph ~inputs))
  | Role_plan plan ->
      let* prepared =
        Constant_arena.create plan ~constants:[]
        |> Result.map_error (fun e ->
            Format.asprintf "%a" Arena_run.pp_error
              (Err.Error.kind e :> Arena_run.error))
      in
      let* runner = Storage_run.create plan prepared |> arena_text in
      let* lease =
        Storage_run.run runner ~retain c.graph ~inputs |> arena_text
      in
      let* env, _ =
        Result_lease.copy_out lease
        |> Result.map_error (fun e ->
            match Err.Error.kind e with
            | `Lease_released -> "the result lease was released"
            | `Quant_missing _ -> "a quantized result has no parameters"
            | #Tensor.dst_error -> "a result could not be copied out")
      in
      Ok env

(* The runner over Native bodies under [config], counting into [notes]. *)
let create ~dots ~config ~notes ~identity :
    Graph_ir.op Graph_body.t Graph_runner.t =
  let module E = struct
    type op = Graph_ir.op
    type nonrec compiled = compiled

    let compile g = compile ~notes ~config g

    let execute c _ ~inputs =
      let* env =
        Direct.with_dot_accumulation dots (fun () -> execute ~notes c ~inputs)
      in
      let rec outs acc = function
        | [] -> Ok (List.rev acc)
        | id :: rest -> (
            match Tid.Map.find_opt id env with
            | None -> Error "a declared output was not produced"
            | Some packed -> outs (packed :: acc) rest)
      in
      outs [] c.graph.outputs

    let release _ = ()
  end in
  let module R = Graph_runner.Make (E) in
  R.create ~identity
