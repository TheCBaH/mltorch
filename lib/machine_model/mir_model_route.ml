(* One invocation's generic program made executable on a route: the generic
   interpreter, or a target's selected program — as selected or scheduled — on
   its semantic interpreter, or that program reference-allocated, verified
   physically and checked symbolically, on the physical interpreter — or
   realized with its frames and published as an artifact, which re-verifies and
   re-checks it, and only then run. A checker rejection is a compiler error:
   the kernel is refused, never run. *)

open Machine_ir
open Machine_interp

module Stage = struct
  type t =
    | Allocated  (** reference allocation *)
    | Realized  (** reference allocation, frames, publication *)
    | Scanned
        (** sink scheduling, linear-scan allocation, frames, publication *)
    | Scheduled of Machine_alloc.Mir_schedule.Policy.t
    | Selected

  let name = function
    | Allocated -> "allocated"
    | Realized -> "realized"
    | Scanned -> "scanned"
    | Scheduled p -> "scheduled " ^ Machine_alloc.Mir_schedule.Policy.name p
    | Selected -> "selected"
end

module Route = struct
  type t = Aarch64 of Stage.t | Generic | X86_64 of Stage.t

  let name = function
    | Aarch64 s -> "aarch64 " ^ Stage.name s
    | Generic -> "generic"
    | X86_64 s -> "x86_64 " ^ Stage.name s
end

(* What a context needs of a kernel: its storage, and a run's status. *)
module Exec = struct
  type t = {
    instantiate :
      Mir_memory.t ->
      shared:(Mir_id.Region.t -> Mir_memory.Key.t option) ->
      (Mir_interp.Binding.t, string) result;
    run :
      ?fuel:int64 ->
      Mir_memory.t ->
      Mir_interp.Binding.t ->
      invocation:int32 ->
      Mir_observation.Status.t;
    artifact : (Format.formatter -> unit) option;
        (** the published artifact's summary, on the realized stage *)
    traffic : unit -> Mir_phys_interp.Traffic.t option;
        (** what this kernel's runs so far executed of the allocation's making,
            on an allocated stage *)
  }
end

let instantiate program memory ~shared =
  Mir_interp.instantiate ~shared program memory ~bound:(fun _ -> None)

let regions_program (p : (_, _) Mir_phys.Program.t) =
  {
    Mir_program.data_model = p.Mir_phys.Program.data_model;
    regions = p.Mir_phys.Program.regions;
    views = p.Mir_phys.Program.views;
    helpers = [];
    funcs = [];
    main = p.Mir_phys.Program.main;
    planning = None;
    revision = Mir_id.Revision.of_int 0;
  }

let models = Mir_math_model.all

(* Seconds spent in each back-end stage of one kernel, and what its published
   artifact holds; the clock is the caller's. *)
module Timing = struct
  type t = {
    select : float;
    schedule : float;  (** sink scheduling *)
    allocate : float;  (** reference allocation *)
    scan : float;  (** linear-scan allocation of the scheduled program *)
    realize : float;
    publish : float;  (** re-verification and re-checking included *)
    relocations : int;
    symbols : int;
  }
end

let generic (g : Mir_verify.Generic.t) =
  {
    Exec.instantiate = instantiate (Mir_verify.Generic.program g);
    run =
      (fun ?fuel memory binding ~invocation ->
        match
          (Mir_interp.run ?fuel ~invocation ~models g memory binding ~args:[])
            .Mir_interp.outcome
        with
        | Mir_interp.Outcome.Success _ -> Mir_observation.Status.Success
        | Mir_interp.Outcome.Failure row -> Mir_observation.Status.Failure row
        | Mir_interp.Outcome.Defect (d, _) -> Mir_observation.Status.Defect d
        | Mir_interp.Outcome.Fuel_exhausted ->
            Mir_observation.Status.Fuel_exhausted
        | Mir_interp.Outcome.Unsupported s ->
            Mir_observation.Status.Unsupported s);
    artifact = None;
    traffic = (fun () -> None);
  }

(* The selected and allocated routes of one target. *)
module Target
    (T : Mir_sel_interp.SEMANTICS)
    (R : Machine_alloc.Mir_linear_scan.POOL)
    (F : Machine_alloc.Mir_frame.FRAME with type op = T.op)
    (X : sig
      type result

      val select :
        sites:Mir_failure.Site_entry.t array ->
        Mir_verify.Generic.t ->
        (result, string) Stdlib.result

      val selected : result -> Mir_sel.Make(T).Verified.t
      val record : result -> Mir_id.View.t
    end) =
struct
  module I = Mir_sel_interp.Make (T)
  module A = Machine_alloc.Mir_ref_alloc.Make (T) (R)
  module Ls = Machine_alloc.Mir_linear_scan.Make (T) (R)
  module Sch = Machine_alloc.Mir_schedule.Make (T)
  module V = Mir_phys_verify.Make (T)
  module C = Machine_check.Mir_checker.Make (T)
  module P = Mir_phys_interp.Make (T)
  module Fr = Machine_alloc.Mir_frame.Make (T) (F)
  module Pub = Mir_artifact.Make (T)
  module Pr = Machine_alloc.Mir_pressure.Make (T)

  let ( let* ) = Result.bind

  let measure ~now ~sites g =
    let timed f =
      let t0 = now () in
      let r = f () in
      (r, now () -. t0)
    in
    let* planning =
      Option.to_result ~none:"a generic program with no planning summary"
        (Mir_verify.Generic.program g).Mir_program.planning
    in
    let res, select = timed (fun () -> X.select ~sites g) in
    let* res = res in
    let v = X.selected res in
    let s, schedule =
      timed (fun () -> Sch.schedule Machine_alloc.Mir_schedule.Policy.Sink v)
    in
    let* s =
      Result.map_error
        (Fmt.str "schedule: %a" Machine_alloc.Mir_schedule.Refusal.pp)
        s
    in
    let _, scan = timed (fun () -> Ls.allocate s) in
    let phys, allocate = timed (fun () -> A.allocate v) in
    let real, realize = timed (fun () -> Fr.realize phys) in
    let* real =
      Result.map_error
        (Fmt.str "frame: %a" Machine_alloc.Mir_frame.Refusal.pp)
        real
    in
    let artifact, publish = timed (fun () -> Pub.publish ~planning v real) in
    let* artifact = Result.map_error (fun e -> "publish: " ^ e) artifact in
    Ok
      {
        Timing.select;
        schedule;
        allocate;
        scan;
        realize;
        publish;
        relocations = List.length (Mir_artifact.relocations artifact);
        symbols = List.length (Mir_artifact.symbols artifact);
      }

  (* The production pipeline's pressure: sink scheduling, linear scan and
     frames. *)
  let pressure ~sites g =
    let* res = X.select ~sites g in
    let* s =
      Result.map_error
        (Fmt.str "schedule: %a" Machine_alloc.Mir_schedule.Refusal.pp)
        (Sch.schedule Machine_alloc.Mir_schedule.Policy.Sink (X.selected res))
    in
    let* real =
      Result.map_error
        (Fmt.str "frame: %a" Machine_alloc.Mir_frame.Refusal.pp)
        (Fr.realize (Ls.allocate s))
    in
    Ok (Pr.report s real)

  let exec stage ~sites g =
    let* res = X.select ~sites g in
    let v = X.selected res in
    let program = (I.S.Verified.selected v).I.S.program in
    let record =
      (Option.get (Mir_program.find_view program (X.record res)))
        .Mir_view.region
    in
    let status memory binding outcome =
      Mir_record.of_outcome memory binding ~sites ~record outcome
    in
    let scheduled policy =
      Result.map_error
        (Fmt.str "schedule: %a" Machine_alloc.Mir_schedule.Refusal.pp)
        (Sch.schedule policy v)
    in
    let selected v =
      {
        Exec.instantiate = instantiate program;
        run =
          (fun ?fuel memory binding ~invocation:_ ->
            status memory binding
              (I.run ?fuel ~models v memory binding ~args:[]).I.outcome);
        artifact = None;
        traffic = (fun () -> None);
      }
    in
    match stage with
    | Stage.Selected -> Ok (selected v)
    | Stage.Scheduled policy -> Result.map selected (scheduled policy)
    | Stage.Realized | Stage.Scanned -> (
        let* planning =
          Option.to_result ~none:"a generic program with no planning summary"
            (Mir_verify.Generic.program g).Mir_program.planning
        in
        let* v, phys =
          if stage = Stage.Scanned then
            Result.map
              (fun s -> (s, Ls.allocate s))
              (scheduled Machine_alloc.Mir_schedule.Policy.Sink)
          else Ok (v, A.allocate v)
        in
        let* real =
          Result.map_error
            (Fmt.str "frame: %a" Machine_alloc.Mir_frame.Refusal.pp)
            (Fr.realize phys)
        in
        match Pub.publish ~planning v real with
        | Error e -> Error ("publish: " ^ e)
        | Ok artifact ->
            let phys = Mir_artifact.program artifact in
            let traffic = ref Mir_phys_interp.Traffic.zero in
            Ok
              {
                Exec.instantiate = instantiate (regions_program phys);
                run =
                  (fun ?fuel memory binding ~invocation:_ ->
                    let r =
                      P.run ?fuel ~models ~realized:true phys memory binding
                        ~args:[]
                    in
                    traffic := Mir_phys_interp.Traffic.add !traffic r.P.traffic;
                    status memory binding r.P.outcome);
                artifact =
                  Some (fun fmt -> Mir_artifact.pp_summary fmt artifact);
                traffic = (fun () -> Some !traffic);
              })
    | Stage.Allocated -> (
        let phys = A.allocate v in
        match Err.payload (V.verify phys) with
        | Error d -> Error (Fmt.str "physical verifier: %a" Mir_diagnostic.pp d)
        | Ok phys -> (
            match Err.payload (C.check v phys) with
            | Error e ->
                Error
                  (Fmt.str "checker: %a" Machine_check.Mir_checker.pp_error e)
            | Ok () ->
                let traffic = ref Mir_phys_interp.Traffic.zero in
                Ok
                  {
                    Exec.instantiate = instantiate (regions_program phys);
                    run =
                      (fun ?fuel memory binding ~invocation:_ ->
                        let r =
                          P.run ?fuel ~models phys memory binding ~args:[]
                        in
                        traffic :=
                          Mir_phys_interp.Traffic.add !traffic r.P.traffic;
                        status memory binding r.P.outcome);
                    artifact = None;
                    traffic = (fun () -> Some !traffic);
                  }))
end

module A64_route =
  Target (Machine_target_aarch64.A64) (Machine_target_aarch64.A64_regs)
    (Machine_target_aarch64.A64_frame)
    (struct
      type result = Machine_target_aarch64.A64_select.result

      let select ~sites g =
        Result.map_error
          (Fmt.str "%a" Machine_target_aarch64.A64_select.Refusal.pp)
          (Err.payload
             (Machine_target_aarch64.A64_select.program ~sites
                ~unlisted:Mir_failure.Unlisted.Unreachable g))

      let selected r = r.Machine_target_aarch64.A64_select.selected
      let record r = r.Machine_target_aarch64.A64_select.record
    end)

module X64_route =
  Target (Machine_target_x86_64.X64) (Machine_target_x86_64.X64_regs)
    (Machine_target_x86_64.X64_frame)
    (struct
      type result = Machine_target_x86_64.X64_select.result

      let select ~sites g =
        Result.map_error
          (Fmt.str "%a" Machine_target_x86_64.X64_select.Refusal.pp)
          (Err.payload
             (Machine_target_x86_64.X64_select.program ~sites
                ~unlisted:Mir_failure.Unlisted.Unreachable g))

      let selected r = r.Machine_target_x86_64.X64_select.selected
      let record r = r.Machine_target_x86_64.X64_select.record
    end)

(* A target route's stage timings for one kernel; the generic route has no
   back end. *)
let measure route ~now ~sites g =
  match route with
  | Route.Aarch64 _ -> A64_route.measure ~now ~sites g
  | Route.Generic -> Error "the generic route has no back end to measure"
  | Route.X86_64 _ -> X64_route.measure ~now ~sites g

(* A target route's pressure for one kernel, whatever its stage. *)
let pressure route ~sites g =
  match route with
  | Route.Aarch64 _ -> A64_route.pressure ~sites g
  | Route.Generic -> Error "the generic route has no registers"
  | Route.X86_64 _ -> X64_route.pressure ~sites g

let exec route ~sites g =
  match route with
  | Route.Aarch64 stage -> A64_route.exec stage ~sites g
  | Route.Generic -> Ok (generic g)
  | Route.X86_64 stage -> X64_route.exec stage ~sites g
