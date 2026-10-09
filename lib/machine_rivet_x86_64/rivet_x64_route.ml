(* The emulated x86-64 route of a model bundle: each invocation's generic
   program is selected, allocated, framed and published, then lowered through
   typed Rivet modules into an image of its own, laid out once. A run copies the
   invocation's storage into the image's data block, executes it as a static
   process under qemu-user, and copies the results back. This is emulation, not
   an x86-64 CPU: it checks the instruction bytes and the control flow, and the
   cost of a run is dominated by the process, so it is for correctness only. *)

open Machine_ir
open Machine_interp
open Machine_target_x86_64
module A = Machine_alloc.Mir_ref_alloc.Make (X64) (X64_regs)
module Ls = Machine_alloc.Mir_linear_scan.Make (X64) (X64_regs)
module Sch = Machine_alloc.Mir_schedule.Make (X64)
module Fr = Machine_alloc.Mir_frame.Make (X64) (X64_frame)
module Pub = Machine_model.Mir_artifact.Make (X64)
module Art = Machine_model.Mir_artifact
module Route = Machine_model.Mir_model_route
module Qemu = Rivet_x64_qemu

let ( let* ) = Result.bind

(* Which allocator makes the physical program. *)
module Allocation = struct
  type t =
    | Reference  (** the stack-based reference allocator *)
    | Scanned  (** sink scheduling, then split linear scan *)

  let name = function Reference -> "reference" | Scanned -> "scanned"
end

type published = {
  artifact : (X64_op.t, X64_op.test) Art.t;
  record : Mir_id.Region.t;
}

(* Selection, allocation, frame realization and publication. *)
let publish ~allocation ~sites (g : Mir_verify.Generic.t) =
  let* planning =
    Option.to_result ~none:"a generic program with no planning summary"
      (Mir_verify.Generic.program g).Mir_program.planning
  in
  let* res =
    Result.map_error
      (Fmt.str "%a" X64_select.Refusal.pp)
      (Err.payload
         (X64_select.program ~sites ~unlisted:Mir_failure.Unlisted.Unreachable g))
  in
  let* v, phys =
    match allocation with
    | Allocation.Reference ->
        Ok (res.X64_select.selected, A.allocate res.X64_select.selected)
    | Allocation.Scanned ->
        let* s =
          Result.map_error
            (Fmt.str "schedule: %a" Machine_alloc.Mir_schedule.Refusal.pp)
            (Sch.schedule Machine_alloc.Mir_schedule.Policy.Sink
               res.X64_select.selected)
        in
        Ok (s, Ls.allocate s)
  in
  let* real =
    Result.map_error
      (Fmt.str "frame: %a" Machine_alloc.Mir_frame.Refusal.pp)
      (Fr.realize phys)
  in
  let* artifact =
    Result.map_error (( ^ ) "publish: ") (Pub.publish ~planning v real)
  in
  let sel = X64_stage.Sel.Verified.selected v in
  let* view =
    Option.to_result ~none:"the failure record names no view"
      (Mir_program.find_view sel.X64_stage.Sel.program res.X64_select.record)
  in
  Ok { artifact; record = view.Mir_view.region }

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

let exec_of ~allocation ~runtime ~sites (g : Mir_verify.Generic.t) =
  let* { artifact; record } = publish ~allocation ~sites g in
  let* prepared = Qemu.prepare ~runtime artifact in
  let phys = Art.program artifact in
  let regions = phys.Mir_phys.Program.regions in
  let bound_bytes memory binding (r : Mir_region.t) =
    match Mir_interp.Binding.instance binding r.Mir_region.id with
    | None -> None
    | Some key ->
        let n = Int64.to_int r.Mir_region.size in
        Some
          (String.init n (fun i ->
               match
                 (Mir_memory.read_bytes memory key ~offset:(Int64.of_int i) ~n:1).(
                 0)
               with
               | Some b -> Char.chr b
               | None -> '\000'))
  in
  let run ?fuel:_ memory binding ~invocation:_ =
    let bound id =
      match
        List.find_opt
          (fun (r : Mir_region.t) -> Mir_id.Region.equal r.Mir_region.id id)
          regions
      with
      | Some ({ Mir_region.init = Mir_region.Bound; _ } as r) ->
          bound_bytes memory binding r
      | _ -> None
    in
    match Qemu.launch prepared ~bound with
    | Error e -> Mir_observation.Status.Unsupported e
    | Ok { Qemu.status; regions = after; _ } ->
        List.iter
          (fun (id, bytes) ->
            let wanted =
              match
                List.find_opt
                  (fun (r : Mir_region.t) ->
                    Mir_id.Region.equal r.Mir_region.id id)
                  regions
              with
              | Some { Mir_region.init = Mir_region.Bound; _ } -> true
              | Some { Mir_region.init = Mir_region.Uninitialized; _ } ->
                  Mir_id.Region.equal id record
              | _ -> false
            in
            if wanted then
              match Mir_interp.Binding.instance binding id with
              | Some key -> Mir_memory.write_string memory key ~offset:0L bytes
              | None -> ())
          after;
        Mir_record.of_outcome memory binding ~sites ~record
          (Mir_interp.Outcome.Success
             [ Mir_datum.Bits (Int64.logand status 0xFFFF_FFFFL) ])
  in
  Ok
    {
      Route.Exec.instantiate =
        (fun memory ~shared ->
          Mir_interp.instantiate ~shared (regions_program phys) memory
            ~bound:(fun _ -> None));
      run;
      artifact = Some (fun fmt -> Art.pp_summary fmt artifact);
      traffic = (fun () -> None);
    }

let route ?(allocation = Allocation.Reference)
    ?(runtime = Machine_rivet_common.Rivet_runtime.Dependency_free) () =
  Route.Route.Custom
    {
      Route.Custom.name =
        Fmt.str "x86_64 emulated %s %s"
          (Allocation.name allocation)
          (Machine_rivet_common.Rivet_runtime.name runtime);
      exec = (fun ~sites g -> exec_of ~allocation ~runtime ~sites g);
    }
