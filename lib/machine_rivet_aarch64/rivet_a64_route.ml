(* The native AArch64 route of a model bundle: each invocation's generic
   program is selected, allocated, framed and published, then lowered through
   typed Rivet modules into an image of its own, loaded once and called on every
   run. A run copies the invocation's storage into the image, calls it and
   copies the results back: the tensors stay the context's own memory, which
   the interpreters share, so a bundle's routes compare byte for byte. The CPU
   cannot say which bytes a kernel wrote, so every byte of a bound region is
   defined after a run. *)

open Machine_ir
open Machine_interp
open Machine_target_aarch64
module A = Machine_alloc.Mir_ref_alloc.Make (A64) (A64_regs)
module Ls = Machine_alloc.Mir_linear_scan.Make (A64) (A64_regs)
module Sch = Machine_alloc.Mir_schedule.Make (A64)
module Fr = Machine_alloc.Mir_frame.Make (A64) (A64_frame)
module Pub = Machine_model.Mir_artifact.Make (A64)
module Art = Machine_model.Mir_artifact
module Route = Machine_model.Mir_model_route
module M = Rivet_a64_module
module I = Rivet_a64_image

let ( let* ) = Result.bind

(* Which allocator makes the physical program. *)
module Allocation = struct
  type t =
    | Reference  (** the stack-based reference allocator *)
    | Scanned  (** sink scheduling, then split linear scan *)

  let name = function Reference -> "reference" | Scanned -> "scanned"
end

type published = {
  artifact : (A64_op.t, A64_op.test) Art.t;
  record : Mir_id.Region.t;
}

(* Selection, reference allocation, frame realization and publication. *)
let publish ~allocation ~sites (g : Mir_verify.Generic.t) =
  let* planning =
    Option.to_result ~none:"a generic program with no planning summary"
      (Mir_verify.Generic.program g).Mir_program.planning
  in
  let* res =
    Result.map_error
      (Fmt.str "%a" A64_select.Refusal.pp)
      (Err.payload
         (A64_select.program ~sites ~unlisted:Mir_failure.Unlisted.Unreachable g))
  in
  let* v, phys =
    match allocation with
    | Allocation.Reference ->
        Ok (res.A64_select.selected, A.allocate res.A64_select.selected)
    | Allocation.Scanned ->
        let* s =
          Result.map_error
            (Fmt.str "schedule: %a" Machine_alloc.Mir_schedule.Refusal.pp)
            (Sch.schedule Machine_alloc.Mir_schedule.Policy.Sink
               res.A64_select.selected)
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
  let sel = A64_stage.Sel.Verified.selected v in
  let* view =
    Option.to_result ~none:"the failure record names no view"
      (Mir_program.find_view sel.A64_stage.Sel.program res.A64_select.record)
  in
  Ok { artifact; record = view.Mir_view.region }

let region_symbol r = Fmt.str "mir_region_%d" (Mir_id.Region.to_int r)

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

(* The typed modules that make an artifact an image — its own and the host
   stubs for its helpers — and the entry symbol. *)
let modules ?mutation ~runtime artifact =
  let phys = Art.program artifact in
  let render pp e = Fmt.str "%a" pp e in
  let* () =
    Result.map_error
      (render Rivet_a64_refusal.pp)
      (Rivet_a64_runtime.admit runtime artifact)
  in
  let* modul =
    Result.map_error
      (render Rivet_a64_refusal.pp)
      (Err.payload (M.of_artifact ?mutation artifact))
  in
  let helpers = Rivet_a64_runtime.helpers artifact in
  let* host =
    if helpers = [] then Ok []
    else
      Result.map
        (fun m -> [ m ])
        (Result.map_error
           (render Rivet_a64_refusal.pp)
           (Err.payload (M.helpers ~host_symbol:I.host_symbol helpers)))
  in
  let main =
    (List.find
       (fun (f : (_, _) Mir_phys.Func.t) ->
         Mir_id.Func.equal f.Mir_phys.Func.id phys.Mir_phys.Program.main)
       phys.Mir_phys.Program.funcs)
      .Mir_phys.Func.name
  in
  Ok (main, modul :: host)

(* Typed modules as a loaded image entered at [entry]. *)
let load ~entry modules =
  let* laid =
    Result.map_error (Fmt.str "%a" I.Error.pp)
      (Err.payload (I.plan ~entry modules))
  in
  Result.map_error (Fmt.str "%a" I.Error.pp) (Err.payload (I.load laid))

(* The artifact as a loaded image: the main function's name is its entry. *)
let image ?mutation ~runtime artifact =
  let* entry, modules = modules ?mutation ~runtime artifact in
  load ~entry modules

(* A loaded image closes when nothing refers to its kernel any more. *)
type kernel = { loaded : I.t }

let kernel loaded =
  let k = { loaded } in
  Gc.finalise (fun k -> I.close k.loaded) k;
  k

let exec_of ?mutation ?check ~allocation ~runtime ~sites
    (g : Mir_verify.Generic.t) =
  let* { artifact; record } = publish ~allocation ~sites g in
  let* () =
    match check with
    | None -> Ok ()
    | Some check ->
        let* entry, ms = modules ?mutation ~runtime artifact in
        check ~entry ms
  in
  let* loaded = image ?mutation ~runtime artifact in
  let k = kernel loaded in
  let phys = Art.program artifact in
  let regions = phys.Mir_phys.Program.regions in
  let io r = Result.map_error (Fmt.str "%a" I.Error.pp) (Err.payload r) in
  let bound_bytes memory binding (r : Mir_region.t) =
    match Mir_interp.Binding.instance binding r.Mir_region.id with
    | None -> Error "a region with no memory instance"
    | Some key ->
        let n = Int64.to_int r.Mir_region.size in
        Ok
          ( key,
            String.init n (fun i ->
                match
                  (Mir_memory.read_bytes memory key ~offset:(Int64.of_int i)
                     ~n:1).(0)
                with
                | Some b -> Char.chr b
                | None -> '\000') )
  in
  let run ?fuel:_ memory binding ~invocation:_ =
    let result =
      let* () =
        List.fold_left
          (fun acc (r : Mir_region.t) ->
            let* () = acc in
            match r.Mir_region.init with
            | Mir_region.Bound ->
                let* _, bytes = bound_bytes memory binding r in
                io
                  (I.write_global k.loaded
                     (region_symbol r.Mir_region.id)
                     bytes)
            | Mir_region.Constant _ | Mir_region.Uninitialized -> Ok ())
          (Ok ()) regions
      in
      let* status = io (I.call k.loaded) in
      let* () =
        List.fold_left
          (fun acc (r : Mir_region.t) ->
            let* () = acc in
            let wanted =
              match r.Mir_region.init with
              | Mir_region.Bound -> true
              | Mir_region.Uninitialized ->
                  Mir_id.Region.equal r.Mir_region.id record
              | Mir_region.Constant _ -> false
            in
            if not wanted then Ok ()
            else
              match Mir_interp.Binding.instance binding r.Mir_region.id with
              | None -> Error "a region with no memory instance"
              | Some key ->
                  let* bytes =
                    io (I.read_global k.loaded (region_symbol r.Mir_region.id))
                  in
                  Mir_memory.write_string memory key ~offset:0L bytes;
                  Ok ())
          (Ok ()) regions
      in
      Ok status
    in
    match result with
    | Error e -> Mir_observation.Status.Unsupported e
    | Ok status ->
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

let route ?mutation ?check ?(allocation = Allocation.Reference)
    ?(runtime = Rivet_a64_runtime.Dependency_free) () =
  Route.Route.Custom
    {
      Route.Custom.name =
        Fmt.str "aarch64 native %s %s"
          (Allocation.name allocation)
          (Rivet_a64_runtime.name runtime);
      exec =
        (fun ~sites g -> exec_of ?mutation ?check ~allocation ~runtime ~sites g);
    }
