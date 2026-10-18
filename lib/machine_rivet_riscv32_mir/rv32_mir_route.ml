(* The RV32 route of a model bundle: each invocation's generic program is
   selected onto ILP32, allocated, framed and published, then emitted as a
   typed Rivet module and planned with the process harness into a static ELF,
   built once. A run is one process under qemu-riscv32: the invocation's
   bound regions and failure record are slots of the shared output file,
   written before and read back after, so the tensors stay the context's own
   memory and a bundle's routes compare byte for byte; the other regions are
   the ELF's zero-filled data. *)

open Machine_ir
open Machine_interp
open Machine_target_riscv32
module A = Machine_alloc.Mir_ref_alloc.Make (Rv32) (Rv32_regs)
module Ls = Machine_alloc.Mir_linear_scan.Make (Rv32) (Rv32_regs)
module Sch = Machine_alloc.Mir_schedule.Make (Rv32)
module Fr = Machine_alloc.Mir_frame.Make (Rv32) (Rv32_frame)
module Pub = Machine_model.Mir_artifact.Make (Rv32)
module Art = Machine_model.Mir_artifact
module Route = Machine_model.Mir_model_route
module Table = Machine_rivet_common.Rivet_table
module H = Machine_rivet_riscv32.Rv32_harness
module Profile = Machine_rivet_riscv32.Rv32_profile

let ( let* ) = Result.bind

(* Which allocator makes the physical program. *)
module Allocation = struct
  type t =
    | Reference  (** the stack-based reference allocator *)
    | Scanned  (** sink scheduling, then split linear scan *)

  let name = function Reference -> "reference" | Scanned -> "scanned"
end

type published = {
  artifact : (Rv32_op.t, Rv32_op.test) Art.t;
  record : Mir_id.Region.t;
}

(* Selection, allocation, frame realization and publication. *)
let publish ?select_mutation ~allocation ~sites (g : Mir_verify.Generic.t) =
  let* planning =
    Option.to_result ~none:"a generic program with no planning summary"
      (Mir_verify.Generic.program g).Mir_program.planning
  in
  let* res =
    Result.map_error
      (Fmt.str "%a" Rv32_select.Refusal.pp)
      (Err.payload
         (Rv32_select.program ?mutation:select_mutation ~sites
            ~unlisted:Mir_failure.Unlisted.Unreachable g))
  in
  let* v, phys =
    match allocation with
    | Allocation.Reference ->
        Ok (res.Rv32_select.selected, A.allocate res.Rv32_select.selected)
    | Allocation.Scanned ->
        let* s =
          Result.map_error
            (Fmt.str "schedule: %a" Machine_alloc.Mir_schedule.Refusal.pp)
            (Sch.schedule Machine_alloc.Mir_schedule.Policy.Sink
               res.Rv32_select.selected)
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
  let sel = Rv32_stage.Sel.Verified.selected v in
  let* view =
    Option.to_result ~none:"the failure record names no view"
      (Mir_program.find_view sel.Rv32_stage.Sel.program res.Rv32_select.record)
  in
  Ok { artifact; record = view.Mir_view.region }

(* The process slots, in the table's order: a bound region or the record is
   in the output file, so its bytes go in and come back; any other region is
   zero-filled memory of the ELF. *)
let harness_slots ~record artifact =
  List.map
    (fun (s : Table.slot) ->
      let placement =
        match s.Table.section with
        | Art.Section.Bound -> H.Placement.Out
        | _ when Mir_id.Region.equal s.Table.region record -> H.Placement.Out
        | _ -> H.Placement.Bss
      in
      {
        H.size = Int64.to_int s.Table.size;
        align = Int64.to_int s.Table.align;
        placement;
      })
    (Rv32_mir_module.slots artifact)

(* The profile an artifact needs: the vector one when it uses Zve32f forms,
   else the default. Rivet's feature set and qemu's CPU follow it. *)
let profile artifact =
  if
    List.mem Mir_target.Feature.Zve32f
      (Art.identity artifact).Art.Identity.features
  then Profile.Rv32imf_zve32f_ilp32f
  else Profile.Rv32imf_ilp32f

(* The artifact as a process image: its module and the harness, planned
   under its profile. *)
let prepare ?mutation { artifact; record } =
  let* m =
    Result.map_error
      (Fmt.str "%a" Rv32_mir_form.Refusal.pp)
      (Err.payload (Rv32_mir_module.of_artifact ?mutation artifact))
  in
  H.prepare ~profile:(profile artifact) ~entry:Table.entry ~kernel:[ m ]
    (harness_slots ~record artifact)

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

let region_bytes memory binding (region : Mir_id.Region.t) size =
  match Mir_interp.Binding.instance binding region with
  | None -> Error "a region with no memory instance"
  | Some key ->
      Ok
        ( key,
          String.init size (fun i ->
              match
                (Mir_memory.read_bytes memory key ~offset:(Int64.of_int i) ~n:1).(
                0)
              with
              | Some b -> Char.chr b
              | None -> '\000') )

(* One run of a prepared image over a context's memory: its bound regions
   in, the process, its bound regions and record out. *)
let run_once ?cpu (prepared : H.prepared) ~published memory binding =
  let slots = Rv32_mir_module.slots published.artifact in
  let* bound =
    List.fold_left
      (fun acc (s : Table.slot) ->
        let* acc = acc in
        match s.Table.section with
        | Art.Section.Bound ->
            let* _, bytes =
              region_bytes memory binding s.Table.region
                (Int64.to_int s.Table.size)
            in
            Ok (Some bytes :: acc)
        | _ -> Ok (None :: acc))
      (Ok []) slots
  in
  let bound = Array.of_list (List.rev bound) in
  let* outcome = H.run ?cpu prepared ~bound:(fun k -> bound.(k)) in
  let* () =
    List.fold_left
      (fun acc (k, bytes) ->
        let* () = acc in
        let s = List.nth slots k in
        match Mir_interp.Binding.instance binding s.Table.region with
        | None -> Error "a region with no memory instance"
        | Some key ->
            Mir_memory.write_string memory key ~offset:0L bytes;
            Ok ())
      (Ok ()) outcome.H.outputs
  in
  Ok outcome.H.status

let exec_of ?mutation ?select_mutation ?cpu ~allocation ~sites
    (g : Mir_verify.Generic.t) =
  let* published = publish ?select_mutation ~allocation ~sites g in
  let* prepared = prepare ?mutation published in
  let phys = Art.program published.artifact in
  let run ?fuel:_ memory binding ~invocation:_ =
    match run_once ?cpu prepared ~published memory binding with
    | Error e -> Mir_observation.Status.Unsupported e
    | Ok status ->
        Mir_record.of_outcome memory binding ~sites ~record:published.record
          (Mir_interp.Outcome.Success
             [
               Mir_datum.Bits
                 (Int64.logand (Int64.of_int32 status) 0xFFFF_FFFFL);
             ])
  in
  Ok
    {
      Route.Exec.instantiate =
        (fun memory ~shared ->
          Mir_interp.instantiate ~shared (regions_program phys) memory
            ~bound:(fun _ -> None));
      run;
      artifact = Some (fun fmt -> Art.pp_summary fmt published.artifact);
      traffic = (fun () -> None);
    }

let route ?mutation ?select_mutation ?cpu ?(allocation = Allocation.Scanned) ()
    =
  Route.Route.Custom
    {
      Route.Custom.name = Fmt.str "riscv32 qemu %s" (Allocation.name allocation);
      exec =
        (fun ~sites g ->
          exec_of ?mutation ?select_mutation ?cpu ~allocation ~sites g);
    }
