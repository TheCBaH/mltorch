(* The RV32 route of a model bundle on a Coral simulator: an invocation is
   published as on the qemu route, its module built into a Coral guest image
   whose runtime calls [mir_entry] with the table of region addresses, and a
   run is one simulator process. A bound region is an [In_out] slot (its bytes
   in through the input frame, back in the result frame), the failure record
   an [Out] slot, every other region [Scratch]. An artifact with Zve32f forms
   is lowered under the vector profile and needs a configuration with Zve32f
   (VLEN 128 on every Coral core); one without the extension is refused. *)

open Machine_ir
open Machine_interp
module Q = Machine_rivet_riscv32_mir.Rv32_mir_route
module Art = Machine_model.Mir_artifact
module Route = Machine_model.Mir_model_route
module Table = Machine_rivet_common.Rivet_table
module Profile = Machine_rivet_riscv32.Rv32_profile

let ( let* ) = Result.bind

(* Where and on what the route runs: the extracted emulators archive, one of
   its configurations, the wall-clock limit of one call (seconds), and the
   simulator's extra flags. *)
module Config = struct
  type t = {
    dir : string;
    configuration : string;
    deadline : float;
    flags : string list;
  }
end

let slots ~record artifact =
  List.map
    (fun (s : Table.slot) ->
      {
        Coral_layout.Slot.align = Int64.max 16L s.Table.align;
        placement =
          (match s.Table.section with
          | Art.Section.Bound -> Coral_layout.Placement.In_out
          | _ when Mir_id.Region.equal s.Table.region record ->
              Coral_layout.Placement.Out
          | _ -> Coral_layout.Placement.Scratch);
        size = s.Table.size;
      })
    (Machine_rivet_riscv32_mir.Rv32_mir_module.slots artifact)

let image ?mutation (c : Config.t) ({ Q.artifact; record } : Q.published) =
  let render pp r = Result.map_error (Fmt.str "%a" pp) (Err.payload r) in
  let* profile =
    render Coral_profile.pp_error
      (Coral_profile.load ~dir:c.Config.dir c.Config.configuration)
  in
  let rv32_profile = Q.profile artifact in
  let* () =
    if
      rv32_profile = Profile.Rv32imf_zve32f_ilp32f
      && not (List.mem "zve32f" profile.Coral_profile.extensions)
    then
      Error
        (Fmt.str "configuration %s has no Zve32f, which the kernel uses"
           c.Config.configuration)
    else Ok ()
  in
  let* m =
    render Machine_rivet_riscv32_mir.Rv32_mir_form.Refusal.pp
      (Machine_rivet_riscv32_mir.Rv32_mir_module.of_artifact ?mutation artifact)
  in
  render Coral_image.pp_error
    (Coral_image.build ~rv32_profile profile ~entry:Table.entry ~kernel:[ m ]
       (slots ~record artifact))

let run_once ?(on_run = fun _ -> ()) (c : Config.t) image
    ~(published : Q.published) memory binding =
  let table =
    Machine_rivet_riscv32_mir.Rv32_mir_module.slots published.Q.artifact
  in
  let* bound =
    List.fold_left
      (fun acc (s : Table.slot) ->
        let* acc = acc in
        match s.Table.section with
        | Art.Section.Bound ->
            let* _, bytes =
              Q.region_bytes memory binding s.Table.region
                (Int64.to_int s.Table.size)
            in
            Ok (Some bytes :: acc)
        | _ -> Ok (None :: acc))
      (Ok []) table
  in
  let bound = Array.of_list (List.rev bound) in
  let* outcome =
    Result.map_error
      (Fmt.str "%a" Coral_run.Run_error.pp)
      (Err.payload
         (Coral_run.run ~deadline:c.Config.deadline ~flags:c.Config.flags
            ~dir:c.Config.dir image ~bound:(fun k -> bound.(k))))
  in
  on_run outcome;
  let* () =
    List.fold_left
      (fun acc (k, bytes) ->
        let* () = acc in
        let s = List.nth table k in
        match Mir_interp.Binding.instance binding s.Table.region with
        | None -> Error "a region with no memory instance"
        | Some key ->
            Mir_memory.write_string memory key ~offset:0L bytes;
            Ok ())
      (Ok ()) outcome.Coral_run.outputs
  in
  Ok outcome.Coral_run.status

let exec_of ?mutation ?on_run c ~sites g =
  let* published = Q.publish ~allocation:Q.Allocation.Scanned ~sites g in
  let* image = image ?mutation c published in
  let phys = Art.program published.Q.artifact in
  let run ?fuel:_ memory binding ~invocation:_ =
    match run_once ?on_run c image ~published memory binding with
    | Error e -> Mir_observation.Status.Unsupported e
    | Ok status ->
        Mir_record.of_outcome memory binding ~sites ~record:published.Q.record
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
          Mir_interp.instantiate ~shared (Q.regions_program phys) memory
            ~bound:(fun _ -> None));
      run;
      artifact = Some (fun fmt -> Art.pp_summary fmt published.Q.artifact);
      traffic = (fun () -> None);
    }

(* [on_run] sees every run's outcome: its cycles and instructions retired. *)
let route ?mutation ?on_run (c : Config.t) =
  Route.Route.Custom
    {
      Route.Custom.name = "riscv32 coral " ^ c.Config.configuration;
      exec = (fun ~sites g -> exec_of ?mutation ?on_run c ~sites g);
    }
