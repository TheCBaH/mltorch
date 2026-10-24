(* The RV32 route of a model bundle on a Coral simulator: an invocation is
   published as on the qemu route, its module built into a Coral guest image
   whose runtime calls [mir_entry] with the table of region addresses, and a
   run is one simulator process. A bound region is an [In_out] slot, the
   failure record an [Out] slot, every other region [Scratch]. An artifact with
   Zve32f forms is lowered under the vector profile and needs a configuration
   with Zve32f (VLEN 128 on every Coral core); one without the extension is
   refused.

   The configuration's simulator picks the runner. A standalone core (MPACT,
   RTL) takes the bound bytes in through its input frame and returns the
   written slots in its result frame ([Coral_run]). The subsystem has no HTIF:
   the bound bytes go in the ELF, and the written slots come back over the SPI
   bridge at about a kilobyte a second ([Coral_soc_run]), so only the slots
   whose guest hash says they changed are read, within a byte budget. *)

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

(* Fault injection for the evidence suite: one deliberate route defect a run
   must detect. No consumer passes one. *)
module Mutation = struct
  type t =
    | Misplaced_output
        (** each written slot's bytes stored into the next written slot of the
            same size, as a read-back from the wrong index would *)
end

(* What one run measured. *)
module Run = struct
  type t = { mcycle : int64; minstret : int64; wall : float }
end

(* The subsystem's default read-back budget: about sixteen seconds of bridge
   time. *)
let read_back_budget = 16384L

let slots ?(home = fun _ -> Coral_layout.Home.Any) ~record artifact =
  List.map
    (fun (s : Table.slot) ->
      {
        Coral_layout.Slot.align = Int64.max 16L s.Table.align;
        home = home s;
        placement =
          (match s.Table.section with
          | Art.Section.Bound -> Coral_layout.Placement.In_out
          | _ when Mir_id.Region.equal s.Table.region record ->
              Coral_layout.Placement.Out
          | _ -> Coral_layout.Placement.Scratch);
        size = s.Table.size;
      })
    (Machine_rivet_riscv32_mir.Rv32_mir_module.slots artifact)

let image ?mutation ?home (c : Config.t) ({ Q.artifact; record } : Q.published)
    =
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
       (slots ?home ~record artifact))

(* The run on the configuration's simulator: each written slot's bytes, by
   index, the status and what was measured. *)
let simulate ?(budget = read_back_budget) (c : Config.t) (image : Coral_image.t)
    ~bound =
  match image.Coral_image.profile.Coral_profile.simulator with
  | Coral_profile.Simulator.Mpact | Coral_profile.Simulator.Rtl ->
      let* o =
        Result.map_error
          (Fmt.str "%a" Coral_run.Run_error.pp)
          (Err.payload
             (Coral_run.run ~deadline:c.Config.deadline ~flags:c.Config.flags
                ~dir:c.Config.dir image ~bound))
      in
      Ok
        ( o.Coral_run.outputs,
          o.Coral_run.status,
          {
            Run.mcycle = o.Coral_run.mcycle;
            minstret = o.Coral_run.minstret;
            wall = o.Coral_run.wall;
          } )
  | Coral_profile.Simulator.Subsystem ->
      let read_back =
        List.concat
          (List.mapi
             (fun k (s : Coral_layout.Placed_slot.t) ->
               if Coral_layout.Placement.written s.slot.placement then [ k ]
               else [])
             image.Coral_image.layout.Coral_layout.slots)
      in
      let* o =
        Result.map_error
          (Fmt.str "%a" Coral_soc_run.Run_error.pp)
          (Err.payload
             (Coral_soc_run.run ~deadline:c.Config.deadline ~read_back
                ~skip_unchanged:true ~budget ~dir:c.Config.dir image ~bound))
      in
      Ok
        ( o.Coral_soc_run.read_back,
          o.Coral_soc_run.status,
          {
            Run.mcycle = o.Coral_soc_run.mcycle;
            minstret = o.Coral_soc_run.minstret;
            wall = o.Coral_soc_run.wall;
          } )

(* The outputs as [Misplaced_output] leaves them: slot [k]'s bytes under the
   next written slot of its size, if there is one. *)
let misplace outputs =
  let n = List.length outputs in
  List.mapi
    (fun i (k, bytes) ->
      let rec next j =
        if j = n then (k, bytes)
        else
          let k', b' = List.nth outputs ((i + j) mod n) in
          if k' <> k && String.length b' = String.length bytes then (k', bytes)
          else next (j + 1)
      in
      next 1)
    outputs

let run_once ?(on_run = fun _ -> ()) ?route_mutation ?budget (c : Config.t)
    image ~(published : Q.published) memory binding =
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
  let* outputs, status, measured =
    simulate ?budget c image ~bound:(fun k -> bound.(k))
  in
  on_run measured;
  let outputs =
    match route_mutation with
    | Some Mutation.Misplaced_output -> misplace outputs
    | None -> outputs
  in
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
      (Ok ()) outputs
  in
  Ok status

let exec_of ?mutation ?route_mutation ?registers ?on_run ?home ?budget c ~sites
    g =
  let* published =
    Q.publish ?registers ~allocation:Q.Allocation.Scanned ~sites g
  in
  let* image = image ?mutation ?home c published in
  let phys = Art.program published.Q.artifact in
  let run ?fuel:_ memory binding ~invocation:_ =
    match
      run_once ?on_run ?route_mutation ?budget c image ~published memory binding
    with
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

(* [on_run] sees what every run measured. [home] places each slot (default
   [Any]); [budget] bounds the subsystem's read-back bytes (default
   {!read_back_budget}). *)
let route ?mutation ?route_mutation ?registers ?on_run ?home ?budget
    (c : Config.t) =
  Route.Route.Custom
    {
      Route.Custom.name = "riscv32 coral " ^ c.Config.configuration;
      exec =
        (fun ~sites g ->
          exec_of ?mutation ?route_mutation ?registers ?on_run ?home ?budget c
            ~sites g);
    }
