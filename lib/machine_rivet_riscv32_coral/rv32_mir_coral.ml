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

(* Whether the generated functions may move their operands with the DMA
   engine. [Forbidden] reads them where they lie (the default, and the only
   choice on a core with no DMA: the standalone cores and MPACT). [Allowed]
   takes, for each contraction, the plan {!Coral_transfer.Transfer_model.choose}
   ranks cheapest, copying or streaming through the DTCM when that wins; the
   configuration must have a DMA-reachable region. Alphabetical. *)
module Dma = struct
  type t = Allowed | Forbidden
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

let slots ?(home = fun _ -> Coral_layout.Home.Any) ?(slack = 0L) ~record
    artifact =
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
        size = Int64.add s.Table.size slack;
      })
    (Machine_rivet_riscv32_mir.Rv32_mir_module.slots artifact)

let image ?mutation ?kernel_mutation ?home ?(engine = true) ?engine_mutation
    ?transfer ?dma ?itcm (c : Config.t) ({ Q.artifact; record } : Q.published) =
  let render pp r = Result.map_error (Fmt.str "%a" pp) (Err.payload r) in
  let* profile =
    render Coral_profile.pp_error
      (Coral_profile.load
         ~traced:
           (List.exists (String.starts_with ~prefix:"--trace") c.Config.flags)
         ~dir:c.Config.dir c.Config.configuration)
  in
  (* a core with less instruction memory than the configuration's *)
  let profile =
    match itcm with
    | None -> profile
    | Some bytes ->
        {
          profile with
          Coral_profile.regions =
            List.map
              (fun (r : Coral_profile.Region.t) ->
                if r.Coral_profile.Region.role = Coral_profile.Role.Itcm then
                  {
                    r with
                    Coral_profile.Region.size =
                      Int64.min r.Coral_profile.Region.size bytes;
                  }
                else r)
              profile.Coral_profile.regions;
        }
  in
  let* transfer =
    match (transfer, dma) with
    | Some _, Some _ -> Error "give ~transfer or ~dma, not both"
    | transfer, (None | Some Dma.Forbidden) -> Ok transfer
    | None, Some Dma.Allowed ->
        if
          List.exists
            (fun (r : Coral_profile.Region.t) ->
              r.Coral_profile.Region.dma_reachable)
            profile.Coral_profile.regions
        then
          Ok
            (Some
               (fun ~budget d -> Coral_transfer.Transfer_model.choose ~budget d))
        else
          Error
            (Fmt.str "configuration %s has no DMA engine (~dma Allowed)"
               c.Config.configuration)
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
  (* on a configuration with the matrix engine, an ordered contraction runs
     there; every other one on the vector unit *)
  let capabilities =
    {
      Contraction.Select.Capabilities.matrix =
        engine && List.mem "zvt" profile.Coral_profile.extensions;
      vector = List.mem "zve32f" profile.Coral_profile.extensions;
    }
  in
  let on_engine d =
    fst (Contraction.Select.choose capabilities d)
    = Contraction.Select.Engine.Zvt
  in
  (* every panel and ring lives in DTCM: the image's functions share five
     eighths of it, so the stack, the staging area and the program's own
     data keep the rest; each takes what the ones before it left *)
  let left =
    ref
      (List.fold_left
         (fun n (r : Coral_profile.Region.t) ->
           match r.Coral_profile.Region.role with
           | Coral_profile.Role.Dtcm ->
               n
               + Int64.to_int
                   (Int64.div (Int64.mul r.Coral_profile.Region.size 5L) 8L)
           | _ -> n)
         0 profile.Coral_profile.regions)
  in
  (* an operand a packed function reads in place is read past its end: every
     slot gets that much more room *)
  let slack = ref 0 in
  let dtcm =
    List.find_map
      (fun (r : Coral_profile.Region.t) ->
        match r.Coral_profile.Region.role with
        | Coral_profile.Role.Dtcm ->
            Some
              ( Int64.to_int r.Coral_profile.Region.base,
                Int64.to_int r.Coral_profile.Region.size )
        | _ -> None)
      profile.Coral_profile.regions
  in
  let other d =
    if on_engine d then
      let module Z = Coralnpu_kernels_zvt in
      let module P = Z.Zvt_packed_contraction in
      (* a B read in place that several blocks read is copied into the DTCM
         unless it already lies there *)
      let stage_b =
        match dtcm with
        | Some (base, bytes) when P.reuses_b d ->
            Some (P.Stage.Outside { base; bytes })
        | _ -> None
      in
      let fitting stage_b =
        match P.panel_bytes ?stage_b d with
        | Ok bytes when bytes <= !left && P.fast d -> Some (stage_b, bytes)
        | _ -> None
      in
      match
        match fitting stage_b with Some _ as f -> f | None -> fitting None
      with
      | Some (stage_b, bytes) -> (
          match P.module_ ?stage_b d with
          | Ok _ as m ->
              left := !left - bytes;
              slack := max !slack (P.overread d);
              Some m
          | Error _ ->
              Some (Z.Zvt_contraction.module_ ?mutation:engine_mutation d))
      | None -> Some (Z.Zvt_contraction.module_ ?mutation:engine_mutation d)
    else None
  in
  (* each function asks [transfer] for its plan within what is left, and a
     plan that does not check is an error *)
  let plan =
    Option.map
      (fun choose ->
        fun d ->
         let module T = Coral_transfer.Transfer_plan in
         match T.check ~budget:!left d (choose ~budget:!left d) with
         | Ok w ->
             left := !left - T.Witness.bytes w;
             Ok (Some w)
         | Error r -> Error (Fmt.str "transfer plan: %a" T.Refusal.pp r))
      transfer
  in
  (* an attention is the fused function on the matrix engine, within what the
     functions before it left *)
  (* each attention function built, by its entry symbol: on the subsystem one
     that does not fit beside the kernel runs as an overlay too *)
  let attended = ref [] in
  let attention d =
    if not capabilities.Contraction.Select.Capabilities.matrix then
      Error "the core has no matrix engine"
    else
      match Coralnpu_kernels_zvt.Zvt_sdpa.panel_bytes d with
      | Error why -> Error why
      | Ok bytes when bytes > !left ->
          Error (Fmt.str "the panels need %d bytes, %d available" bytes !left)
      | Ok bytes -> (
          match Coralnpu_kernels_zvt.Zvt_sdpa.module_ d with
          | Ok md as m ->
              left := !left - bytes;
              attended := (Attention_block.symbol d, md) :: !attended;
              m
          | Error _ as e -> e)
  in
  let* helpers =
    Q.helper_modules ?kernel_mutation ?plan ~other ~attention artifact
  in
  let build ?overlays ?(resident = helpers) ?(rv32_profile = rv32_profile) m =
    Coral_image.build ?overlays ~rv32_profile
      ~matrix:
        (List.exists on_engine (Q.generated artifact)
        || Q.attentions artifact <> [])
      profile ~entry:Table.entry ~kernel:(m :: resident)
      (slots ?home ~slack:(Int64.of_int !slack) ~record artifact)
  in
  let first = build m in
  (* a kernel past the instruction memory runs as overlays when the core has
     the DMA that loads them: two of its regions move out of the image *)
  let outlined () =
    let module O = Machine_rivet_riscv32_mir.Rv32_mir_outline in
    let vector =
      Profile.vector rv32_profile
      || List.mem "zve32f" profile.Coral_profile.extensions
    in
    (* the attention functions leave the resident image as overlays of their
       own, each reached through the stub that bears its symbol *)
    let attentions =
      List.map
        (fun (symbol, md) ->
          {
            Coral_overlay.Overlay.name = symbol;
            entry = symbol;
            modules = [ md ];
          })
        !attended
    in
    let resident =
      List.filter
        (fun h -> not (List.exists (fun (_, md) -> md == h) !attended))
        helpers
    in
    (* beside the attentions, the slot is at least the largest of them: every
       kernel region that fits it and calls no attention moves out too *)
    let outline =
      match attentions with
      | [] -> O.of_artifact ?mutation artifact
      | _ ->
          let text (o : Coral_overlay.Overlay.t) =
            match
              Err.payload
                (Coral_overlay.link ~rv32_profile:Profile.Rv32imf_zve32f_ilp32f
                   ~slot_base:0L ~slot_size:0x1000000L ~data_base:0x1000000L
                   ~data_size:0x1000000L o)
            with
            | Ok l -> String.length l.Coral_overlay.Linked.text
            | Error _ -> 0
          in
          let slot = List.fold_left (fun n o -> max n (text o)) 0 attentions in
          O.of_artifact ?mutation ~avoid:(List.map fst !attended)
            ~choose:(O.fill ~slot) artifact
    in
    let regions, m =
      match outline with
      | Ok (m', (_ :: _ as overlays)) ->
          let modules =
            List.map (fun (o : O.outlined) -> (o, O.module_ o)) overlays
          in
          if List.exists (fun (_, r) -> Result.is_error r) modules then ([], m)
          else
            ( List.map
                (fun ((o : O.outlined), r) ->
                  {
                    Coral_overlay.Overlay.name = o.entry;
                    entry = o.entry;
                    modules = [ Result.get_ok r ];
                  })
                modules,
              m' )
      | Ok _ | Error _ -> ([], m)
    in
    match regions @ attentions with
    | [] -> None
    | overlays when vector ->
        Some
          (build
             ~rv32_profile:
               (* the manager's [fence.i] is Zifencei, which the scalar
                  profile leaves out and the vector ones enable *)
               (if Profile.vector rv32_profile then rv32_profile
                else Profile.Rv32imf_zve32f_ilp32f)
             ~overlays ~resident m)
    | _ -> None
  in
  let retried =
    match Err.payload first with
    | Error (`Over_capacity { Coral_layout.Over_capacity.region = "itcm"; _ })
      when profile.Coral_profile.simulator = Coral_profile.Simulator.Subsystem
      ->
        outlined ()
    | _ -> None
  in
  render Coral_image.pp_error (Option.value retried ~default:first)

(* The run on the configuration's simulator: each written slot's bytes, by
   index, the status and what was measured. *)
let simulate ?(budget = read_back_budget) ?label ?expect_cycles (c : Config.t)
    (image : Coral_image.t) ~bound =
  match image.Coral_image.profile.Coral_profile.simulator with
  | Coral_profile.Simulator.Mpact | Coral_profile.Simulator.Rtl ->
      let* o =
        Result.map_error
          (Fmt.str "%a" Coral_run.Run_error.pp)
          (Err.payload
             (Coral_run.run ~deadline:c.Config.deadline ~flags:c.Config.flags
                ?label ?expect_cycles ~dir:c.Config.dir image ~bound))
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
             (Coral_soc_run.run ~deadline:c.Config.deadline ?label ~read_back
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

let run_once ?(on_run = fun _ -> ()) ?(on_elf = fun _ -> ()) ?route_mutation
    ?budget ?label ?expect_cycles (c : Config.t) image
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
  on_elf image.Coral_image.elf;
  let* outputs, status, measured =
    simulate ?budget ?label ?expect_cycles c image ~bound:(fun k ->
        (* the slots an image adds for its overlays are not the table's *)
        if k < Array.length bound then bound.(k) else None)
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

let exec_of ?mutation ?kernel_mutation ?route_mutation ?registers ?engine
    ?engine_mutation ?transfer ?dma ?itcm ?on_run ?on_elf
    ?(on_image = fun _ -> ()) ?home ?budget ?label ?expect_cycles c ~sites g =
  let* published =
    Q.publish ?registers ~allocation:Q.Allocation.Scanned ~sites g
  in
  let* image =
    image ?mutation ?kernel_mutation ?home ?engine ?engine_mutation ?transfer
      ?dma ?itcm c published
  in
  (* where each table slot landed: its Machine IR region and the memory's
     role (an image's slots are the table's, in its order, then any the image
     adds itself: the overlays' three) *)
  let table_slots =
    Machine_rivet_riscv32_mir.Rv32_mir_module.slots published.Q.artifact
  in
  on_image
    (List.map2
       (fun (s : Table.slot) (p : Coral_layout.Placed_slot.t) ->
         ( s.Table.region,
           match
             List.find_opt
               (fun (r : Coral_profile.Region.t) ->
                 String.equal r.Coral_profile.Region.name
                   p.Coral_layout.Placed_slot.region)
               image.Coral_image.profile.Coral_profile.regions
           with
           | Some r -> r.Coral_profile.Region.role
           | None -> Coral_profile.Role.Ram ))
       table_slots
       (List.filteri
          (fun i _ -> i < List.length table_slots)
          image.Coral_image.layout.Coral_layout.slots));
  let phys = Art.program published.Q.artifact in
  let run ?fuel:_ memory binding ~invocation:_ =
    match
      run_once ?on_run ?on_elf ?route_mutation ?budget ?label ?expect_cycles c
        image ~published memory binding
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

(* [on_run] sees what every run measured; [on_elf] the ELF each run loads,
   before the simulator starts (the run's own [guest.elf] is removed after it,
   so this is the supported copy for a disassembler; a trace needs no hook:
   [--trace_file=PATH] in {!Config.flags} reaches the RTL simulator);
   [on_image] each image's placement, when it is built (before any run): every
   table slot's Machine IR region and the role of the memory it landed in. [home] places each slot (default
   [Any]); [budget] bounds the subsystem's read-back bytes (default
   {!read_back_budget}); [label] names each run in the simulator's work
   directory and [expect_cycles] limits an RTL run to twice that (see
   {!Coral_run.run}); [itcm] caps the instruction memory the image may use,
   as on a smaller core (the subsystem then loads the kernel's overflow as
   overlays); [dma] says whether the functions may use the DMA engine
   ({!Dma}; default [Forbidden]), where [transfer] is a rule of one's own,
   which excludes it. *)
let route ?mutation ?kernel_mutation ?route_mutation ?registers ?engine
    ?engine_mutation ?transfer ?dma ?itcm ?on_run ?on_elf ?on_image ?home
    ?budget ?label ?expect_cycles (c : Config.t) =
  Route.Route.Custom
    {
      Route.Custom.name = "riscv32 coral " ^ c.Config.configuration;
      exec =
        (fun ~sites g ->
          exec_of ?mutation ?kernel_mutation ?route_mutation ?registers ?engine
            ?engine_mutation ?transfer ?dma ?itcm ?on_run ?on_elf ?on_image
            ?home ?budget ?label ?expect_cycles c ~sites g);
    }
