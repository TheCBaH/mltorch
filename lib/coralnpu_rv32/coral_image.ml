open Err.Syntax
module Rv32_image = Machine_rivet_riscv32.Rv32_image

type t = {
  elf : string;
  image : Image.t;
  layout : Coral_layout.t;
  profile : Coral_profile.t;
}

type error =
  [ Coral_layout.error
  | Coral_elf.error
  | Rv32_image.Error.t
  | `Layout_unsettled
  | `Mailbox_full of int
  | `Runtime_form of string ]

let pp_error ppf : error -> unit = function
  | #Coral_layout.error as e -> Coral_layout.pp_error ppf e
  | #Coral_elf.error as e -> Coral_elf.pp_error ppf e
  | #Rv32_image.Error.t as e -> Rv32_image.Error.pp ppf e
  | `Layout_unsettled ->
      Fmt.string ppf "the layout did not settle within four rounds"
  | `Mailbox_full n ->
      Fmt.pf ppf "%d written slots, but the mailbox holds %d hashes" n
        Coral_mailbox.max_hashes
  | `Runtime_form what -> Fmt.pf ppf "the runtime has no form for %s" what

(* Every slot at address 0, read directly: the first round's guess. *)
let unplaced slots =
  {
    Coral_layout.digest = 0l;
    htif = 0L;
    packet = 0L;
    segments = [];
    slots =
      List.map
        (fun slot ->
          {
            Coral_layout.Placed_slot.address = 0L;
            htif_reachable = true;
            region = "";
            slot;
          })
        slots;
    stack_top = 0L;
    staging = 0L;
    staging_size = 0L;
    trap = 0L;
  }

let segments laid =
  List.map
    (fun (s : Image.segment_plan) ->
      {
        Coral_layout.Segment.align = Int64.of_int s.alignment;
        executable = s.perms.execute;
        name = s.seg_name;
        size = Int64.of_int (s.init_size + s.zero_fill);
      })
    (Image.plan_of laid).segments

let build ?params ?(abi = Machine_rivet_riscv32.Rv32_harness.Abi.Table) ?lowered
    ?rv32_profile ?matrix profile ~entry ~kernel slots =
  let vector =
    Option.fold ~none:false ~some:Machine_rivet_riscv32.Rv32_profile.vector
      rv32_profile
  in
  let mailbox = profile.Coral_profile.simulator = Subsystem in
  let written =
    List.length
      (List.filter
         (fun (s : Coral_layout.Slot.t) ->
           Coral_layout.Placement.written s.placement)
         slots)
  in
  let* () =
    if mailbox && written > Coral_mailbox.max_hashes then
      Err.fail (`Mailbox_full written)
    else Err.return ()
  in
  let plan l =
    let* runtime =
      Coral_runtime.module_ ~vector ?matrix ~mailbox l ~abi ~entry
      |> Result.map_error (fun what -> `Runtime_form what)
      |> Err.import ~pos:__POS__ Fun.id
    in
    Rv32_image.plan ?profile:rv32_profile ?lowered
      ~entry:Coral_runtime.entry_symbol (runtime :: kernel)
    |> Err.map_error ~pos:__POS__ (fun e ->
        (e : Rv32_image.Error.t :> [> error ]))
  in
  let rec settle l rounds =
    if rounds = 0 then Err.fail `Layout_unsettled
    else
      let* laid = plan l in
      let* next = Coral_layout.place ?params profile (segments laid) slots in
      if next = l then Err.return (laid, l) else settle next (rounds - 1)
  in
  let* laid, layout = settle (unplaced slots) 4 in
  let* image =
    match Image.bind_image laid ~addresses:layout.segments with
    | Ok image -> Err.return image
    | Error e -> Err.fail (`Rivet_bind (Foundation.Diag.render e))
  in
  let* elf = Coral_elf.write profile layout image in
  Err.return { elf; image; layout; profile }
