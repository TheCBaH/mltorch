open Err.Syntax
module Rv32_image = Machine_rivet_riscv32.Rv32_image

type t = {
  elf : string;
  image : Image.t;
  layout : Coral_layout.t;
  profile : Coral_profile.t;
  internal : (int * string) list;
}

type error =
  [ Coral_layout.error
  | Coral_dma.error
  | Coral_elf.error
  | Coral_overlay.error
  | Rv32_image.Error.t
  | `Layout_unsettled
  | `Mailbox_full of int
  | `Overlay_needs_dma
  | `Runtime_form of string ]

let pp_error ppf : error -> unit = function
  | #Coral_layout.error as e -> Coral_layout.pp_error ppf e
  | #Coral_dma.error as e -> Coral_dma.pp_error ppf e
  | #Coral_overlay.error as e -> Coral_overlay.pp_error ppf e
  | #Coral_elf.error as e -> Coral_elf.pp_error ppf e
  | `Overlay_needs_dma ->
      Fmt.string ppf
        "code overlays are loaded by the DMA, which only a subsystem profile \
         has"
  | `Layout_unsettled ->
      Fmt.string ppf "the layout did not settle within four rounds"
  | `Mailbox_full n ->
      Fmt.pf ppf "%d written slots, but the mailbox holds %d hashes" n
        Coral_mailbox.max_hashes
  | `Runtime_form what -> Fmt.pf ppf "the runtime has no form for %s" what

(* The subsystem's regions that signal a mailbox runtime's completion. *)
let uart_region = "uart1"
let clock_region = "clk_table"

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

(* {1 Code overlays}

   The overlays add three slots after the caller's: the data region the
   overlays share, the manager's state word, and a blob in SRAM holding each
   overlay's DMA chain, then the table the manager reads, then the images. Sizes
   come from linking every overlay once at address 0, which does not change them
   (text addresses its data through lui/addi pairs of fixed size); the final
   link, at the slot's and the region's real addresses, fills the blob. *)

module Overlays = struct
  type t = {
    probes : Coral_overlay.Linked.t list;
    resident : string list;
    slot_bytes : int64;
    region_bytes : int64;
    chains_bytes : int64;
    blob_bytes : int64;
  }

  let align_up n a = Int64.mul (Int64.div (Int64.add n (Int64.sub a 1L)) a) a
  let length s = Int64.of_int (String.length s)
  let big = 0x1000000L

  (* A chain copies the text, and the data when there is any. *)
  let descriptors = function
    | l when l.Coral_overlay.Linked.data = "" -> 1L
    | _ -> 2L

  (* [resident] is what an overlay may call or address: the exports of the
     image without the manager, so never a stub, which would load another
     overlay over the caller. Sizes do not depend on where imports are, so
     they are all at 0 here. *)
  let probe ~rv32_profile ~resident overlays =
    let imports = List.map (fun name -> (name, 0L)) resident in
    let* probes =
      Err.List.map
        (fun o ->
          Coral_overlay.link ~imports ~rv32_profile ~slot_base:0L ~slot_size:big
            ~data_base:big ~data_size:big o)
        overlays
    in
    let max_of f = List.fold_left (fun m l -> max m (f l)) 0L probes in
    let sum f = List.fold_left (fun n l -> Int64.add n (f l)) 0L probes in
    let chains_bytes =
      sum (fun l ->
          Int64.mul (descriptors l) (Int64.of_int Coral_dma.Descriptor.size))
    in
    let table_bytes =
      Int64.mul Coral_overlay_manager.Table.entry_bytes
        (Int64.of_int (List.length probes))
    in
    Err.return
      {
        probes;
        resident;
        slot_bytes = align_up (max_of (fun l -> length l.text)) 16L;
        region_bytes =
          align_up
            (max_of (fun l ->
                 Int64.add
                   (Int64.sub l.Coral_overlay.Linked.bss_at big)
                   (align_up l.bss 16L)))
            64L;
        chains_bytes;
        blob_bytes =
          Int64.add chains_bytes
            (Int64.add table_bytes
               (sum (fun l -> Int64.add (length l.text) (length l.data))));
      }

  let slots t =
    let open Coral_layout in
    [
      Slot.make ~home:Dtcm ~align:64L Placement.Scratch (max t.region_bytes 64L);
      Slot.make ~home:Dtcm ~align:16L Placement.Scratch 16L;
      Slot.make ~home:Ram ~align:32L Placement.In t.blob_bytes;
    ]

  let manager t (l : Coral_layout.t) ~first =
    let address k =
      (List.nth l.slots (first + k)).Coral_layout.Placed_slot.address
    in
    Coral_overlay_manager.module_ ~slot_bytes:t.slot_bytes ~state:(address 1)
      ~table:(Int64.add (address 2) t.chains_bytes)
      (List.map
         (fun (p : Coral_overlay.Linked.t) ->
           {
             Coral_overlay_manager.Entry.name = p.overlay.entry;
             entry_offset = p.entry_offset;
           })
         t.probes)

  (* The blob's bytes once the image is bound and the slots are placed. *)
  let blob t ~rv32_profile profile overlays (image : Image.t)
      (l : Coral_layout.t) ~first =
    let address k =
      (List.nth l.slots (first + k)).Coral_layout.Placed_slot.address
    in
    let region = address 0 and blob = address 2 in
    let slot_base =
      List.assoc Coral_overlay_manager.slot_symbol image.exports
    in
    let imports =
      List.filter (fun (name, _) -> List.mem name t.resident) image.exports
    in
    let* linked =
      Err.List.map
        (fun o ->
          Coral_overlay.link ~imports ~rv32_profile ~slot_base
            ~slot_size:t.slot_bytes ~data_base:region
            ~data_size:(max t.region_bytes 64L) o)
        overlays
    in
    let table_at = Int64.add blob t.chains_bytes in
    let images_at =
      Int64.add table_at
        (Int64.mul Coral_overlay_manager.Table.entry_bytes
           (Int64.of_int (List.length linked)))
    in
    let* _, chains, images, entries =
      Err.List.fold_left
        (fun (chain_at, chains, images, entries) (l : Coral_overlay.Linked.t) ->
          let image_at =
            Int64.add images_at (length (String.concat "" images))
          in
          let text_at = image_at in
          let data_at = Int64.add image_at (length l.text) in
          let copies =
            Coral_dma.Descriptor.copy ~src:text_at ~dst:slot_base
              (length l.text)
            ::
            (if l.data = "" then []
             else
               [
                 Coral_dma.Descriptor.copy ~src:data_at ~dst:region
                   (length l.data);
               ])
          in
          let* chain = Coral_dma.chain profile ~at:chain_at copies in
          Err.return
            ( Int64.add chain_at (length chain),
              chain :: chains,
              images @ [ l.text ^ l.data ],
              {
                Coral_overlay_manager.Table.chain = chain_at;
                bss_at = l.bss_at;
                bss_len = align_up l.bss 16L;
              }
              :: entries ))
        (blob, [], [], []) linked
    in
    Err.return
      (String.concat "" (List.rev chains)
      ^ Coral_overlay_manager.Table.encode (List.rev entries)
      ^ String.concat "" images)
end

let build ?overlays ?params
    ?(abi = Machine_rivet_riscv32.Rv32_harness.Abi.Table) ?lowered ?rv32_profile
    ?matrix profile ~entry ~kernel slots =
  let vector =
    Option.fold ~none:false ~some:Machine_rivet_riscv32.Rv32_profile.vector
      rv32_profile
  in
  let mailbox = profile.Coral_profile.simulator = Subsystem in
  let region name =
    List.find_map
      (fun (r : Coral_profile.Region.t) ->
        if String.equal r.name name then Some r.base else None)
      profile.regions
  in
  let uart =
    match (mailbox, region uart_region, region clock_region) with
    | true, Some base, Some clock_table ->
        Some { Coral_runtime.Uart.base; clock_table }
    | _ -> None
  in
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
  let overlays = Option.value overlays ~default:[] in
  let first = List.length slots in
  let resident ?externs l manager =
    let* runtime =
      Coral_runtime.module_ ~vector ?matrix ~mailbox ?uart l ~abi ~entry
      |> Result.map_error (fun what -> `Runtime_form what)
      |> Err.import ~pos:__POS__ Fun.id
    in
    Rv32_image.plan ?profile:rv32_profile ?lowered ?externs
      ~entry:Coral_runtime.entry_symbol
      ((runtime :: kernel) @ manager)
    |> Err.map_error ~pos:__POS__ (fun e ->
        (e : Rv32_image.Error.t :> [> error ]))
  in
  let* ovl =
    if overlays = [] then Err.return None
    else if not mailbox then Err.fail `Overlay_needs_dma
    else
      let rv32_profile =
        Option.value rv32_profile
          ~default:Machine_rivet_riscv32.Rv32_profile.default
      in
      (* The names the overlays may import, from the image without them;
         its calls of the overlays reach the stubs only later. *)
      let* laid =
        resident
          ~externs:
            (List.map (fun (o : Coral_overlay.Overlay.t) -> o.entry) overlays)
          (unplaced slots) []
      in
      Err.map
        (fun o -> Some o)
        (Overlays.probe ~rv32_profile ~resident:(Image.plan_of laid).exports
           overlays)
  in
  let slots =
    slots @ Option.fold ~none:[] ~some:(fun o -> Overlays.slots o) ovl
  in
  let plan l =
    let* manager =
      match ovl with
      | None -> Err.return []
      | Some o ->
          Overlays.manager o l ~first
          |> Result.map_error (fun what -> `Runtime_form what)
          |> Err.import ~pos:__POS__ Fun.id
          |> Err.map (fun m -> [ m ])
    in
    resident l manager
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
  let* internal =
    match ovl with
    | None -> Err.return []
    | Some o ->
        let rv32_profile =
          Option.value rv32_profile
            ~default:Machine_rivet_riscv32.Rv32_profile.default
        in
        Err.map
          (fun blob -> [ (first + 2, blob) ])
          (Overlays.blob o ~rv32_profile profile overlays image layout ~first)
  in
  let* elf = Coral_elf.write profile layout image in
  Err.return { elf; image; layout; profile; internal }
