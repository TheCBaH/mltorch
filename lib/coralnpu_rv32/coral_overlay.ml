open Err.Syntax
module Rv32_image = Machine_rivet_riscv32.Rv32_image

module Overlay = struct
  type t = {
    name : string;
    entry : string;
    modules : Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_ list;
  }
end

module Linked = struct
  type t = {
    overlay : Overlay.t;
    text : string;
    data : string;
    bss : int64;
    bss_at : int64;
    entry_offset : int64;
  }
end

module Too_large = struct
  type t = { bytes : int64; capacity : int64 }
end

type error =
  [ Rv32_image.Error.t
  | `Overlay_data_too_large of Too_large.t
  | `Overlay_entry_missing of string
  | `Overlay_segment of string
  | `Overlay_text_too_large of Too_large.t ]

let pp_error ppf : error -> unit = function
  | #Rv32_image.Error.t as e -> Rv32_image.Error.pp ppf e
  | `Overlay_data_too_large { Too_large.bytes; capacity } ->
      Fmt.pf ppf "the overlay's data and bss take %Ld bytes, the region has %Ld"
        bytes capacity
  | `Overlay_entry_missing name ->
      Fmt.pf ppf "the overlay exports no entry symbol %s" name
  | `Overlay_segment name ->
      Fmt.pf ppf "the overlay has a segment %s, which has no home" name
  | `Overlay_text_too_large { Too_large.bytes; capacity } ->
      Fmt.pf ppf "the overlay's text takes %Ld bytes, the slot has %Ld" bytes
        capacity

let beat = 16L
let align_up n a = Int64.mul (Int64.div (Int64.add n (Int64.sub a 1L)) a) a
let size (s : Image.segment_plan) = Int64.of_int (s.init_size + s.zero_fill)

let pad bytes =
  let n = String.length bytes in
  let to_beat = Int64.to_int (align_up (Int64.of_int n) beat) - n in
  bytes ^ String.make to_beat '\000'

let link ?(imports = []) ~rv32_profile ~slot_base ~slot_size ~data_base
    ~data_size (overlay : Overlay.t) =
  let* laid =
    Rv32_image.plan ~profile:rv32_profile ~externs:(List.map fst imports)
      ~entry:overlay.entry overlay.modules
    |> Err.map_error ~pos:__POS__ (fun e ->
        (e : Rv32_image.Error.t :> [> error ]))
  in
  let segments = (Image.plan_of laid).segments in
  let find name =
    List.find_opt (fun (s : Image.segment_plan) -> s.seg_name = name) segments
  in
  let* () =
    match
      List.find_opt
        (fun (s : Image.segment_plan) ->
          not (List.mem s.seg_name [ ".bss"; ".data"; ".text" ]))
        segments
    with
    | Some s -> Err.fail (`Overlay_segment s.seg_name)
    | None -> Err.return ()
  in
  let text = Option.fold ~none:0L ~some:size (find ".text") in
  let data = Option.fold ~none:0L ~some:size (find ".data") in
  let* () =
    if text > slot_size then
      Err.fail
        (`Overlay_text_too_large
           { Too_large.bytes = text; capacity = slot_size })
    else Err.return ()
  in
  (* The bss follows the data at its own alignment. *)
  let bss_at =
    let alignment =
      Option.fold ~none:1L
        ~some:(fun (s : Image.segment_plan) -> Int64.of_int s.alignment)
        (find ".bss")
    in
    align_up (Int64.add data_base data) alignment
  in
  let bss = Option.fold ~none:0L ~some:size (find ".bss") in
  let used = Int64.sub (Int64.add bss_at bss) data_base in
  let* () =
    if used > data_size then
      Err.fail
        (`Overlay_data_too_large
           { Too_large.bytes = used; capacity = data_size })
    else Err.return ()
  in
  let* image =
    match
      Image.bind_image laid ~imports
        ~addresses:
          [ (".text", slot_base); (".data", data_base); (".bss", bss_at) ]
    with
    | Ok image -> Err.return image
    | Error e -> Err.fail (`Rivet_bind (Rv32_image.render e))
  in
  let bytes name =
    List.find_map
      (fun (s : Image.segment) ->
        if s.name = name then Some (s.bytes ^ String.make s.zero_fill '\000')
        else None)
      image.segments
    |> Option.value ~default:""
  in
  match List.assoc_opt overlay.entry image.exports with
  | None -> Err.fail (`Overlay_entry_missing overlay.entry)
  | Some entry ->
      Err.return
        {
          Linked.overlay;
          text = pad (bytes ".text");
          data = pad (bytes ".data");
          bss;
          bss_at;
          entry_offset = Int64.sub entry slot_base;
        }
