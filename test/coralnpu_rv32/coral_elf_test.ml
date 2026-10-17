(* A module with code, data and a zero-filled tail, laid out for the
   fragment's profiles and written as ELF, then read back: one PT_LOAD per
   region, every byte in the file, the HTIF symbols where the layout put
   them. *)

open Machine_rivet_riscv32
open Rv32_asm

let profile name =
  let text =
    In_channel.with_open_bin "../coralnpu/fragment.json" In_channel.input_all
  in
  match
    Err.payload
      (Err.bind (Coral_manifest.of_string text) (fun m ->
           Coral_profile.of_manifest m name))
  with
  | Ok p -> p
  | Error e -> Fmt.failwith "%a" Coral_profile.pp_error e

let guest =
  match
    module_ "guest" (fun () ->
        [
          text;
          align 4;
          global "_start";
          label "_start";
          insn "j" [ sym "_start" ];
        ]
        @ [ data; align 8; label "table"; words [ 1; 2; 3 ] ]
        @ [ bss; align 16; label "scratch"; zero 16 ])
  with
  | Ok m -> m
  | Error what -> failwith what

let or_fail pp = function Ok v -> v | Error e -> Fmt.failwith "%a" pp e

(* The ELF for [guest] on [p], with [slots], or the refusal. *)
let build ?(move = Fun.id) p slots =
  let laid =
    or_fail Rv32_image.Error.pp
      (Err.payload (Rv32_image.plan ~entry:"_start" [ guest ]))
  in
  let segments =
    List.map
      (fun (s : Image.segment_plan) ->
        {
          Coral_layout.Segment.align = Int64.of_int s.alignment;
          executable = s.perms.execute;
          name = s.seg_name;
          size = Int64.of_int (s.init_size + s.zero_fill);
        })
      (Image.plan_of laid).segments
  in
  let l =
    or_fail Coral_layout.pp_error
      (Err.payload (Coral_layout.place p segments slots))
  in
  let l = move l in
  match Image.bind_image laid ~addresses:l.segments with
  | Error e -> Error (Foundation.Diag.render e)
  | Ok image ->
      Result.map_error
        (Fmt.str "%a" Coral_elf.pp_error)
        (Err.payload (Coral_elf.write p l image))

let dump elf =
  let d =
    or_fail Elf.Decode.pp_error (Err.payload (Elf.Decode.of_string elf))
  in
  Fmt.pr "entry 0x%Lx flags 0x%Lx@." d.header.e_entry d.header.e_flags;
  Array.iter
    (fun (h : Elf.Types.phdr) ->
      if h.p_type = 1L then
        Fmt.pr "LOAD vaddr 0x%08Lx filesz %Ld memsz %Ld flags %Ld@." h.p_vaddr
          h.p_filesz h.p_memsz h.p_flags)
    d.phdrs;
  match Elf.Decode.find_section d ".symtab" with
  | None -> Fmt.pr "no .symtab@."
  | Some s ->
      Array.iter
        (fun (y : Elf.Decode.symbol) ->
          if y.sym_name <> "" then
            Fmt.pr "%-15s 0x%08Lx size %Ld@." y.sym_name y.sym.st_value
              y.sym.st_size)
        (or_fail Elf.Decode.pp_error (Err.payload (Elf.Decode.symbols d s)))

let slots =
  Coral_layout.(Placement.[ { Slot.align = 8L; placement = In; size = 64L } ])

let%expect_test "an image per profile" =
  List.iter
    (fun name ->
      Fmt.pr "-- %s@." name;
      match build (profile name) slots with
      | Ok elf -> dump elf
      | Error e -> Fmt.pr "error: %s@." e)
    [ "tiny-mpact"; "tiny-rtl" ];
  [%expect
    {|
    -- tiny-mpact
    entry 0x0 flags 0x2
    LOAD vaddr 0x00000000 filesz 4 memsz 4 flags 5
    LOAD vaddr 0x00010000 filesz 320 memsz 320 flags 6
    tohost          0x00010040 size 8
    fromhost        0x00010080 size 8
    tohost_ready    0x000100c0 size 8
    fromhost_ready  0x00010100 size 8
    _start          0x00000000 size 0
    -- tiny-rtl
    entry 0x0 flags 0x2
    LOAD vaddr 0x00000000 filesz 4 memsz 4 flags 5
    LOAD vaddr 0x00010000 filesz 320 memsz 320 flags 6
    tohost          0x00010040 size 8
    fromhost        0x00010080 size 8
    tohost_ready    0x000100c0 size 8
    fromhost_ready  0x00010100 size 8
    _start          0x00000000 size 0 |}]

(* A layout that put data in RAM: refused on RTL, whose testbench cannot load
   it, and accepted on MPACT. *)
let%expect_test "a segment outside the loadable regions" =
  let to_ram (l : Coral_layout.t) =
    {
      l with
      segments =
        List.map
          (fun (n, a) ->
            if a >= 0x10000L then
              (n, Int64.add 0x20000000L (Int64.sub a 0x10000L))
            else (n, a))
          l.segments;
    }
  in
  List.iter
    (fun name ->
      match build ~move:to_ram (profile name) slots with
      | Ok _ -> Fmt.pr "%s: written@." name
      | Error e -> Fmt.pr "%s: %s@." name e)
    [ "tiny-mpact"; "tiny-rtl" ];
  [%expect
    {|
    tiny-mpact: written
    tiny-rtl: segment .data (16 bytes at 0x20000000) is outside every loadable region |}]
