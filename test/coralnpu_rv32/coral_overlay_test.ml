(* Overlays linked on their own for a slot and a data region: the sizes, the
   entry, the refusals, and the link depending on where the slot is. *)

open Machine_rivet_riscv32
open Rv32_asm

let or_fail what = function
  | Ok v -> v
  | Error e -> Fmt.failwith "%s: %s" what e

let overlay ?(extra = []) name =
  let m =
    or_fail name
      (module_ name (fun () ->
           [
             text;
             align 4;
             global name;
             label name;
             insn "la" [ reg "t0"; sym "table" ];
             insn "lw" [ reg "a0"; mem "t0" 0 ];
             insn "ret" [];
           ]
           @ extra
           @ [ data; align 4; label "table"; words [ 7 ] ]
           @ [ bss; align 16; label "scratch"; zero 24 ]))
  in
  { Coral_overlay.Overlay.name; entry = name; modules = [ m ] }

let link ?imports ?(slot_base = 0x1000L) ?(slot_size = 64L)
    ?(data_base = 0x2000L) ?(data_size = 64L) o =
  Err.payload
    (Coral_overlay.link ?imports
       ~rv32_profile:Rv32_profile.Rv32imf_zve32f_ilp32f ~slot_base ~slot_size
       ~data_base ~data_size o)

let show = function
  | Ok (l : Coral_overlay.Linked.t) ->
      Fmt.pr "text %d, data %d, bss %Ld, entry at +%Ld@." (String.length l.text)
        (String.length l.data) l.bss l.entry_offset
  | Error e -> Fmt.pr "refused: %a@." Coral_overlay.pp_error e

(* Text and data pad to the DMA's 16-byte beat; the bss is the 24 requested. *)
let%expect_test "an overlay's sizes and entry" =
  show (link (overlay "ovl"));
  [%expect {| text 16, data 16, bss 24, entry at +0 |}]

(* [la] is absolute, but it names the data: the text follows the data region's
   address, and the slot's only through branches, which are relative. *)
let%expect_test "what the text depends on" =
  let text ?slot_base ?data_base () =
    match link ?slot_base ?data_base (overlay "ovl") with
    | Ok l -> l.Coral_overlay.Linked.text
    | Error _ -> assert false
  in
  Fmt.pr "same text at another slot: %b@." (text () = text ~slot_base:0x1040L ());
  Fmt.pr "same text at another data region: %b@."
    (text () = text ~data_base:0x2040L ());
  [%expect
    {|
    same text at another slot: true
    same text at another data region: false |}]

let%expect_test "an overlay larger than its room" =
  show (link ~slot_size:8L (overlay "ovl"));
  show (link ~data_size:32L (overlay "ovl"));
  [%expect
    {|
    refused: the overlay's text takes 16 bytes, the slot has 8
    refused: the overlay's data and bss take 40 bytes, the region has 32 |}]

let%expect_test "a reference outside the overlay" =
  show
    (link
       (overlay ~extra:[ insn "jal" [ reg "ra"; sym "resident_helper" ] ] "ovl"));
  [%expect
    {| refused: <synthesized by riscv32.encode>: error[image.undefined]: fixup target references undefined symbol resident_helper |}]

let%expect_test "a call into the resident image through an import" =
  let o =
    overlay ~extra:[ insn "jal" [ reg "ra"; sym "resident_helper" ] ] "ovl"
  in
  (match link ~imports:[ ("resident_helper", 0x400L) ] o with
  | Ok l ->
      let w = String.get_int32_le l.Coral_overlay.Linked.text 16 in
      Fmt.pr "jal at +16: %08lx@." w
  | Error e -> Fmt.pr "refused: %a@." Coral_overlay.pp_error e);
  show (link ~imports:[ ("other", 0L) ] o);
  [%expect
    {|
    jal at +16: bf0ff0ef
    refused: <synthesized by riscv32.encode>: error[image.undefined]: fixup target references undefined symbol resident_helper |}]
