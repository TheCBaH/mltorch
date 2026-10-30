(* The resident manager, without a simulator: its size does not depend on the
   addresses it is given (the image builder iterates a layout on that), and its
   [fence.i] is in the encoded bytes. The RTL cannot show the fence's omission,
   so this is what pins it. *)

open Machine_rivet_riscv32

let entries =
  Coral_overlay_manager.Entry.
    [
      { name = "ovl_a"; entry_offset = 0L };
      { name = "ovl_b"; entry_offset = 0x2468L };
    ]

let text ~state ~table =
  match Coral_overlay_manager.module_ ~slot_bytes:64L ~state ~table entries with
  | Error what -> failwith what
  | Ok m -> (
      match
        Err.payload
          (Rv32_image.plan ~profile:Rv32_profile.Rv32imf_zve32f_ilp32f
             ~entry:"ovl_a" [ m ])
      with
      | Error e -> Fmt.failwith "%a" Rv32_image.Error.pp e
      | Ok laid -> (
          match Image.bind_image laid ~addresses:[ (".text", 0L) ] with
          | Error e -> failwith (Rv32_image.render e)
          | Ok img ->
              (List.find
                 (fun (s : Image.segment) -> s.name = ".text")
                 img.segments)
                .bytes))

let words s =
  List.init (String.length s / 4) (fun k -> String.get_int32_le s (4 * k))

let%expect_test "the size follows no address" =
  let a = text ~state:0x7000_0000L ~table:0x7000_0010L
  and b = text ~state:0x0000_0800L ~table:0x8000_0FFCL in
  Fmt.pr "%d bytes; the same size at other addresses: %b@." (String.length a)
    (String.length a = String.length b);
  [%expect {| 304 bytes; the same size at other addresses: true |}]

let%expect_test "fence.i is encoded" =
  let w = words (text ~state:0x7000_0000L ~table:0x7000_0010L) in
  Fmt.pr "fence.i (0x0000100f) words: %d@."
    (List.length (List.filter (( = ) 0x0000100fl) w));
  [%expect {| fence.i (0x0000100f) words: 1 |}]
