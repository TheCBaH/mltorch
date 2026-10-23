(* DMA descriptors as the engine reads them, and the refusals, on the
   fragment's RTL profile with its TCMs and RAM marked reachable by DMA, as the
   subsystem's are. *)

module D = Coral_dma

let profile =
  let text = In_channel.with_open_bin "fragment.json" In_channel.input_all in
  match
    Err.payload
      (Err.bind (Coral_manifest.of_string text) (fun m ->
           Coral_profile.of_manifest m "tiny-rtl"))
  with
  | Ok p ->
      {
        p with
        regions =
          List.map
            (fun (r : Coral_profile.Region.t) ->
              match r.role with
              | Dtcm | Itcm | Ram -> { r with dma_reachable = true }
              | Csr | Ddr | Mmio | Rom -> r)
            p.regions;
      }
  | Error e -> Fmt.failwith "%a" Coral_profile.pp_error e

let words s =
  List.init (String.length s / 4) (fun k -> String.get_int32_le s (4 * k))

let show r =
  match Err.payload r with
  | Ok s ->
      Fmt.pr "%a@." Fmt.(list ~sep:sp (fun ppf -> pf ppf "%08lx")) (words s)
  | Error e -> Fmt.pr "error: %a@." D.pp_error e

let%expect_test "len_flags" =
  List.iter
    (fun (d : D.Descriptor.t) -> Fmt.pr "%08lx@." (D.len_flags d))
    [
      D.Descriptor.copy ~src:0L ~dst:0L 64L;
      D.Descriptor.copy ~beat:B4 ~src:0L ~dst:0L 0xFF_FFFCL;
      { (D.Descriptor.copy ~beat:B1 ~src:0L ~dst:0L 3L) with dst_fixed = true };
      { (D.Descriptor.copy ~beat:B2 ~src:0L ~dst:0L 2L) with src_fixed = true };
    ];
  [%expect {|
    04000040
    02fffffc
    10000003
    09000002 |}]

let%expect_test "a chain of two, SRAM to DTCM and back" =
  show
    (D.chain profile ~at:0x2000_1000L
       [
         D.Descriptor.copy ~src:0x2000_0000L ~dst:0x0001_0000L 4096L;
         D.Descriptor.copy ~src:0x0001_0000L ~dst:0x2000_2000L 4096L;
       ]);
  [%expect
    {|
    20000000 00010000 04001000 20001020 00000000 00000000 00000000 00000000
    00010000 20002000 04001000 00000000 00000000 00000000 00000000
    00000000 |}]

let%expect_test "refusals" =
  let one d = show (D.chain profile ~at:0x2000_1000L [ d ]) in
  one (D.Descriptor.copy ~src:0x2000_0000L ~dst:0x0001_0000L 0L);
  one (D.Descriptor.copy ~src:0x2000_0000L ~dst:0x0001_0000L 0x100_0000L);
  one (D.Descriptor.copy ~src:0x2000_0000L ~dst:0x0001_0000L 24L);
  one (D.Descriptor.copy ~src:0x2000_0008L ~dst:0x0001_0000L 16L);
  one (D.Descriptor.copy ~beat:B4 ~src:0x2000_0004L ~dst:0x0001_0002L 16L);
  (* Past the end of DTCM, and into the CSR block. *)
  one (D.Descriptor.copy ~src:0x2000_0000L ~dst:0x0001_7FF0L 32L);
  one (D.Descriptor.copy ~src:0x2000_0000L ~dst:0x0003_0000L 16L);
  show (D.chain profile ~at:0x2000_1010L []);
  [%expect
    {|
    error: a DMA transfer of 0 bytes
    error: a DMA transfer of 16777216 bytes
    error: a DMA transfer of 24 bytes
    error: DMA address 0x20000008 is not 16-byte aligned
    error: DMA address 0x00010002 is not 4-byte aligned
    error: 32 bytes at 0x00017ff0 are outside every region the DMA reaches
    error: 16 bytes at 0x00030000 are outside every region the DMA reaches
    error: DMA address 0x20001010 is not 32-byte aligned |}]
