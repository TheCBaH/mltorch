(* Layouts on the fragment's two profiles, whose TCMs are the default
   CoreMini's (8 KiB ITCM, 32 KiB DTCM), and on a copy of the RTL one with the
   highmem TCMs (1 MiB each). *)

let profile name =
  let text = In_channel.with_open_bin "fragment.json" In_channel.input_all in
  match
    Err.payload
      (Err.bind (Coral_manifest.of_string text) (fun m ->
           Coral_profile.of_manifest m name))
  with
  | Ok p -> p
  | Error e -> Fmt.failwith "%a" Coral_profile.pp_error e

let rtl = profile "tiny-rtl"
let mpact = profile "tiny-mpact"

let highmem =
  {
    rtl with
    configuration_id = "coralnpu:tiny-highmem:x";
    regions =
      List.map
        (fun (r : Coral_profile.Region.t) ->
          match r.role with
          | Itcm -> { r with size = 0x100000L }
          | Dtcm -> { r with base = 0x100000L; size = 0x100000L }
          | Csr | Mmio | Ram -> r)
        rtl.regions;
  }

module L = Coral_layout

let code size = { L.Segment.align = 4L; executable = true; name = "text"; size }

let data size =
  { L.Segment.align = 8L; executable = false; name = "data"; size }

let slot placement size = { L.Slot.align = 8L; placement; size }

let show p segments slots =
  match Err.payload (L.place p segments slots) with
  | Ok l -> Fmt.pr "%a@." L.pp l
  | Error e -> Fmt.pr "error: %a@." L.pp_error e

let small = [ code 1200L; data 300L ]
let slots_small = L.Placement.[ slot In 64L; slot Out 64L; slot Scratch 128L ]

let%expect_test "a small image: every slot in DTCM" =
  show rtl small slots_small;
  [%expect
    {|
    digest 90bf9718
    text       0x00000000
    data       0x00010000
    htif       0x00010140
    packet     0x00010240
    staging    0x00010280
    trap       0x00011280
    stack top  0x00018000
    slot in    0x00011300 64 dtcm
    slot out   0x00011340 64 dtcm
    slot scratch 0x00011380 128 dtcm |}]

let%expect_test "a slot too large for DTCM goes to RAM, staged" =
  let slots = L.Placement.[ slot In 100_000L; slot Out 64L ] in
  show rtl small slots;
  show mpact small slots;
  show highmem small slots;
  [%expect
    {|
    digest e3bf944a
    text       0x00000000
    data       0x00010000
    htif       0x00010140
    packet     0x00010240
    staging    0x00010280
    trap       0x00011280
    stack top  0x00018000
    slot in    0x20000000 100000 testbench-ram staged
    slot out   0x00011300 64 dtcm
    digest 4c2dc642
    text       0x00000000
    data       0x00010000
    htif       0x00010140
    packet     0x00010240
    staging    0x00010280
    trap       0x00011280
    stack top  0x00018000
    slot in    0x20000000 100000 extmem staged
    slot out   0x00011300 64 dtcm
    digest c0749e18
    text       0x00000000
    data       0x00100000
    htif       0x00100140
    packet     0x00100240
    staging    0x00100280
    trap       0x00101280
    stack top  0x00200000
    slot in    0x00101300 100000 dtcm
    slot out   0x001199a0 64 dtcm |}]

let%expect_test "over capacity" =
  show rtl [ code 9000L ] slots_small;
  show rtl [ data 30_000L ] slots_small;
  show rtl small L.Placement.[ slot In 5_000_000L ];
  show
    {
      rtl with
      regions =
        List.filter
          (fun (r : Coral_profile.Region.t) -> r.role <> Ram)
          rtl.regions;
    }
    small
    L.Placement.[ slot In 100_000L ];
  [%expect
    {|
    error: itcm: 9000 bytes needed, 8192 available
    error: dtcm: 30000 bytes needed, 28672 available
    error: testbench-ram: 5000000 bytes needed, 4194304 available
    error: the profile has no usable ram region |}]

(* Equal assignments give equal digests; any moved address, or another
   profile, changes it. A size change that moves nothing leaves it. *)
let%expect_test "the digest follows the assignment" =
  let digest p segments slots =
    match Err.payload (L.place p segments slots) with
    | Ok l -> Fmt.str "%08lx" l.digest
    | Error e -> Fmt.str "%a" L.pp_error e
  in
  let base = digest rtl small slots_small in
  List.iter
    (fun (what, d) ->
      Fmt.pr "%-16s %s@." what (if d = base then "same" else "differs"))
    [
      ("again", digest rtl small slots_small);
      (* 300 and 308 bytes both end below the HTIF block's 64-byte boundary. *)
      ("data +8 bytes", digest rtl [ code 1200L; data 308L ] slots_small);
      ("data +64 bytes", digest rtl [ code 1200L; data 364L ] slots_small);
      ("slot order", digest rtl small (List.rev slots_small));
      ("other profile", digest mpact small slots_small);
      ( "staging 8 KiB",
        match
          Err.payload
            (L.place
               ~params:{ L.Params.default with staging = 8192L }
               rtl small slots_small)
        with
        | Ok l -> Fmt.str "%08lx" l.digest
        | Error _ -> "error" );
    ];
  [%expect
    {|
    again            same
    data +8 bytes    same
    data +64 bytes   differs
    slot order       differs
    other profile    differs
    staging 8 KiB    differs |}]
