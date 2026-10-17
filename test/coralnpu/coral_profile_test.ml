let fragment = In_channel.with_open_bin "fragment.json" In_channel.input_all

let show pp = function
  | Ok v -> Fmt.pr "%a@." pp v
  | Error e -> Fmt.pr "error: %a@." Coral_profile.pp_error e

let manifest text = Err.payload (Coral_manifest.of_string text)

(* [fragment] with its first [from] replaced by [into]. *)
let edit ~from ~into =
  let i =
    let n = String.length from in
    let rec find i =
      if String.sub fragment i n = from then i else find (i + 1)
    in
    find 0
  in
  String.sub fragment 0 i ^ into
  ^ String.sub fragment
      (i + String.length from)
      (String.length fragment - i - String.length from)

let profile text name =
  Result.bind (manifest text) (fun m ->
      Err.payload (Coral_profile.of_manifest m name))

let%expect_test "the fragment's IDs recompute, and each profile" =
  show Coral_profile.pp (profile fragment "tiny-mpact");
  show Coral_profile.pp (profile fragment "tiny-rtl");
  [%expect
    {|
    tiny-mpact mpact coralnpu:tiny-mpact:ba3bab005bb827732a7a
    f,i,m,zicsr
    itcm           itcm 0x00000000     8192 loadable
    dtcm           dtcm 0x00010000    32768 writable loadable htif
    extmem         ram  0x20000000  4194304 writable loadable
    tiny-rtl rtl coralnpu:tiny-rtl:3b3475e7a285b4c6eaaf
    f,i,m
    itcm           itcm 0x00000000     8192 loadable
    dtcm           dtcm 0x00010000    32768 writable loadable htif
    core-csr       csr  0x00030000     4096
    testbench-ram  ram  0x20000000  4194304 writable
    uart           mmio 0x54000000        4 writable |}]

(* The record without its [id], exactly as the producer's [json.dumps] wrote
   it when the fragment's ID was computed. *)
let%expect_test "canonical JSON: sorted code points, Python's escapes" =
  let json = Jsont_bytesrw.decode_string Jsont.json fragment |> Result.get_ok in
  let members = function
    | Jsont.Object (mems, _) -> List.map (fun ((n, _), v) -> (n, v)) mems
    | _ -> []
  in
  let body =
    match List.assoc_opt "configurations" (members json) with
    | Some (Array (Object (mems, meta) :: _, _)) ->
        Jsont.Object (List.filter (fun ((n, _), _) -> n <> "id") mems, meta)
    | _ -> assert false
  in
  (match Err.payload (Coral_json.to_string body) with
  | Ok s -> print_endline s
  | Error e -> Fmt.pr "%a@." Coral_json.pp_error e);
  [%expect {| {"description":"MPACT — \"quoted\", back\\slash, tab\t, \u0001 control, café","isa":{"Zeta":[],"extensions":["f","i","m","zicsr"],"xlen":32,"é":-7},"memory_regions":[{"access":{"dma":false,"guest_load":true,"guest_store":false,"host_load":true,"instruction_fetch":true},"base_address":0,"name":"itcm","size_bytes":8192,"type":"itcm"},{"access":{"dma":false,"guest_load":true,"guest_store":true,"host_load":true,"instruction_fetch":false},"base_address":65536,"name":"dtcm","size_bytes":32768,"type":"dtcm"},{"access":{"dma":false,"guest_load":true,"guest_store":true,"host_load":true,"instruction_fetch":false},"base_address":536870912,"name":"extmem","size_bytes":4194304,"type":"ram"}],"name":"tiny-mpact","simulation":{"backend":"mpact","clock":{"configured_frequency_mhz":null},"interface":"standalone-cli"}} |}]

(* One byte of a record changes its ID, so the declared one no longer holds. *)
let%expect_test "a changed record is refused" =
  show Coral_profile.pp
    (profile
       (edit ~from:"\"size_bytes\": 8192" ~into:"\"size_bytes\": 8193")
       "tiny-mpact");
  show Coral_profile.pp (profile (edit ~from:"café" ~into:"cafe") "tiny-mpact");
  [%expect
    {|
    error: configuration coralnpu:tiny-mpact:ba3bab005bb827732a7a: its record hashes to coralnpu:tiny-mpact:e0c6762b75c825bff935
    error: configuration coralnpu:tiny-mpact:ba3bab005bb827732a7a: its record hashes to coralnpu:tiny-mpact:5e66e4a4ca2ed2bd3b94 |}]

let%expect_test "numbers and members the canonical form cannot reproduce" =
  let canonical s =
    match Jsont_bytesrw.decode_string Jsont.json s with
    | Error e -> print_endline e
    | Ok j -> (
        match Err.payload (Coral_json.to_string j) with
        | Ok s -> print_endline s
        | Error e -> Fmt.pr "error: %a@." Coral_json.pp_error e)
  in
  canonical {|{"b":[1,-2,0,-0,4294967295],"a":null,"c":true}|};
  canonical {|{"x":1.5}|};
  canonical {|{"x":9007199254740992}|};
  canonical {|{"x":1,"x":2}|};
  [%expect
    {|
    {"a":null,"b":[1,-2,0,0,4294967295],"c":true}
    error: number 0x1.8p+0 is not an exact integer
    error: number 0x1p+53 is not an exact integer
    error: member "x" appears twice |}]

let%expect_test "refusals" =
  show Coral_profile.pp (profile fragment "tiny-absent");
  show Coral_profile.pp
    (profile
       (edit ~from:"\"schema_version\": \"1.0\""
          ~into:"\"schema_version\": \"2.0\"")
       "tiny-rtl");
  (* A backend or region type outside the known sets, past the ID check. *)
  let m = Result.get_ok (manifest fragment) in
  let edited f =
    { m with configurations = List.map f m.Coral_manifest.configurations }
  in
  show Coral_profile.pp
    (Err.payload
       (Coral_profile.of_manifest
          (edited (fun c -> { c with backend = "gem5" }))
          "tiny-rtl"));
  show Coral_profile.pp
    (Err.payload
       (Coral_profile.of_manifest
          (edited (fun c ->
               {
                 c with
                 regions =
                   List.map
                     (fun (r : Coral_manifest.Region.t) ->
                       { r with kind = "sram" })
                     c.regions;
               }))
          "tiny-rtl"));
  show Coral_profile.pp
    (Err.payload
       (Coral_profile.of_manifest { m with artifacts = [] } "tiny-rtl"));
  [%expect {|
    error: no configuration named "tiny-absent"
    error: unsupported manifest schema_version "2.0"
    error: unknown simulation backend "gem5"
    error: unknown memory region type "sram"
    error: no artifact simulates coralnpu:tiny-rtl:3b3475e7a285b4c6eaaf |}]

(* [s] with every [dir] spelled [$DIR], so output does not name a temporary
   directory. *)
let scrub dir s =
  let n = String.length dir in
  let b = Buffer.create (String.length s) in
  let rec go i =
    if i >= String.length s then ()
    else if i + n <= String.length s && String.sub s i n = dir then (
      Buffer.add_string b "$DIR";
      go (i + n))
    else (
      Buffer.add_char b s.[i];
      go (i + 1))
  in
  go 0;
  Buffer.contents b

let with_archive f =
  let dir = Filename.temp_file "coral_" "" in
  Sys.remove dir;
  Sys.mkdir dir 0o755;
  let bin = Filename.concat dir "bin" in
  Sys.mkdir bin 0o755;
  let write name text =
    Out_channel.with_open_bin (Filename.concat bin name) (fun oc ->
        output_string oc text)
  in
  write "mpact" "mpact\n";
  write "rtl" "rtl\n";
  Out_channel.with_open_bin (Filename.concat dir Coral_profile.manifest_file)
    (fun oc -> output_string oc fragment);
  Fun.protect
    ~finally:(fun () ->
      List.iter
        (fun f -> try Sys.remove f with Sys_error _ -> ())
        [
          Filename.concat bin "mpact";
          Filename.concat bin "rtl";
          Filename.concat dir Coral_profile.manifest_file;
        ];
      (try Sys.rmdir bin with Sys_error _ -> ());
      try Sys.rmdir dir with Sys_error _ -> ())
    (fun () -> f ~dir ~write)

let%expect_test "artifacts are verified, and the launch command filled" =
  with_archive (fun ~dir ~write ->
      let load name =
        match Err.payload (Coral_profile.load ~dir name) with
        | Ok p ->
            Fmt.pr "%s: %s@." name
              (scrub dir
                 (String.concat " "
                    (Coral_profile.argv ~dir ~elf:"guest.elf" p)))
        | Error e ->
            Fmt.pr "%s: error: %s@." name
              (scrub dir (Fmt.str "%a" Coral_profile.pp_error e))
      in
      load "tiny-mpact";
      load "tiny-rtl";
      write "rtl" "rtl!\n";
      load "tiny-rtl";
      Sys.remove (Filename.concat dir "bin/rtl");
      load "tiny-rtl");
  [%expect
    {|
    tiny-mpact: $DIR/bin/mpact --semihost_htif --exit_on_ebreak guest.elf
    tiny-rtl: $DIR/bin/rtl --binary=guest.elf
    tiny-rtl: error: $DIR/bin/rtl: sha256 is 392b80b122693bd8908315e46f5a81d94847cfaaf1733d1a5c68332f8d76e077, the manifest pins 50d2f6599ee68748044579342c58e92b6f7232c1784821eb9566c2f7991a98aa
    tiny-rtl: error: $DIR/bin/rtl: No such file or directory |}]
