(* The five ordered archive checkpoints (Pt2_zip, .ai/model_explorer_design.md).

   Every fixture here is a HAND-BUILT archive, byte by byte, because zipc's
   writer cannot produce any of the things being tested: it will not emit two
   members with one path, an encrypted entry, a compression method it does not
   support, or a header whose declared sizes disagree with the bytes beside
   them. A test that could only build well-formed archives would be testing the
   writer.

   Each case pins the EXACT error rather than "it failed", because for four of
   them the whole point is WHICH checkpoint caught it — an aggregate that is
   detected at materialization instead of in the central directory, or a stored
   size mismatch reported as a CRC failure, means the ordering has silently
   changed while every "rejects" assertion stayed green. *)

let le16 b v =
  Buffer.add_char b (Char.chr (v land 0xff));
  Buffer.add_char b (Char.chr ((v lsr 8) land 0xff))

let le32 b v =
  le16 b (v land 0xffff);
  le16 b ((v lsr 16) land 0xffff)

let le32_i32 b (v : int32) = le32 b (Int32.to_int (Int32.logand v 0xffffffffl))

module Entry = struct
  (* Every header field a fixture might want to lie about. [data] is what is
     actually written between the headers; [compressed] and [decompressed] are
     what the headers CLAIM, which is the whole subject of checkpoints 4 and
     5. *)
  type t = {
    path : string;
    data : string;
    method_ : int;
    gp_flags : int;
    crc : int32;
    compressed : int;
    decompressed : int;
  }

  (* A consistent STORED entry: the archive zipc itself would write. *)
  let stored path data =
    {
      path;
      data;
      method_ = 0;
      gp_flags = 0;
      crc = Zipc_deflate.Crc_32.string data;
      compressed = String.length data;
      decompressed = String.length data;
    }

  let deflated path contents =
    match
      Zipc_deflate.crc_32_and_deflate ~start:0 ~len:(String.length contents)
        contents
    with
    | Error e -> failwith ("deflate: " ^ e)
    | Ok (crc, data) ->
        {
          path;
          data;
          method_ = 8;
          gp_flags = 0;
          crc;
          compressed = String.length data;
          decompressed = String.length contents;
        }
end

(* [declared_count] overrides the EOCD's total-entries field, which is what
   checkpoint 2 reads and checkpoint 3 compares against. *)
let make_archive ?declared_count (entries : Entry.t list) =
  let b = Buffer.create 512 in
  let offsets =
    List.map
      (fun (e : Entry.t) ->
        let offset = Buffer.length b in
        le32 b 0x04034b50;
        le16 b 20;
        le16 b e.gp_flags;
        le16 b e.method_;
        le16 b 0;
        le16 b 0x21;
        le32_i32 b e.crc;
        le32 b e.compressed;
        le32 b e.decompressed;
        le16 b (String.length e.path);
        le16 b 0;
        Buffer.add_string b e.path;
        Buffer.add_string b e.data;
        offset)
      entries
  in
  let cd_start = Buffer.length b in
  List.iter2
    (fun (e : Entry.t) offset ->
      le32 b 0x02014b50;
      le16 b 20;
      le16 b 20;
      le16 b e.gp_flags;
      le16 b e.method_;
      le16 b 0;
      le16 b 0x21;
      le32_i32 b e.crc;
      le32 b e.compressed;
      le32 b e.decompressed;
      le16 b (String.length e.path);
      le16 b 0;
      le16 b 0;
      le16 b 0;
      le16 b 0;
      le32 b 0;
      le32 b offset;
      Buffer.add_string b e.path)
    entries offsets;
  let cd_size = Buffer.length b - cd_start in
  let count =
    match declared_count with Some n -> n | None -> List.length entries
  in
  le32 b 0x06054b50;
  le16 b 0;
  le16 b 0;
  le16 b count;
  le16 b count;
  le32 b cd_size;
  le32 b cd_start;
  le16 b 0;
  Buffer.contents b

let show ?limits s =
  match Pt2_zip.of_string ?limits s with
  | Ok zip ->
      Printf.printf "opened, %d entries\n" (List.length (Pt2_zip.entries zip))
  | Error e -> Format.printf "%a@." Pt2_zip.pp_error e.Core.Error.kind

let read_show ?limits s name =
  match Pt2_zip.of_string ?limits s with
  | Error e -> Format.printf "open: %a@." Pt2_zip.pp_error e.Core.Error.kind
  | Ok zip -> (
      match Pt2_zip.read zip name with
      | Ok (Some data) -> Printf.printf "read %d bytes\n" (String.length data)
      | Ok None -> print_endline "no such entry"
      | Error e -> Format.printf "read: %a@." Pt2_zip.pp_error e.Core.Error.kind
      )

let tight ~max_entries ~max_entry_bytes ~max_total_bytes =
  Core.or_raise Pt2_zip.Limits.pp_error
    (Pt2_zip.Limits.create ~max_entries ~max_entry_bytes ~max_total_bytes
       ~max_path_bytes:1024 ~max_path_depth:16)

(* --- the well-formed baseline, so every rejection below is a difference --- *)

let%expect_test "a well-formed archive opens and reads" =
  let a = make_archive [ Entry.stored "m/models/model.json" "{\"k\":1}" ] in
  show a;
  read_show a "m/models/model.json";
  [%expect {|
    opened, 1 entries
    read 7 bytes
    |}]

let%expect_test "a deflated entry round-trips through the produced-length check"
    =
  let contents = String.concat "" (List.init 200 (fun _ -> "abcdefgh")) in
  let a = make_archive [ Entry.deflated "m/data/0" contents ] in
  read_show a "m/data/0";
  [%expect {| read 1600 bytes |}]

(* --- checkpoint 2: the raw EOCD count, before zipc decodes anything --- *)

let%expect_test "checkpoint 2: declared entry count over the limit" =
  let a =
    make_archive
      (List.init 4 (fun i -> Entry.stored (Printf.sprintf "m/data/%d" i) "x"))
  in
  show
    ~limits:
      (tight ~max_entries:3 ~max_entry_bytes:4096L ~max_total_bytes:65536L)
    a;
  [%expect {| zip declares more than 3 entries |}]

let%expect_test "checkpoint 2: no end-of-central-directory record" =
  show "not a zip at all";
  [%expect {| zip has no end-of-central-directory record |}]

(* --- checkpoint 3: the duplicate a path-keyed count cannot see --- *)

let%expect_test "checkpoint 3: two entries, one path, count below the limit" =
  (* Both members are well formed and the archive declares two, which is under
     any sane [max_entries] — so checkpoint 2 passes. zipc keys members by path,
     so the second overwrites the first and the decoded count is 1. Comparing
     the two counts is the ONLY place this is visible. *)
  let a =
    make_archive
      [ Entry.stored "m/data/0" "first"; Entry.stored "m/data/0" "second" ]
  in
  show a;
  [%expect {| zip declares 2 entries but has fewer distinct paths |}]

(* --- checkpoint 4: paths, encryption, method, sizes, aggregate --- *)

let%expect_test "checkpoint 4: a traversal path" =
  show (make_archive [ Entry.stored "m/../../etc/passwd" "x" ]);
  [%expect {| zip path "m/../../etc/passwd" has a ".." component |}]

let%expect_test "checkpoint 4: an absolute path" =
  show (make_archive [ Entry.stored "/etc/passwd" "x" ]);
  [%expect {| zip path "/etc/passwd" is absolute |}]

let%expect_test "checkpoint 4: a control byte in a path" =
  show (make_archive [ Entry.stored "m/data\x00/0" "x" ]);
  [%expect {| zip path "m/data\000/0" has a control byte |}]

let%expect_test "checkpoint 4: an over-deep path" =
  let deep = "m/" ^ String.concat "/" (List.init 20 (fun _ -> "d")) ^ "/f" in
  show (make_archive [ Entry.stored deep "x" ]);
  [%expect
    {| zip path "m/d/d/d/d/d/d/d/d/d/d/d/d/d/d/d/d/d/d/d/d/f" is deeper than 16 components |}]

let%expect_test "checkpoint 4: an encrypted member" =
  let e = Entry.stored "m/data/0" "x" in
  show (make_archive [ { e with Entry.gp_flags = 1 } ]);
  [%expect {| zip entry "m/data/0" is encrypted |}]

let%expect_test "checkpoint 4: a method that is neither stored nor deflate" =
  let e = Entry.stored "m/data/0" "x" in
  show (make_archive [ { e with Entry.method_ = 93 } ]);
  [%expect {| zip entry "m/data/0" uses an unsupported compression method |}]

let%expect_test "checkpoint 4: a bomb, rejected on the DECLARED size" =
  (* Ten bytes on disk claiming to inflate to 512MB. Nothing decompresses: the
     central directory says so, and that is checked before any inflate runs. *)
  let e = Entry.deflated "m/data/0" "0123456789" in
  show
    ~limits:
      (tight ~max_entries:16 ~max_entry_bytes:1_048_576L
         ~max_total_bytes:0x8000_0000L)
    (make_archive [ { e with Entry.decompressed = 512 * 1024 * 1024 } ]);
  [%expect
    {| zip entry "m/data/0" declares decompressed size over 1048576 bytes |}]

let%expect_test "checkpoint 4: an aggregate of entries that each pass" =
  (* Four honest 300-byte entries against a 1000-byte total and a 400-byte
     per-entry ceiling: every one is individually legal and the sum is not.
     Bounding each separately does not bound the sum.

     The bytes are really there rather than merely declared, because zipc
     validates at decode that each compressed span fits inside the archive
     (zipc.ml:335) — so a fixture that only LIED about the sizes would be
     rejected as a corrupted local header and would never reach checkpoint 4.
     Finding that out is itself worth recording: the aggregate a caller can
     actually present is an aggregate of real bytes. *)
  let a =
    make_archive
      (List.init 4 (fun i ->
           Entry.stored (Printf.sprintf "m/data/%d" i) (String.make 300 'x')))
  in
  show
    ~limits:(tight ~max_entries:16 ~max_entry_bytes:400L ~max_total_bytes:1000L)
    a;
  [%expect {| zip entries total over 1000 bytes |}]

(* --- checkpoint 5: at materialization, per method --- *)

let%expect_test "checkpoint 5: a stored member whose two sizes disagree" =
  (* The archive holds five bytes and the headers claim it decompresses to
     three. Checkpoint 4 lets it through — both figures are under the ceiling.

     The golden is the PRE-materialization message. Disabling the pre-check
     leaves the post-check catching the same archive with a different message
     ("decompressed size disagrees"), which is what makes this test non-vacuous
     and what it is really asserting: not that the mismatch is caught, but that
     it is caught before the string is built. *)
  let e = Entry.stored "m/data/0" "12345" in
  read_show (make_archive [ { e with Entry.decompressed = 3 } ]) "m/data/0";
  [%expect {| read: zip entry "m/data/0" is stored with disagreeing sizes |}]

(* --- checkpoint 1: the file is bounded before the string exists --- *)

let%expect_test "checkpoint 1: an oversized file is rejected before input_all" =
  let path = Filename.temp_file "pt2_zip_limits" ".pt2" in
  Out_channel.with_open_bin path (fun oc ->
      Out_channel.output_string oc (String.make 4096 'x'));
  (* The path is a temp name, so the payload is reported structurally rather
     than through [pp_error] — a golden holding /tmp/pt2_zip_limitsXXXXXX would
     differ on every run and on every backend. *)
  (match Pt2_archive.open_pt2 ~max_bytes:1024L path with
  | Ok _ -> print_endline "unexpectedly opened"
  | Error { Core.Error.kind = `Io_too_large (_, limit); _ } ->
      Printf.printf "rejected before reading, limit=%Ld\n" limit
  | Error e -> Format.printf "%a@." Pt2_archive.pp_error e.Core.Error.kind);
  Sys.remove path;
  [%expect {| rejected before reading, limit=1024 |}]

let%expect_test "checkpoint 1: a caller cannot raise the bound past the ceiling"
    =
  (* The length is narrowed to the [int] [really_input_string] takes, and that
     [int] is 32 bits under js_of_ocaml — so [~max_bytes] is CLAMPED rather than
     trusted: a bound the caller sets above the domain is not a bound.

     Asserted on the rule itself. Driving it through [open_pt2] would need a
     512MB fixture to reach the clamp at all, and the cheap version of that test
     — a small file, asking for 4GB — takes the same branch whether the clamp
     exists or not. *)
  List.iter
    (fun request ->
      Printf.printf "%Ld -> %Ld\n" request
        (Pt2_archive.effective_max_bytes request))
    [ 1024L; 0x2000_0000L; 0x2000_0001L; 0x1_0000_0000L; Int64.max_int ];
  [%expect
    {|
    1024 -> 1024
    536870912 -> 536870912
    536870913 -> 536870912
    4294967296 -> 536870912
    9223372036854775807 -> 536870912
    |}]

(* --- the profile itself --- *)

let%expect_test "limits may tighten, never widen" =
  let show_create r =
    match r with
    | Ok _ -> print_endline "ok"
    | Error e -> Format.printf "%a@." Pt2_zip.Limits.pp_error e.Core.Error.kind
  in
  show_create
    (Pt2_zip.Limits.create ~max_entries:16 ~max_entry_bytes:1024L
       ~max_total_bytes:4096L ~max_path_bytes:256 ~max_path_depth:8);
  show_create
    (Pt2_zip.Limits.create ~max_entries:0x10001 ~max_entry_bytes:1024L
       ~max_total_bytes:4096L ~max_path_bytes:256 ~max_path_depth:8);
  show_create
    (Pt2_zip.Limits.create ~max_entries:16 ~max_entry_bytes:0x8000_0000L
       ~max_total_bytes:4096L ~max_path_bytes:256 ~max_path_depth:8);
  show_create
    (Pt2_zip.Limits.create ~max_entries:16 ~max_entry_bytes:1024L
       ~max_total_bytes:0L ~max_path_bytes:256 ~max_path_depth:8);
  [%expect
    {|
    ok
    invalid zip limit max_entries = 65537
    invalid zip limit max_entry_bytes = 2147483648
    invalid zip limit max_total_bytes = 0
    |}]
