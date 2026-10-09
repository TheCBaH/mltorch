open Pt2_fixture_test.World
module U = Support.U

let run ?limits gz =
  match Err.payload (U.Targz.members ?limits gz) with
  | Ok ms ->
      List.iter
        (fun (m : U.Targz.member) ->
          Printf.printf "%s (%d)\n" m.name (String.length m.data))
        ms
  | Error e -> Support.show_error e

let%expect_test "a well-formed archive yields its regular files" =
  run (gzip (tar [ ("a.txt", "hello"); ("dir/b.txt", String.make 700 'x') ]));
  run (gzip (tar []));
  [%expect {|
    a.txt (5)
    dir/b.txt (700) |}]

let%expect_test "member hazards are refused, not interpreted" =
  let one ?typeflag ?linkname ?prefix name data =
    run
      (gzip
         (tar_member ?typeflag ?linkname ?prefix name data
         ^ String.make 1024 '\000'))
  in
  print_endline "-- names";
  one "/etc/passwd" "x";
  one "../escape" "x";
  one "a/../../escape" "x";
  one "a//b" "x";
  one "./a" "x";
  one "a\\b" "x";
  one ~prefix:(String.make 155 'p') (String.make 100 'n') "x";
  one ~prefix:"deep" "name" "x";
  print_endline "-- kinds";
  one ~typeflag:'2' ~linkname:"target" "link" "";
  one ~typeflag:'1' ~linkname:"target" "hard" "";
  one ~typeflag:'x' "pax" "path=a";
  one ~typeflag:'5' "dir/" "";
  one ~typeflag:'3' "dev" "";
  print_endline "-- duplicates";
  run (gzip (tar [ ("a", "1"); ("a", "2") ]));
  [%expect
    {|
    -- names
    absolute member name: "/etc/passwd"
    unsafe member name: "../escape"
    unsafe member name: "a/../../escape"
    unsafe member name: "a//b"
    unsafe member name: "./a"
    unsafe member name: "a\\b"
    member name too long: "ppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp/nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn"
    deep/name (1)
    -- kinds
    link member: "link"
    link member: "hard"
    unsupported member type: "pax"
    unsupported member type: "dir/"
    unsupported member type: "dev"
    -- duplicates
    duplicate member: "a" |}]

let%expect_test "framing damage" =
  let good = gzip (tar [ ("a.txt", "hello world") ]) in
  let flip i s =
    String.mapi
      (fun j c -> if i = j then Char.chr (Char.code c lxor 0xff) else c)
      s
  in
  let n = String.length good in
  print_endline "-- gzip";
  run (String.sub good 0 10);
  run (flip 0 good);
  run (flip 2 good);
  run (String.sub good 0 (n - 1));
  run (flip (n - 5) good);
  run (flip (n - 1) good);
  run (flip 30 good);
  print_endline "-- tar";
  let raw = tar [ ("a.txt", "hello world") ] in
  run (gzip (flip 3 raw));
  run (gzip (String.sub raw 0 700));
  run (gzip (String.sub raw 0 512));
  let bad_size =
    let b = Bytes.of_string raw in
    Bytes.blit_string "9999999999\000" 0 b 124 11;
    Bytes.to_string b
  in
  run (gzip bad_size);
  [%expect
    {|
    -- gzip
    archive is not valid gzip: shorter than a gzip header and trailer
    archive is not valid gzip: bad magic
    archive is not valid gzip: compression method is not deflate
    archive is not valid gzip: Corrupted data stream
    archive is not valid gzip: CRC-32 mismatch
    archive is not valid gzip: length mismatch
    archive is not valid gzip: Corrupted data stream
    -- tar
    header checksum mismatch: ""
    truncated archive: ""
    truncated archive: "a.txt"
    header checksum mismatch: "" |}]

let%expect_test "limits are enforced from headers" =
  let big = gzip (tar [ ("a", String.make 5000 'a') ]) in
  run ~limits:{ U.Targz.default_limits with max_member_bytes = 1000 } big;
  run ~limits:{ U.Targz.default_limits with max_archive_bytes = 2000 } big;
  run
    ~limits:{ U.Targz.default_limits with max_members = 1 }
    (gzip (tar [ ("a", "1"); ("b", "2") ]));
  [%expect
    {|
    member size is 5000, over the limit 1000
    archive is not valid gzip: Expected decompression size exceeded
    member count is 2, over the limit 1 |}]

(* Optional gzip header fields (RFC 1952) are skipped, not misread as data. *)
let%expect_test "gzip header fields" =
  let raw = tar [ ("a.txt", "hello") ] in
  let body =
    match Zipc_deflate.deflate raw with Ok s -> s | Error m -> failwith m
  in
  let le32 v =
    let b = Bytes.create 4 in
    Bytes.set_int32_le b 0 v;
    Bytes.to_string b
  in
  let trailer =
    le32 (Zipc_deflate.Crc_32.string raw)
    ^ le32 (Int32.of_int (String.length raw))
  in
  let with_header flags fields =
    "\x1f\x8b\x08"
    ^ String.make 1 (Char.chr flags)
    ^ "\x00\x00\x00\x00\x00\xff" ^ fields ^ body ^ trailer
  in
  print_endline "-- FEXTRA, FNAME, FCOMMENT, FHCRC";
  run (with_header 4 "\x03\x00abc");
  run (with_header 8 "name.tar\x00");
  run (with_header 16 "a comment\x00");
  run (with_header 2 "\x12\x34");
  run (with_header 0x1e "\x01\x00X" ^ "");
  print_endline "-- damaged";
  run (with_header 8 "unterminated");
  run (with_header 4 "\xff\xff");
  run (with_header 0x20 "");
  [%expect
    {|
    -- FEXTRA, FNAME, FCOMMENT, FHCRC
    a.txt (5)
    a.txt (5)
    a.txt (5)
    a.txt (5)
    archive is not valid gzip: truncated header
    -- damaged
    archive is not valid gzip: Corrupted data stream
    archive is not valid gzip: truncated header
    archive is not valid gzip: reserved flags set |}]
