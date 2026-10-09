let hex d = Pt2_sha256.Digest.to_hex d

(* Bytes (7i + 3) mod 256, independent of any digest under test; the expected
   values below were produced by Python's hashlib. *)
let pattern n = String.init n (fun i -> Char.chr (((i * 7) + 3) land 255))

let bigstring_of s =
  let b =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout (String.length s)
  in
  String.iteri (Bigarray.Array1.set b) s;
  b

(* FIPS 180-4 / NIST CAVS examples. *)
let%expect_test "published vectors" =
  let show s = print_endline (hex (Pt2_sha256.string s)) in
  show "";
  show "abc";
  show "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  show (String.make 1_000_000 'a');
  [%expect
    {|
    e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1
    cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0
    |}]

(* Lengths straddling the padding (55/56) and block (64) boundaries. *)
let expected =
  [
    (0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    (1, "084fed08b978af4d7d196a7446a86b58009e636b611db16211b65a9aadff29c5");
    (55, "e7313d333c272e639f790978283f9eb392e843d0f29b7016828bb1daa4aac70b");
    (56, "4324d65f3c103567f5589c710bc08f8523f929a9272e3af36fc968e52abc6c27");
    (57, "35df609437dcfea3279283ab79fd554e2bf78f8f7ae2de532d8ee300b09e8f73");
    (63, "81c80242132f230c3bd41b3e63bbcff16107339549214a99614ff26664625055");
    (64, "39e3d7b6b5d075d37d053ad89b24b41bef4f3c29760c84447cab3f3be1882241");
    (65, "aacca6ff74fdbb296d165a45cecfa04e5127bc008770fbbdd48006f2d2fae95e");
    (119, "9ce7368e4daf32341631b492e80359dc9f594b48453cd0dd5bf0b19279cc177e");
    (120, "7836b787757e95e58b3ca5aec90b1b004e8deba1e50e9675af9cabf1a13a04b5");
    (127, "a8d23e75d936f303d248888d9b165ee543f4cbafcad3c9dd2a79bd84faa11d07");
    (128, "d2742f1f4ac6bb7ca2b239ee18402ba8b3f9f8e652d2a72973c2b9ba11c08cf6");
    (129, "307f8fc2c1622b92762e818d39a185d4d667ad49a4b07ceae1f4afa008a93ec4");
    (1000, "1e9bc38cbf860b9ec31918b065f9b52476c549a782e0e7990bed8ce3868d2371");
    (100003, "b4bec991fc613fcb4d2a26eb529e493ed4e3152cc00f0a49bd39b3e48b34824e");
  ]

let%expect_test "string and bigstring agree with the reference at every length"
    =
  List.iter
    (fun (n, want) ->
      let s = pattern n in
      let a = hex (Pt2_sha256.string s) in
      let b = hex (Pt2_sha256.bigstring (bigstring_of s)) in
      if a <> want || b <> want then
        Fmt.pr "length %d: string %s bigstring %s want %s@." n a b want)
    expected;
  [%expect {| |}]

(* The digest cannot depend on how the input was cut, including cuts that
   leave a partial block, fill one exactly, and span several. *)
let%expect_test "every chunk size gives the single-shot digest" =
  let s = pattern 1000 in
  let want = List.assoc 1000 expected in
  for chunk = 1 to 150 do
    let t = Pt2_sha256.create () in
    let b = bigstring_of s in
    let u = Pt2_sha256.create () in
    let pos = ref 0 in
    while !pos < 1000 do
      let len = min chunk (1000 - !pos) in
      Pt2_sha256.add_string t ~pos:!pos ~len s;
      Pt2_sha256.add_bigstring u ~pos:!pos ~len b;
      pos := !pos + len
    done;
    let a = hex (Pt2_sha256.finish t) and c = hex (Pt2_sha256.finish u) in
    if a <> want || c <> want then Fmt.pr "chunk %d: %s %s@." chunk a c
  done;
  [%expect {| |}]

let%expect_test "ranges are checked and a finished state is closed" =
  let t = Pt2_sha256.create () in
  let try_ f =
    match f () with
    | () -> print_endline "ok"
    | exception Invalid_argument m -> print_endline m
  in
  try_ (fun () -> Pt2_sha256.add_string t ~pos:2 ~len:2 "abc");
  try_ (fun () -> Pt2_sha256.add_string t ~pos:(-1) "abc");
  try_ (fun () -> Pt2_sha256.add_string t ~len:(-1) "abc");
  try_ (fun () -> Pt2_sha256.add_string t ~pos:4 "abc");
  try_ (fun () -> Pt2_sha256.add_string t ~pos:3 "abc");
  try_ (fun () -> Pt2_sha256.add_string t ~pos:1 ~len:max_int "abc");
  Pt2_sha256.add_string t "abc";
  print_endline (hex (Pt2_sha256.finish t));
  try_ (fun () -> Pt2_sha256.add_string t "x");
  [%expect
    {|
    Pt2_sha256.add_string: range outside the input
    Pt2_sha256.add_string: range outside the input
    Pt2_sha256.add_string: range outside the input
    Pt2_sha256.add_string: range outside the input
    ok
    Pt2_sha256.add_string: range outside the input
    ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    Pt2_sha256.add_string: already finished
    |}]

let%expect_test "hex form is exactly 64 lower-case digits" =
  let ok = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" in
  let show s =
    match Pt2_sha256.Digest.of_hex s with
    | None -> print_endline "none"
    | Some d -> print_endline (hex d)
  in
  show ok;
  show (String.uppercase_ascii ok);
  show (String.sub ok 0 63);
  show (ok ^ "0");
  show (String.make 64 'g');
  show "";
  [%expect
    {|
    ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    none
    none
    none
    none
    none
    |}]
