open Support
module W = Pt2_fixture_test.World
module Pin = Pt2_checkpoint_map.Document.Pin

let pin_of name data =
  match
    Err.payload
      (Pt2_checkpoint_map.Document.pin_of_fields ~name ~sha256:(W.hex data)
         ~size:(Int64.of_int (String.length data))
         ~url:(W.url name))
  with
  | Ok p -> p
  | Error _ -> failwith "pin"

let open_fixture ?transport ?(cohort = W.cohort) ?(entry = W.entry) dir =
  let config = U.Bundle.config ?transport (cache dir) in
  Err.payload (U.Fixture.open_ config cohort entry)

let%expect_test "cold fetch, then the same fixture offline with no requests" =
  with_dir (fun dir ->
      let calls = ref [] in
      (match open_fixture ~transport:(transport calls) dir with
      | Error e -> show_error e
      | Ok f ->
          Fmt.pr "captures: %s@."
            (String.concat " " (Pt2_checkpoint_map.Prepare.targets f.captures));
          Fmt.pr "requests: %d@." (List.length !calls);
          (* The archive answers through the existing interface. *)
          let w =
            Err.payload (Pt2_archive.load_captured_tensor f.archive "w")
          in
          (match w with
          | Ok t ->
              Fmt.pr "w: %d elements, dtype %s@." (Pt2_tensor.numel t)
                (Pt2_dtype.to_string t.dtype)
          | Error _ -> print_endline "w failed");
          Fmt.pr "no model.pt2 in the bundle: %b@."
            (not
               (Sys.file_exists
                  (Filename.concat f.bundle.dir "models/model.pt2"))));
      let again = ref [] in
      (match open_fixture dir with
      | Ok _ -> Fmt.pr "offline reopen ok, requests %d@." (List.length !again)
      | Error e -> show_error e);
      (* Offline into an empty cache is refused, naming the first missing file. *)
      with_dir (fun other -> report (open_fixture other |> Result.map ignore)));
  [%expect
    {|
    captures: b c e h k w
    requests: 5
    w: 6 elements, dtype float32
    no model.pt2 in the bundle: true
    offline reopen ok, requests 0
    "publication.json" is not in the cache and no transport is configured |}]

let%expect_test "downloads are verified before they reach the cache" =
  let attempt ?tweak ?fail () =
    with_dir (fun dir ->
        let calls = ref [] in
        report
          (open_fixture ~transport:(transport ?tweak ?fail calls) dir
          |> Result.map ignore);
        let c = Filename.concat dir "cache" in
        Fmt.pr "  blobs %d, tmp leftovers %d@."
          (Array.length (Sys.readdir (Filename.concat c "blobs")))
          (Array.length (Sys.readdir (Filename.concat c "tmp"))))
  in
  print_endline "-- truncated archive";
  attempt
    ~tweak:(fun u b ->
      if u = W.url "archive.tar.gz" then String.sub b 0 (String.length b - 3)
      else b)
    ();
  print_endline "-- same-size corruption of the archive";
  attempt
    ~tweak:(fun u b ->
      if u = W.url "archive.tar.gz" then
        String.mapi
          (fun i c -> if i = 20 then Char.chr (Char.code c lxor 1) else c)
          b
      else b)
    ();
  print_endline "-- a source of the right size and wrong bytes";
  attempt
    ~tweak:(fun u b ->
      if u = W.url "toy.safetensors" then String.make (String.length b) 'z'
      else b)
    ();
  print_endline "-- transport failure mid-download";
  attempt
    ~fail:(fun u ->
      if u = W.url "manifest.json" then Some "connection reset" else None)
    ();
  print_endline "-- wrong publication index";
  attempt
    ~tweak:(fun u b -> if u = W.url "publication.json" then b ^ " " else b)
    ();
  [%expect
    {|
    -- truncated archive
    the archive: size is 1937 bytes, expected 1940
      blobs 2, tmp leftovers 0
    -- same-size corruption of the archive
    the archive: digest is d9089128df72d46dbde5ab82c1742ecbc438d214ad458653d586672cab474afc, expected 1b2884d6a47093d45e0bc504d913eb01687afa516f7efb89e1a5df391a55f26f
      blobs 2, tmp leftovers 0
    -- a source of the right size and wrong bytes
    source "toy.safetensors": digest is 84d01f4bbee62f3dd865efe30944e45bb286cbc39d86f8aa1c2e1441fa9165ae, expected ef5f96f5c32e0bbad0811851b5ccab377754223601ceb3c5213f38376cb69d16
      blobs 3, tmp leftovers 0
    -- transport failure mid-download
    download of https://example.org/manifest.json failed: connection reset
      blobs 1, tmp leftovers 0
    -- wrong publication index
    the publication index: size is 584 bytes, expected 583
      blobs 0, tmp leftovers 0 |}]

let%expect_test "a corrupt cached file is an error offline and replaced online"
    =
  with_dir (fun dir ->
      let calls = ref [] in
      ignore (open_fixture ~transport:(transport calls) dir);
      let c = Filename.concat dir "cache" in
      let blob =
        Filename.concat (Filename.concat c "blobs") (W.hex W.Fx.toy_bytes)
      in
      let original = read blob in
      write blob
        (String.mapi
           (fun i ch -> if i = 40 then Char.chr (Char.code ch lxor 1) else ch)
           original);
      print_endline "-- offline";
      report (open_fixture dir |> Result.map ignore);
      print_endline "-- online";
      let again = ref [] in
      report (open_fixture ~transport:(transport again) dir |> Result.map ignore);
      Fmt.pr "re-fetched: %s@."
        (String.concat " "
           (List.map (fun u -> Filename.basename u) (List.rev !again)));
      Fmt.pr "blob restored: %b@." (read blob = original));
  [%expect
    {|
    -- offline
    source "toy.safetensors": digest is 379c535b1186b603153121ee5746235f55105d2eec27c6c671935c4db9b78d1d, expected ef5f96f5c32e0bbad0811851b5ccab377754223601ceb3c5213f38376cb69d16
    -- online
    ok
    re-fetched: toy.safetensors
    blob restored: true |}]

let%expect_test "the map's sources must be the ones the cohort pinned" =
  let with_source pin =
    let json =
      Pt2_checkpoint_map_test.Fixtures.replace
        ~sub:(W.pin_json ~name:"toy.safetensors" ~data:W.Fx.toy_bytes)
        ~by:pin W.cohort_bytes
    in
    W.decode_cohort json
  in
  let run cohort =
    with_dir (fun dir ->
        let calls = ref [] in
        let entry = List.hd cohort.Pt2_fixture.Cohort.entries in
        report
          (open_fixture ~transport:(transport calls) ~cohort ~entry dir
          |> Result.map ignore))
  in
  print_endline "-- the cohort pins the source with another digest";
  run
    (with_source
       (W.pin_json ~name:"toy.safetensors" ~data:(W.Fx.toy_bytes ^ "!")));
  print_endline "-- the cohort pins a different source";
  run (with_source (W.pin_json ~name:"other.safetensors" ~data:W.Fx.toy_bytes));
  print_endline "-- the cohort pins another URL";
  run
    (with_source
       (Pt2_checkpoint_map_test.Fixtures.replace ~sub:"example.org"
          ~by:"mirror.example"
          (W.pin_json ~name:"toy.safetensors" ~data:W.Fx.toy_bytes)));
  [%expect
    {|
    -- the cohort pins the source with another digest
    source "toy.safetensors": digest is ef5f96f5c32e0bbad0811851b5ccab377754223601ceb3c5213f38376cb69d16, expected b89bdf380eb2c2239469a4489b70695e6d85e55404dcf25dad70cfc9417df392
    -- the cohort pins a different source
    the cohort pins no source named "toy.safetensors"
    -- the cohort pins another URL
    source "toy.safetensors": URL is "https://example.org/toy.safetensors", expected "https://mirror.example/toy.safetensors" |}]
