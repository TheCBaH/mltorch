open Support
module W = Pt2_fixture_test.World

(* A release whose archive is [members'], with every pin above it recomputed so
   the layers agree with each other -- but the manifest still lists the
   original members. Only the archive's contents contradict the manifest. *)
let world_with_archive members' =
  let archive = W.gzip (W.tar members') in
  let manifest =
    Pt2_checkpoint_map_test.Fixtures.replace
      ~sub:
        (Printf.sprintf {|"sha256":%S,"size":%d|} (W.hex W.archive_bytes)
           (String.length W.archive_bytes))
      ~by:
        (Printf.sprintf {|"sha256":%S,"size":%d|} (W.hex archive)
           (String.length archive))
      W.manifest_bytes
  in
  let archive_pin = W.pin_json ~name:"archive.tar.gz" ~data:archive in
  let manifest_pin = W.pin_json ~name:"manifest.json" ~data:manifest in
  let publication =
    W.publication_json ~archive:archive_pin ~manifest:manifest_pin ()
  in
  let cohort =
    W.decode_cohort
      (W.cohort_json ~publication ~archive:archive_pin ~manifest:manifest_pin ())
  in
  let served =
    [
      (W.url "publication.json", publication);
      (W.url "manifest.json", manifest);
      (W.url "archive.tar.gz", archive);
      (W.url "toy.safetensors", W.Fx.toy_bytes);
      (W.url "pack.safetensors", W.Fx.pack_bytes);
    ]
  in
  (cohort, served)

let fetch_with (cohort, served) =
  with_dir (fun dir ->
      let calls = ref [] in
      let t : U.Transport.t =
       fun ~url ~dest ->
        calls := url :: !calls;
        match List.assoc_opt url served with
        | Some b ->
            write dest b;
            Ok ()
        | None -> Error "404"
      in
      let config = U.Bundle.config ~transport:t (cache dir) in
      let r =
        Err.payload (U.Bundle.ensure config cohort (List.hd cohort.entries))
      in
      report (Result.map ignore r);
      Fmt.pr "  bundles on disk: %d@."
        (Array.length
           (Sys.readdir
              (Filename.concat (Filename.concat dir "cache") "bundles"))))

let%expect_test "the archive's members must be exactly the manifest's" =
  print_endline "-- the untouched archive";
  fetch_with (world_with_archive W.members);
  print_endline "-- an extra member";
  fetch_with (world_with_archive (("extra.txt", "x") :: W.members));
  print_endline "-- a missing member";
  fetch_with (world_with_archive (List.remove_assoc "cases.json" W.members));
  print_endline "-- a member changed, same size";
  fetch_with
    (world_with_archive
       (( "contract.json",
          String.map (fun c -> if c = 't' then 'T' else c) W.contract_json )
       :: List.remove_assoc "contract.json" W.members));
  print_endline "-- a member of a different size";
  fetch_with
    (world_with_archive
       (("cases.json", "{}") :: List.remove_assoc "cases.json" W.members));
  [%expect
    {|
    -- the untouched archive
    ok
      bundles on disk: 1
    -- an extra member
    archive has member "extra.txt", which the manifest does not list
      bundles on disk: 0
    -- a missing member
    archive lacks member "cases.json"
      bundles on disk: 0
    -- a member changed, same size
    archive member "contract.json": digest is 31b58305e15484df1507717a9baa9688cbfd2114317eeae7cdb8d26ff442c749, expected 5340bfc94de5e845755366acd4a0ed397fe123709f14edc036498c3d40237449
      bundles on disk: 0
    -- a member of a different size
    archive member "cases.json": size is 2 bytes, expected 12
      bundles on disk: 0 |}]

let bundle dir =
  let calls = ref [] in
  let config = U.Bundle.config ~transport:(transport calls) (cache dir) in
  match Err.payload (U.Bundle.ensure config W.cohort W.entry) with
  | Ok b -> (config, b)
  | Error e ->
      show_error e;
      failwith "bundle"

let%expect_test "an extracted directory is reverified every time it is opened" =
  with_dir (fun dir ->
      let config, b = bundle dir in
      let reopen label =
        print_endline ("-- " ^ label);
        report
          (Result.map ignore
             (Err.payload
                (U.Bundle.ensure
                   { config with transport = None }
                   W.cohort W.entry)))
      in
      reopen "pristine";
      let path n = Filename.concat b.dir n in
      let original = read (path "contract.json") in
      write (path "contract.json")
        (String.map (fun c -> if c = 't' then 'T' else c) original);
      reopen "member changed";
      write (path "contract.json") original;
      write (path "stray.txt") "x";
      reopen "extra file";
      Sys.remove (path "stray.txt");
      Unix.symlink "contract.json" (path "link");
      reopen "symlink";
      Sys.remove (path "link");
      Unix.mkdir (path "empty_dir_is_fine") 0o755;
      reopen "empty directory";
      Unix.rmdir (path "empty_dir_is_fine");
      Sys.remove (path "cases.json");
      reopen "missing member";
      write (path "cases.json") {|{"cases":[]}|};
      reopen "restored");
  [%expect
    {|
    -- pristine
    ok
    -- member changed
    archive member "contract.json": digest is 31b58305e15484df1507717a9baa9688cbfd2114317eeae7cdb8d26ff442c749, expected 5340bfc94de5e845755366acd4a0ed397fe123709f14edc036498c3d40237449
    -- extra file
    archive has member "stray.txt", which the manifest does not list
    -- symlink
    archive has member "link", which the manifest does not list
    -- empty directory
    ok
    -- missing member
    archive lacks member "cases.json"
    -- restored
    ok |}]

let contains ~sub s =
  let n = String.length sub in
  let rec go i =
    i + n <= String.length s && (String.sub s i n = sub || go (i + 1))
  in
  go 0

(* The directory is verified once more before the rename. A hasher that
   misreports files while they sit in the temporary directory makes that last
   check fail; nothing may then appear under bundles/, and nothing may be left
   in tmp/. *)
let%expect_test
    "extraction is atomic: a bundle exists whole and verified or not at all" =
  with_dir (fun dir ->
      let calls = ref [] in
      let lying_hasher path =
        if contains ~sub:"/tmp/bundle" path then Ok (Pt2_sha256.string "lie")
        else U.Cache.default_hasher path
      in
      let config =
        U.Bundle.config ~hash:lying_hasher ~transport:(transport calls)
          (cache dir)
      in
      report
        (Result.map ignore
           (Err.payload (U.Bundle.ensure config W.cohort W.entry)));
      let c = Filename.concat dir "cache" in
      Fmt.pr "bundles: %d, tmp: %d@."
        (Array.length (Sys.readdir (Filename.concat c "bundles")))
        (Array.length (Sys.readdir (Filename.concat c "tmp"))));
  [%expect
    {|
    archive member "captures.json": digest is c8b1a1a3e6bcee8660ad4a4ab2e2d318295fadc325f85525285f09a16e50ebc4, expected cbe85ed816124fd91663a8d3bc69abc25bf8c52bc36e8cf86e4896cda6895afc
    bundles: 0, tmp: 0 |}]
