module Map = Pt2_checkpoint_map
open Pt2_checkpoint_map_test.Fixtures

let write path data =
  Out_channel.with_open_bin path (fun oc -> output_string oc data)

let with_dir f =
  let dir = Filename.temp_file "pt2_map_unix" "" in
  Sys.remove dir;
  Unix.mkdir dir 0o700;
  Fun.protect
    ~finally:(fun () ->
      Array.iter (fun n -> Sys.remove (Filename.concat dir n)) (Sys.readdir dir);
      Unix.rmdir dir)
    (fun () -> f dir)

let document () =
  match Err.payload (Map.Document.of_string map_json) with
  | Ok d -> d
  | Error _ -> failwith "map must decode"

(* Everything the host held -- the mapping handles, the verified sources, the
   files on disk -- is gone by the time the captures are read. *)
let prepare_and_release dir =
  let doc = document () in
  let set =
    let open Err.Syntax in
    let* sources = Pt2_checkpoint_map_unix.sources_in_dir doc ~dir in
    let* verified = Map.Prepare.verify_sources doc sources in
    Map.Prepare.capture_set doc verified
  in
  match Err.payload set with Ok s -> s | Error _ -> failwith "prepare failed"

let%expect_test "captures outlive the files and the host's references" =
  with_dir (fun dir ->
      write (Filename.concat dir "toy.safetensors") toy_bytes;
      write (Filename.concat dir "pack.safetensors") pack_bytes;
      let set = prepare_and_release dir in
      Sys.remove (Filename.concat dir "toy.safetensors");
      Sys.remove (Filename.concat dir "pack.safetensors");
      Gc.full_major ();
      Gc.compact ();
      let doc = document () in
      List.iter
        (fun target ->
          let s = Option.get (Map.Prepare.find set target) in
          let entry = Schema_runtime.String_map.find target doc.tensors in
          let ok =
            Pt2_sha256.Digest.equal (Pt2_sha256.bigstring s) entry.sha256
          in
          Fmt.pr "%s %b@." target ok)
        (Map.Prepare.targets set));
  [%expect
    {|
    b true
    c true
    e true
    h true
    k true
    w true
    |}]

let%expect_test "a missing or unreadable file is a named error" =
  with_dir (fun dir ->
      write (Filename.concat dir "toy.safetensors") toy_bytes;
      (match
         Err.payload (Pt2_checkpoint_map_unix.sources_in_dir (document ()) ~dir)
       with
      | Ok _ -> print_endline "mapped?"
      | Error (`Source_io (path, _)) ->
          (* The operating system's wording after the path is not stable. *)
          Fmt.pr "cannot read %s@." (Filename.basename path));
      write (Filename.concat dir "pack.safetensors") "";
      match
        Err.payload
          (Pt2_checkpoint_map_unix.map_file
             (Filename.concat dir "pack.safetensors"))
      with
      | Ok b -> Fmt.pr "empty file maps to %d bytes@." (Bigarray.Array1.dim b)
      | Error _ -> print_endline "error");
  [%expect
    {|
    cannot read pack.safetensors
    empty file maps to 0 bytes |}]
