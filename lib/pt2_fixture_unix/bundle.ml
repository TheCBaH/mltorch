open Err.Syntax
module Cohort = Pt2_fixture.Cohort
module Manifest = Pt2_fixture.Manifest
module String_map = Schema_runtime.String_map

type config = {
  cache : Cache.t;
  hash : Cache.file_hasher option;
  limits : Targz.limits;
  transport : Transport.t option;
}

let config ?hash ?(limits = Targz.default_limits) ?transport cache =
  { cache; hash; limits; transport }

type t = { dir : string; entry : Cohort.entry; manifest : Manifest.t }

let rec files_under dir rel =
  let here = if rel = "" then dir else Filename.concat dir rel in
  Array.to_list (Sys.readdir here)
  |> List.sort String.compare
  |> List.concat_map (fun n ->
      let rel = if rel = "" then n else rel ^ "/" ^ n in
      match (Unix.lstat (Filename.concat dir rel)).Unix.st_kind with
      | Unix.S_REG -> [ (rel, true) ]
      | Unix.S_DIR -> files_under dir rel
      | _ -> [ (rel, false) ])

let verify_inventory ?(hash = Cache.default_hasher) ~dir inventory =
  let* () =
    match Unix.lstat dir with
    | stat when stat.Unix.st_kind = Unix.S_DIR -> Ok ()
    | _ -> Err.fail (`File_io (dir, "bundle root is not a directory"))
    | exception Unix.Unix_error (e, _, _) ->
        Err.fail (`File_io (dir, Unix.error_message e))
  in
  match files_under dir "" with
  | exception (Sys_error m | Failure m) -> Err.fail (`File_io (dir, m))
  | exception Unix.Unix_error (e, _, _) ->
      Err.fail (`File_io (dir, Unix.error_message e))
  | found ->
      let* () =
        Err.List.iter
          (fun (name, regular) ->
            if regular && String_map.mem name inventory then Err.return ()
            else Err.fail (`Member_surplus name))
          found
      in
      Err.List.iter
        (fun (name, (m : Manifest.member)) ->
          if not (List.mem_assoc name found) then
            Err.fail (`Member_missing name)
          else
            let path = Filename.concat dir name in
            let layer = Pt2_fixture.Fault.Member name in
            let* () =
              Pt2_fixture.Check.size layer
                (Unix.LargeFile.stat path).Unix.LargeFile.st_size m.size
            in
            let* actual =
              hash path |> Err.import ~pos:__POS__ (fun e -> `File_io (path, e))
            in
            Pt2_fixture.Check.digest layer actual m.sha256)
        (String_map.bindings inventory)

let verify_dir ?hash ~dir (manifest : Manifest.t) =
  verify_inventory ?hash ~dir manifest.members

let rec remove_tree path =
  match (Unix.lstat path).Unix.st_kind with
  | exception Unix.Unix_error _ -> ()
  | Unix.S_DIR -> (
      Array.iter
        (fun n -> remove_tree (Filename.concat path n))
        (Sys.readdir path);
      try Unix.rmdir path with Unix.Unix_error _ -> ())
  | _ -> ( try Sys.remove path with Sys_error _ -> ())

let write_member dir (m : Targz.member) =
  let path = Filename.concat dir m.name in
  Cache.mkdir_p (Filename.dirname path);
  Out_channel.with_open_bin path (fun oc -> output_string oc m.data)

(* Members first, against the manifest, in memory; nothing reaches the disk
   until the whole set is right. *)
let check_members inventory (members : Targz.member list) =
  let listed = List.map (fun (m : Targz.member) -> m.name) members in
  let* () =
    Err.List.iter
      (fun n ->
        if String_map.mem n inventory then Err.return ()
        else Err.fail (`Member_surplus n))
      listed
  in
  let* () =
    Err.List.iter
      (fun (n, _) ->
        if List.mem n listed then Err.return ()
        else Err.fail (`Member_missing n))
      (String_map.bindings inventory)
  in
  Err.List.iter
    (fun (m : Targz.member) ->
      let (pin : Manifest.member) = String_map.find m.name inventory in
      let layer = Pt2_fixture.Fault.Member m.name in
      let* () =
        Pt2_fixture.Check.size layer
          (Int64.of_int (String.length m.data))
          pin.size
      in
      Pt2_fixture.Check.digest layer (Pt2_sha256.string m.data) pin.sha256)
    members

let extract (c : config) (archive_pin : Cache.Pin.t) inventory archive_path =
  let dest = Cache.bundle_path c.cache archive_pin.sha256 in
  if Sys.file_exists dest then
    let+ () = verify_inventory ?hash:c.hash ~dir:dest inventory in
    dest
  else
    let* gz = Fetch.read ~max_bytes:0x2000_0000 archive_path in
    let* members = Targz.members ~limits:c.limits gz in
    let* () = check_members inventory members in
    let tmp =
      Filename.temp_file ~temp_dir:(Cache.temp_dir c.cache) "bundle" ".d"
    in
    Sys.remove tmp;
    let cleanup () = remove_tree tmp in
    match
      Unix.mkdir tmp 0o755;
      List.iter (write_member tmp) members
    with
    | exception (Sys_error m | Failure m) ->
        cleanup ();
        Err.fail (`File_io (tmp, m))
    | exception Unix.Unix_error (e, _, _) ->
        cleanup ();
        Err.fail (`File_io (tmp, Unix.error_message e))
    | () -> (
        match verify_inventory ?hash:c.hash ~dir:tmp inventory with
        | Error _ as e ->
            cleanup ();
            e
        | Ok () -> (
            match Unix.rename tmp dest with
            | () -> Err.return dest
            | exception Unix.Unix_error (e, _, _) ->
                cleanup ();
                (* A concurrent extraction may have won the rename. *)
                if Sys.file_exists dest then
                  let+ () = verify_inventory ?hash:c.hash ~dir:dest inventory in
                  dest
                else Err.fail (`File_io (dest, Unix.error_message e))))

let ensure_inventory (c : config) archive inventory =
  let* () =
    Err.List.iter
      (fun (name, _) ->
        if Targz.safe_name name then Ok () else Err.fail (`Member_surplus name))
      (String_map.bindings inventory)
  in
  let existing = Cache.bundle_path c.cache archive.Cache.Pin.sha256 in
  if Sys.file_exists existing then
    let+ () = verify_inventory ?hash:c.hash ~dir:existing inventory in
    existing
  else
    let* path =
      Fetch.ensure ?hash:c.hash ?transport:c.transport
        ~layer:Pt2_fixture.Fault.Archive c.cache archive
    in
    extract c archive inventory path

let ensure (c : config) (cohort : Cohort.t) (want : Cohort.entry) =
  let* pub_path =
    Fetch.ensure ?hash:c.hash ?transport:c.transport
      ~layer:Pt2_fixture.Fault.Publication c.cache cohort.publication
  in
  let* pub_text = Fetch.read pub_path in
  let* publication = Pt2_fixture.Publication.of_string pub_text in
  let* _ = Pt2_fixture.Publication.check cohort publication want in
  let* manifest_path =
    Fetch.ensure ?hash:c.hash ?transport:c.transport
      ~layer:Pt2_fixture.Fault.Manifest c.cache want.manifest
  in
  let* manifest_text = Fetch.read manifest_path in
  let* manifest = Manifest.of_string manifest_text in
  let* () = Manifest.check want manifest in
  (* A bundle already extracted for this archive digest is reused after it has
     been verified again; only otherwise is the archive fetched. *)
  let existing = Cache.bundle_path c.cache want.archive.sha256 in
  let* dir =
    if Sys.file_exists existing then
      let+ () = verify_dir ?hash:c.hash ~dir:existing manifest in
      existing
    else
      let* archive_path =
        Fetch.ensure ?hash:c.hash ?transport:c.transport
          ~layer:Pt2_fixture.Fault.Archive c.cache want.archive
      in
      extract c want.archive manifest.members archive_path
  in
  Err.return { dir; entry = want; manifest }

let read_member ?max_bytes t name =
  if not (String_map.mem name t.manifest.members) then
    Err.fail (`Member_missing name)
  else Fetch.read ?max_bytes (Filename.concat t.dir name)
