open Err.Syntax
module Pin = Pt2_checkpoint_map.Document.Pin

type t = { root : string }
type file_hasher = string -> (Pt2_sha256.Digest.t, string) result

let default_hasher path =
  match Err.payload (Pt2_checkpoint_map_unix.map_file path) with
  | Ok bytes -> Ok (Pt2_sha256.bigstring bytes)
  | Error (`Source_io (_, m)) -> Error m

let io path e = `File_io (path, Printexc.to_string e)

let rec mkdir_p dir =
  if not (Sys.file_exists dir) then begin
    mkdir_p (Filename.dirname dir);
    try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ()
  end

let create root =
  match
    List.iter
      (fun d -> mkdir_p (Filename.concat root d))
      [ "blobs"; "bundles"; "tmp" ]
  with
  | () -> Err.return { root }
  | exception e -> Err.fail (io root e)

let root t = t.root

let blob_path t d =
  Filename.concat (Filename.concat t.root "blobs") (Pt2_sha256.Digest.to_hex d)

let bundle_path t d =
  Filename.concat
    (Filename.concat t.root "bundles")
    (Pt2_sha256.Digest.to_hex d)

let temp_dir t = Filename.concat t.root "tmp"

type lookup = Absent | Present of string

(* Size first, which is free; the digest only for a file of the right size. *)
let check_file ?(hash = default_hasher) ~layer path (pin : Pin.t) =
  match Unix.LargeFile.stat path with
  | exception Unix.Unix_error (e, _, _) ->
      Err.fail (`File_io (path, Unix.error_message e))
  | st ->
      let* () =
        Pt2_fixture.Check.size layer st.Unix.LargeFile.st_size pin.size
      in
      let* actual =
        hash path |> Err.import ~pos:__POS__ (fun m -> `File_io (path, m))
      in
      Pt2_fixture.Check.digest layer actual pin.sha256

let lookup ?hash ~layer t (pin : Pin.t) =
  let path = blob_path t pin.sha256 in
  if not (Sys.file_exists path) then Err.return Absent
  else
    let+ () = check_file ?hash ~layer path pin in
    Present path

let promote ?hash ~layer t ~tmp (pin : Pin.t) =
  let remove () = try Sys.remove tmp with Sys_error _ -> () in
  match check_file ?hash ~layer tmp pin with
  | Error _ as e ->
      remove ();
      e
  | Ok () -> (
      let dest = blob_path t pin.sha256 in
      match Unix.rename tmp dest with
      | () -> Err.return dest
      | exception Unix.Unix_error (e, _, _) ->
          remove ();
          Err.fail (`File_io (dest, Unix.error_message e)))
