open Err.Syntax
module Document = Pt2_checkpoint_map.Document

type t = {
  env : Hf_hub.Env.t;
  hasher : Hf_hub_unix.hasher option;
  http : Hf_hub_unix.http option;
}

let make ?http ?hasher env = { env; hasher; http }
let of_environment () = make (Hf_hub_unix.Env.of_environment ())
let env t = t.env

(* Pins name the public endpoint whatever mirror [HF_ENDPOINT] selects. *)
let public = Hf_hub.Env.make ~cache_dir:"" ()

let canonical_url ~repo_id ~revision ~name =
  let open Result in
  bind (Hf_hub.Repo_id.of_string repo_id) (fun repo ->
      map
        (fun revision ->
          ( repo,
            revision,
            Hf_hub.Download.resolve_url public repo revision ~filename:name ))
        (Hf_hub.Revision.of_string revision))

let starts_with ~prefix s =
  String.length s >= String.length prefix
  && String.equal (String.sub s 0 (String.length prefix)) prefix

type route =
  | Mismatch
  | Transport
  | Hub of Hf_hub.Repo_id.t * Hf_hub.Revision.t

let route (s : Document.Source.t) =
  match s.provenance with
  | Document.Source.Converted _ -> Transport
  | Upstream { repo_id; revision } -> (
      match canonical_url ~repo_id ~revision ~name:s.pin.name with
      | Ok (repo, revision, url) when String.equal url s.pin.url ->
          Hub (repo, revision)
      | Ok _ | Error _ ->
          if starts_with ~prefix:Hf_hub.Env.default_endpoint s.pin.url then
            Mismatch
          else Transport)

let check ?(hash = Cache.default_hasher) (s : Document.Source.t)
    (blob : Hf_hub.Blob.t) =
  let name = s.pin.name in
  let layer = Pt2_fixture.Fault.Source name in
  let* () =
    match Unix.LargeFile.stat blob.path with
    | st -> Pt2_fixture.Check.size layer st.Unix.LargeFile.st_size s.pin.size
    | exception Unix.Unix_error (e, _, _) ->
        Err.fail (`File_io (blob.path, Unix.error_message e))
  in
  match blob.etag with
  | Some etag when String.equal etag (Pt2_sha256.Digest.to_hex s.pin.sha256) ->
      (* A git-LFS etag is the file's SHA-256, which the driver has already
         checked against the downloaded bytes. *)
      Err.return ()
  | Some _ | None ->
      let* actual =
        hash blob.path
        |> Err.import ~pos:__POS__ (fun m -> `File_io (blob.path, m))
      in
      Pt2_fixture.Check.digest layer actual s.pin.sha256

let ensure ?hash ~online t (s : Document.Source.t) =
  match route s with
  | Transport -> Err.return None
  | Mismatch -> Err.fail (`Hub_url_mismatch s.pin.name)
  | Hub (repo, revision) -> (
      let env = if online then t.env else { t.env with offline = true } in
      match
        Hf_hub_unix.download ~env ?http:t.http ?hasher:t.hasher ~revision ~repo
          ~filename:s.pin.name ()
      with
      | Ok blob ->
          let+ () = check ?hash s blob in
          Some blob.path
      | Error (Hf_hub.Error.Offline_miss _) -> Err.fail (`Offline s.pin.name)
      | Error e -> Err.fail (`Hub (s.pin.name, e)))
