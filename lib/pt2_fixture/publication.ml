open Err.Syntax
module Pin = Pt2_checkpoint_map.Document.Pin

type entry = {
  archive : Pin.t;
  artifact_id : string;
  graph_sha256 : Pt2_sha256.Digest.t;
  manifest : Pin.t;
  sources : Pin.t list;
}

type t = { entries : entry list; release_tag : string; repository : string }

module Wire = struct
  type artifact = {
    artifact_id : string;
    assets : Pin_wire.t Schema_runtime.String_map.t;
    graph_sha256 : string;
  }

  type document = {
    artifacts : artifact list;
    release_tag : string;
    repository : string;
  }

  let artifact_jsont =
    Jsont.Object.map ~kind:"publication artifact"
      (fun artifact_id assets graph_sha256 ->
        { artifact_id; assets; graph_sha256 })
    |> Jsont.Object.mem "artifact_id" Jsont.string
    |> Jsont.Object.mem "assets" (Jsont.Object.as_string_map Pin_wire.jsont)
    |> Jsont.Object.mem "graph_sha256" Jsont.string
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  let jsont =
    Jsont.Object.map ~kind:"publication"
      (fun artifacts release_tag repository ->
        { artifacts; release_tag; repository })
    |> Jsont.Object.mem "artifacts" (Jsont.list artifact_jsont)
    |> Jsont.Object.mem "release_tag" Jsont.string
    |> Jsont.Object.mem "repository" Jsont.string
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish
end

let digest_of s =
  match Pt2_sha256.Digest.of_hex s with
  | Some d -> Err.return d
  | None -> Err.fail (`Bad_digest s)

let is_source_key key = String.length key > 3 && String.sub key 0 3 = "v2:"

let entry_of (w : Wire.artifact) =
  let find key =
    match Schema_runtime.String_map.find_opt key w.assets with
    | Some p -> Pin_wire.to_pin p
    | None ->
        Err.fail (`Entry_missing (Fault.Publication, w.artifact_id ^ "/" ^ key))
  in
  let* archive = find "archive" in
  let* manifest = find "manifest" in
  let* graph_sha256 = digest_of w.graph_sha256 in
  let+ sources =
    Err.List.map
      (fun (_, p) -> Pin_wire.to_pin p)
      (List.filter
         (fun (k, _) -> is_source_key k)
         (Schema_runtime.String_map.bindings w.assets))
  in
  { archive; artifact_id = w.artifact_id; graph_sha256; manifest; sources }

let of_string text =
  let* w =
    Jsont_bytesrw.decode_string Wire.jsont text
    |> Err.import ~pos:__POS__ (fun e -> `Publication_decode e)
  in
  let+ entries = Err.List.map entry_of w.artifacts in
  { entries; release_tag = w.release_tag; repository = w.repository }

let check (cohort : Cohort.t) (t : t) (want : Cohort.entry) =
  let* () =
    Check.field Fault.Publication Fault.Release_tag ~actual:t.release_tag
      ~expected:cohort.release_tag
  in
  let* () =
    Check.field Fault.Publication Fault.Repository ~actual:t.repository
      ~expected:cohort.repository
  in
  match
    List.find_opt
      (fun e -> String.equal e.artifact_id want.artifact_id)
      t.entries
  with
  | None -> Err.fail (`Entry_missing (Fault.Publication, want.artifact_id))
  | Some e ->
      let* () = Check.pin Fault.Archive e.archive want.archive in
      let* () = Check.pin Fault.Manifest e.manifest want.manifest in
      let* () =
        if Pt2_sha256.Digest.equal e.graph_sha256 want.graph_sha256 then
          Err.return ()
        else
          Err.fail
            (`Digest_clash
               {
                 Fault.Digest_clash.layer = Fault.Graph;
                 actual = e.graph_sha256;
                 expected = want.graph_sha256;
               })
      in
      Err.return e
