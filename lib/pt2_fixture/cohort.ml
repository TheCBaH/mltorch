open Err.Syntax
module Pin = Pt2_checkpoint_map.Document.Pin

type entry = {
  artifact_id : string;
  archive : Pin.t;
  cases : string list;
  contract_sha256 : Pt2_sha256.Digest.t;
  graph_sha256 : Pt2_sha256.Digest.t;
  manifest : Pin.t;
  map_member : string;
  map_sha256 : Pt2_sha256.Digest.t;
  role : string;
  sources : Pin.t list;
}

type t = {
  entries : entry list;
  publication : Pin.t;
  release_producer_commit : string;
  release_tag : string;
  repository : string;
}

module Wire = struct
  type sources = { files : Pin_wire.t list }
  type map_sources = { checkpoint : sources }

  type entry = {
    artifact_id : string;
    archive : Pin_wire.t;
    cases : string list;
    contract_sha256 : string;
    graph_sha256 : string;
    manifest : Pin_wire.t;
    map_member : string;
    map_sha256 : string;
    map_sources : map_sources;
    role : string;
  }

  type document = {
    artifacts : entry list;
    publication : Pin_wire.anonymous;
    release_producer_commit : string;
    release_tag : string;
    repository : string;
  }

  let sources_jsont =
    Jsont.Object.map ~kind:"checkpoint sources" (fun files -> { files })
    |> Jsont.Object.mem "files" (Jsont.list Pin_wire.jsont)
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  let map_sources_jsont =
    Jsont.Object.map ~kind:"map sources" (fun checkpoint -> { checkpoint })
    |> Jsont.Object.mem "checkpoint" sources_jsont
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  let entry_jsont =
    Jsont.Object.map ~kind:"cohort artifact"
      (fun
        artifact_id
        archive
        cases
        contract_sha256
        graph_sha256
        manifest
        map_member
        map_sha256
        map_sources
        role
      ->
        {
          artifact_id;
          archive;
          cases;
          contract_sha256;
          graph_sha256;
          manifest;
          map_member;
          map_sha256;
          map_sources;
          role;
        })
    |> Jsont.Object.mem "artifact_id" Jsont.string
    |> Jsont.Object.mem "archive" Pin_wire.jsont
    |> Jsont.Object.mem "cases" (Jsont.list Jsont.string)
    |> Jsont.Object.mem "contract_sha256" Jsont.string
    |> Jsont.Object.mem "graph_sha256" Jsont.string
    |> Jsont.Object.mem "manifest" Pin_wire.jsont
    |> Jsont.Object.mem "map_member" Jsont.string
    |> Jsont.Object.mem "map_sha256" Jsont.string
    |> Jsont.Object.mem "map_sources" map_sources_jsont
    |> Jsont.Object.mem "role" Jsont.string
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish

  let jsont =
    Jsont.Object.map ~kind:"cohort"
      (fun
        artifacts publication release_producer_commit release_tag repository ->
        {
          artifacts;
          publication;
          release_producer_commit;
          release_tag;
          repository;
        })
    |> Jsont.Object.mem "artifacts" (Jsont.list entry_jsont)
    |> Jsont.Object.mem "publication" Pin_wire.anonymous_jsont
    |> Jsont.Object.mem "release_producer_commit" Jsont.string
    |> Jsont.Object.mem "release_tag" Jsont.string
    |> Jsont.Object.mem "repository" Jsont.string
    |> Jsont.Object.skip_unknown |> Jsont.Object.finish
end

let digest_of s =
  match Pt2_sha256.Digest.of_hex s with
  | Some d -> Err.return d
  | None -> Err.fail (`Bad_digest s)

let entry_of (w : Wire.entry) =
  let* archive = Pin_wire.to_pin w.archive in
  let* manifest = Pin_wire.to_pin w.manifest in
  let* contract_sha256 = digest_of w.contract_sha256 in
  let* graph_sha256 = digest_of w.graph_sha256 in
  let* map_sha256 = digest_of w.map_sha256 in
  let+ sources = Err.List.map Pin_wire.to_pin w.map_sources.checkpoint.files in
  {
    artifact_id = w.artifact_id;
    archive;
    cases = w.cases;
    contract_sha256;
    graph_sha256;
    manifest;
    map_member = w.map_member;
    map_sha256;
    role = w.role;
    sources;
  }

let of_string text =
  let* w =
    Jsont_bytesrw.decode_string Wire.jsont text
    |> Err.import ~pos:__POS__ (fun e -> `Cohort_decode e)
  in
  let* publication = Pin_wire.anonymous_to_pin w.publication in
  let+ entries = Err.List.map entry_of w.artifacts in
  {
    entries;
    publication;
    release_producer_commit = w.release_producer_commit;
    release_tag = w.release_tag;
    repository = w.repository;
  }

let find t id = List.find_opt (fun e -> String.equal e.artifact_id id) t.entries
