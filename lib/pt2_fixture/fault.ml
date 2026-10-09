type layer =
  | Archive
  | Contract
  | Graph
  | Manifest
  | Map
  | Member of string
  | Publication
  | Source of string

type field =
  | Artifact_id
  | Cases
  | File_name
  | Graph_sha256
  | Map_member
  | Payload
  | Release_tag
  | Repository
  | Schema_version
  | Url

module Digest_clash = struct
  type t = {
    actual : Pt2_sha256.Digest.t;
    expected : Pt2_sha256.Digest.t;
    layer : layer;
  }
end

module Size_clash = struct
  type t = { actual : int64; expected : int64; layer : layer }
end

module Field_clash = struct
  type t = { actual : string; expected : string; field : field; layer : layer }
end

type error =
  [ `Cohort_decode of string
  | `Digest_clash of Digest_clash.t
  | `Entry_missing of layer * string
  | `Field_clash of Field_clash.t
  | `Manifest_decode of string
  | `Member_missing of string
  | `Member_surplus of string
  | `Publication_decode of string
  | `Size_clash of Size_clash.t
  | `Source_pin_missing of string
  | Pt2_checkpoint_map.Fault.error ]

let pp_layer ppf = function
  | Archive -> Fmt.string ppf "the archive"
  | Contract -> Fmt.string ppf "contract.json"
  | Graph -> Fmt.string ppf "the graph"
  | Manifest -> Fmt.string ppf "the manifest"
  | Map -> Fmt.string ppf "the checkpoint map"
  | Member name -> Fmt.pf ppf "archive member %S" name
  | Publication -> Fmt.string ppf "the publication index"
  | Source name -> Fmt.pf ppf "source %S" name

let pp_field ppf f =
  Fmt.string ppf
    (match f with
    | Artifact_id -> "artifact id"
    | Cases -> "case list"
    | File_name -> "file name"
    | Graph_sha256 -> "graph digest"
    | Map_member -> "map member"
    | Payload -> "payload"
    | Release_tag -> "release tag"
    | Repository -> "repository"
    | Schema_version -> "schema version"
    | Url -> "URL")

let pp_error ppf : error -> unit = function
  | `Cohort_decode m -> Fmt.pf ppf "failed to decode the cohort manifest: %s" m
  | `Digest_clash { Digest_clash.layer; actual; expected } ->
      Fmt.pf ppf "%a: digest is %a, expected %a" pp_layer layer
        Pt2_sha256.Digest.pp actual Pt2_sha256.Digest.pp expected
  | `Entry_missing (layer, id) ->
      Fmt.pf ppf "%a has no entry for artifact %S" pp_layer layer id
  | `Field_clash { Field_clash.layer; field; actual; expected } ->
      Fmt.pf ppf "%a: %a is %S, expected %S" pp_layer layer pp_field field
        actual expected
  | `Manifest_decode m -> Fmt.pf ppf "failed to decode the manifest: %s" m
  | `Member_missing name -> Fmt.pf ppf "archive lacks member %S" name
  | `Member_surplus name ->
      Fmt.pf ppf "archive has member %S, which the manifest does not list" name
  | `Publication_decode m ->
      Fmt.pf ppf "failed to decode the publication index: %s" m
  | `Size_clash { Size_clash.layer; actual; expected } ->
      Fmt.pf ppf "%a: size is %Ld bytes, expected %Ld" pp_layer layer actual
        expected
  | `Source_pin_missing name ->
      Fmt.pf ppf "the cohort pins no source named %S" name
  | #Pt2_checkpoint_map.Fault.error as e ->
      Pt2_checkpoint_map.Fault.pp_error ppf e
