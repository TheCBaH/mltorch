(** What can be wrong between the consumer's pins and a released fixture. Each
    layer is named, so "the publication index differs from its pin" is never
    confused with "the archive differs from the manifest". *)

type layer =
  | Archive
  | Contract
  | Graph
  | Manifest
  | Map
  | Member of string
  | Publication
  | Source of string
      (** Alphabetical. A [Member] is one file inside the archive; a [Source] a
          checkpoint file the map names. *)

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
  | Url  (** A scalar both sides of a comparison carry. *)

module Digest_clash : sig
  type t = {
    actual : Pt2_sha256.Digest.t;
    expected : Pt2_sha256.Digest.t;
    layer : layer;
  }
end

module Size_clash : sig
  type t = { actual : int64; expected : int64; layer : layer }
end

module Field_clash : sig
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

val pp_layer : layer Fmt.t
val pp_error : error Fmt.t
