(** What can be wrong between the consumer's pins and a released fixture. Each
    layer is named, so "the publication index differs from its pin" is never
    confused with "the archive differs from the manifest". *)

type layer =
  | Archive
  | Cases
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
  | Case_ids
  | Cases
  | Dialect
  | File_name
  | Graph_sha256
  | Map_member
  | Payload
  | Release_tag
  | Repository
  | Schema_version
  | Tolerances
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

type role = Inputs | Outputs

module Case_digest : sig
  type t = {
    actual : Pt2_sha256.Digest.t;
    case : string;
    expected : Pt2_sha256.Digest.t;
    role : role;
  }
end

module Case_names : sig
  type t = {
    actual : string list;
    case : string;
    expected : string list;
    role : role;
  }
end

module Logical_fault : sig
  type t =
    [ `Logical_numel_overflow
    | `Logical_over_limit of int64
    | `Logical_stride_range of int
    | `Logical_unsupported_dtype of Pt2_dtype.t ]
end

type error =
  [ `Case_digest of Case_digest.t
  | `Case_names of Case_names.t
  | `Cases_decode of string
  | `Cohort_decode of string
  | `Contract_decode of string
  | `Contract_mutations of string list
  | `Contract_shape of string
  | `Digest_clash of Digest_clash.t
  | `Digest_name of string
  | `Entry_missing of layer * string
  | `Field_clash of Field_clash.t
  | `Logical_tensor of string * Logical_fault.t
  | `Manifest_decode of string
  | `Member_missing of string
  | `Member_surplus of string
  | `Publication_decode of string
  | `Size_clash of Size_clash.t
  | `Source_pin_missing of string
  | Pt2_checkpoint_map.Fault.error ]

val pp_layer : layer Fmt.t
val pp_error : error Fmt.t
