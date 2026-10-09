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

type role = Inputs | Outputs

module Case_digest = struct
  type t = {
    actual : Pt2_sha256.Digest.t;
    case : string;
    expected : Pt2_sha256.Digest.t;
    role : role;
  }
end

module Case_names = struct
  type t = {
    actual : string list;
    case : string;
    expected : string list;
    role : role;
  }
end

module Logical_fault = struct
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

let pp_layer ppf = function
  | Archive -> Fmt.string ppf "the archive"
  | Cases -> Fmt.string ppf "cases.json"
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
    | Case_ids -> "case ids"
    | Cases -> "case list"
    | Dialect -> "dialect"
    | File_name -> "file name"
    | Graph_sha256 -> "graph digest"
    | Map_member -> "map member"
    | Payload -> "payload"
    | Release_tag -> "release tag"
    | Repository -> "repository"
    | Schema_version -> "schema version"
    | Tolerances -> "tolerances"
    | Url -> "URL")

let pp_error ppf : error -> unit = function
  | `Case_digest { Case_digest.case; role; actual; expected } ->
      Fmt.pf ppf "case %s %s: content digest is %a, descriptor says %a" case
        (match role with Inputs -> "inputs" | Outputs -> "outputs")
        Pt2_sha256.Digest.pp actual Pt2_sha256.Digest.pp expected
  | `Case_names { Case_names.case; role; actual; expected } ->
      let names = Fmt.(brackets (list ~sep:semi string)) in
      Fmt.pf ppf "case %s %s: tensors are %a, expected %a" case
        (match role with Inputs -> "inputs" | Outputs -> "outputs")
        names actual names expected
  | `Cases_decode m -> Fmt.pf ppf "failed to decode cases.json: %s" m
  | `Cohort_decode m -> Fmt.pf ppf "failed to decode the cohort manifest: %s" m
  | `Contract_decode m -> Fmt.pf ppf "failed to decode contract.json: %s" m
  | `Contract_mutations ms ->
      Fmt.pf ppf "the contract declares mutations: %a"
        Fmt.(list ~sep:(any ", ") string)
        ms
  | `Contract_shape m ->
      Fmt.pf ppf "the contract is not a plain tensor call: %s" m
  | `Digest_clash { Digest_clash.layer; actual; expected } ->
      Fmt.pf ppf "%a: digest is %a, expected %a" pp_layer layer
        Pt2_sha256.Digest.pp actual Pt2_sha256.Digest.pp expected
  | `Digest_name n ->
      Fmt.pf ppf
        "tensor name %S is not plain ASCII; its digest line is not reproduced" n
  | `Entry_missing (layer, id) ->
      Fmt.pf ppf "%a has no entry for artifact %S" pp_layer layer id
  | `Field_clash { Field_clash.layer; field; actual; expected } ->
      Fmt.pf ppf "%a: %a is %S, expected %S" pp_layer layer pp_field field
        actual expected
  | `Manifest_decode m -> Fmt.pf ppf "failed to decode the manifest: %s" m
  | `Member_missing name -> Fmt.pf ppf "archive lacks member %S" name
  | `Member_surplus name ->
      Fmt.pf ppf "archive has member %S, which the manifest does not list" name
  | `Logical_tensor (name, e) ->
      Fmt.pf ppf "tensor %S: %a" name Logical.pp_error e
  | `Publication_decode m ->
      Fmt.pf ppf "failed to decode the publication index: %s" m
  | `Size_clash { Size_clash.layer; actual; expected } ->
      Fmt.pf ppf "%a: size is %Ld bytes, expected %Ld" pp_layer layer actual
        expected
  | `Source_pin_missing name ->
      Fmt.pf ppf "the cohort pins no source named %S" name
  | #Pt2_checkpoint_map.Fault.error as e ->
      Pt2_checkpoint_map.Fault.pp_error ppf e
