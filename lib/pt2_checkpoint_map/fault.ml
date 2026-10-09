module Pair = struct
  type 'a t = { actual : 'a; expected : 'a }
end

type view = Computed | Config | Inventory | Signature | Source

module Clash = struct
  type 'a t = { actual : 'a; against : view; expected : 'a; target : string }
end

module Cast = struct
  type t = { from : Dtype.t; target : string; to_ : Dtype.t }
end

module Size_clash = struct
  type t = { actual : int64; expected : int64; target : string }
end

module Key_ref = struct
  type t = { file : string; key : string; target : string }
end

module Source_digest = struct
  type t = {
    actual : Pt2_sha256.Digest.t;
    expected : Pt2_sha256.Digest.t;
    name : string;
  }
end

module Stored_dtype = struct
  type t = { actual : string; expected : Dtype.t; target : string }
end

module Over_limit = struct
  type t = { actual : int64; limit : int64; what : Limits.which }
end

type domain = Capture | Checkpoint_file | Pack_key

type document =
  | Captures_json
  | Map  (** Which of the bundle's documents a statement is about. *)

type capture_kind = Buffer | Constant_tensor | Parameter
type pin_field = Name | Provenance | Revision | Sha256 | Size | Url

type error =
  [ `Artifact_mismatch of document * string Pair.t
  | `Bad_digest of string
  | `Bad_pin of pin_field * string
  | `Base64_invalid of string
  | `Cast_inconsistent of Cast.t
  | `Captures_decode of string
  | `Config_missing of string
  | `Config_surplus of string
  | `Conversion_malformed of string
  | `Dtype_clash of Dtype.t Clash.t
  | `Duplicate of domain * string
  | `Element_width of Size_clash.t
  | `Empty_not_empty of string
  | `Generated_malformed of string
  | `Graph_digest_mismatch of document * Pt2_sha256.Digest.t Pair.t
  | `Graph_layout of string
  | `Graph_tensor of string * Pt2_tensor.error
  | `Hex_invalid of string
  | `Inline_size of Size_clash.t
  | `Inventory_missing of string
  | `Inventory_surplus of string
  | `Key_missing of Key_ref.t
  | `Kind_clash of capture_kind Clash.t
  | `Map_json_decode of string
  | `Missing_tensor of string
  | `Over_limit of Over_limit.t
  | `Pack_without_source of string
  | `Negative_extent of string
  | `Safetensors_header of string * string
  | `Safetensors_view of string * string
  | `Schema_version of int
  | `Shape_clash of int64 list Clash.t
  | `Source_digest_mismatch of Source_digest.t
  | `Source_missing of string
  | `Source_size_mismatch of Size_clash.t
  | `Source_surplus of string
  | `Stored_dtype of Stored_dtype.t
  | `Surplus_tensor of string
  | `Unknown_dtype of string
  | `Unknown_source_file of string * string
  | `Unmapped of string list
  | `Unsupported_conversion of Cast.t
  | `Value_digest_clash of Pt2_sha256.Digest.t Clash.t
  | `Value_length of Size_clash.t ]

let pp_shape = Fmt.(hbox (brackets (list ~sep:semi int64)))

let pp_domain ppf d =
  Fmt.string ppf
    (match d with
    | Capture -> "capture"
    | Checkpoint_file -> "checkpoint file"
    | Pack_key -> "pack key")

let pp_document ppf d =
  Fmt.string ppf
    (match d with Captures_json -> "captures.json" | Map -> "the map")

let pp_view ppf v =
  Fmt.string ppf
    (match v with
    | Computed -> "the prepared bytes"
    | Config -> "the graph config"
    | Inventory -> "captures.json"
    | Signature -> "the graph signature"
    | Source -> "the source header")

let pp_kind ppf k =
  Fmt.string ppf
    (match k with
    | Buffer -> "BUFFER"
    | Constant_tensor -> "CONSTANT_TENSOR"
    | Parameter -> "PARAMETER")

let pp_field ppf f =
  Fmt.string ppf
    (match f with
    | Name -> "name"
    | Provenance -> "provenance (repo_id and revision, or derived_from)"
    | Revision -> "revision"
    | Sha256 -> "sha256"
    | Size -> "size"
    | Url -> "url")

let pp_error ppf : error -> unit = function
  | `Artifact_mismatch (doc, { Pair.actual; expected }) ->
      Fmt.pf ppf "%a is for artifact %S, not %S" pp_document doc actual expected
  | `Bad_digest s -> Fmt.pf ppf "%S is not a 64-digit lower-case sha256" s
  | `Bad_pin (field, v) ->
      Fmt.pf ppf "invalid pinned-file %a %S" pp_field field v
  | `Base64_invalid target ->
      Fmt.pf ppf "capture %S: inline data is not canonical base64" target
  | `Cast_inconsistent { Cast.target; from; to_ } ->
      Fmt.pf ppf "capture %S: cast %a to %a does not produce the declared dtype"
        target Dtype.pp from Dtype.pp to_
  | `Captures_decode m -> Fmt.pf ppf "failed to decode captures.json: %s" m
  | `Config_missing target ->
      Fmt.pf ppf "capture %S is in neither weights nor constants config" target
  | `Conversion_malformed target ->
      Fmt.pf ppf "capture %S: malformed conversion" target
  | `Config_surplus target ->
      Fmt.pf ppf "a payload config lists %S, which the graph does not capture"
        target
  | `Dtype_clash { Clash.target; against; expected; actual } ->
      Fmt.pf ppf "capture %S: per %a dtype is %a, map says %a" target pp_view
        against Dtype.pp expected Dtype.pp actual
  | `Duplicate (domain, name) ->
      Fmt.pf ppf "duplicate %a %S" pp_domain domain name
  | `Element_width { Size_clash.target; expected; actual } ->
      Fmt.pf ppf "capture %S: fill element is %Ld bytes, dtype needs %Ld" target
        actual expected
  | `Empty_not_empty target ->
      Fmt.pf ppf "capture %S: generated empty but the shape has elements" target
  | `Generated_malformed target ->
      Fmt.pf ppf "capture %S: malformed generated origin" target
  | `Graph_digest_mismatch (doc, { Pair.actual; expected }) ->
      Fmt.pf ppf "graph bytes hash to %a, %a pins %a" Pt2_sha256.Digest.pp
        actual pp_document doc Pt2_sha256.Digest.pp expected
  | `Graph_layout target ->
      Fmt.pf ppf "capture %S: graph tensor is not a dense row-major buffer"
        target
  | `Graph_tensor (target, e) ->
      Fmt.pf ppf "capture %S: %a" target Pt2_tensor.pp_error e
  | `Hex_invalid target ->
      Fmt.pf ppf "capture %S: fill element is not lower-case hex" target
  | `Inline_size { Size_clash.target; expected; actual } ->
      Fmt.pf ppf "capture %S: inline data is %Ld bytes, shape needs %Ld" target
        actual expected
  | `Inventory_missing target ->
      Fmt.pf ppf "graph capture %S is not in captures.json" target
  | `Inventory_surplus target ->
      Fmt.pf ppf "captures.json lists %S, which the graph does not capture"
        target
  | `Key_missing { Key_ref.target; file; key } ->
      Fmt.pf ppf "capture %S: %S has no tensor %S" target file key
  | `Kind_clash { Clash.target; against; expected; actual } ->
      Fmt.pf ppf "capture %S: per %a kind is %a, captures.json says %a" target
        pp_view against pp_kind expected pp_kind actual
  | `Map_json_decode m -> Fmt.pf ppf "failed to decode the checkpoint map: %s" m
  | `Missing_tensor target ->
      Fmt.pf ppf "capture %S has no entry in the map" target
  | `Over_limit { Over_limit.what; limit; actual } ->
      Fmt.pf ppf "%a is %Ld, over the limit %Ld" Limits.pp_which what actual
        limit
  | `Pack_without_source target ->
      Fmt.pf ppf "capture %S reads the graph-owned file, which the map omits"
        target
  | `Negative_extent target ->
      Fmt.pf ppf "capture %S: a shape extent is negative" target
  | `Safetensors_header (file, m) ->
      Fmt.pf ppf "source %S is not a valid safetensors file: %s" file m
  | `Safetensors_view (target, m) ->
      Fmt.pf ppf "capture %S: cannot read its stored tensor: %s" target m
  | `Schema_version v ->
      Fmt.pf ppf "unsupported checkpoint map schema_version %d" v
  | `Shape_clash { Clash.target; against; expected; actual } ->
      Fmt.pf ppf "capture %S: per %a shape is %a, map says %a" target pp_view
        against pp_shape expected pp_shape actual
  | `Source_digest_mismatch { Source_digest.name; actual; expected } ->
      Fmt.pf ppf "source %S hashes to %a, the map pins %a" name
        Pt2_sha256.Digest.pp actual Pt2_sha256.Digest.pp expected
  | `Source_missing name -> Fmt.pf ppf "source %S was not supplied" name
  | `Source_size_mismatch { Size_clash.target; expected; actual } ->
      Fmt.pf ppf "source %S is %Ld bytes, the map pins %Ld" target actual
        expected
  | `Source_surplus name ->
      Fmt.pf ppf "source %S was supplied but the map declares no such file" name
  | `Stored_dtype { Stored_dtype.target; expected; actual } ->
      Fmt.pf ppf "capture %S: the checkpoint stores %s, the map needs %a" target
        actual Dtype.pp expected
  | `Surplus_tensor target ->
      Fmt.pf ppf "map entry %S is not a capture of the graph" target
  | `Unknown_dtype code -> Fmt.pf ppf "unknown dtype code %S" code
  | `Unknown_source_file (target, file) ->
      Fmt.pf ppf "capture %S reads %S, which no source declares" target file
  | `Unmapped names ->
      Fmt.pf ppf "the map lists %d unmapped capture(s): %a" (List.length names)
        Fmt.(list ~sep:(any ", ") string)
        names
  | `Unsupported_conversion { Cast.target; from; to_ } ->
      Fmt.pf ppf "capture %S: conversion %a to %a is not implemented" target
        Dtype.pp from Dtype.pp to_
  | `Value_digest_clash { Clash.target; against; expected; actual } ->
      Fmt.pf ppf "capture %S: per %a digest is %a, map says %a" target pp_view
        against Pt2_sha256.Digest.pp expected Pt2_sha256.Digest.pp actual
  | `Value_length { Size_clash.target; expected; actual } ->
      Fmt.pf ppf "capture %S: prepared value is %Ld bytes, the shape needs %Ld"
        target actual expected
