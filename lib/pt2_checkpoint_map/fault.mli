(** What can be wrong with a checkpoint map or the sources it names. Payloads
    carry the data that identifies the defect -- capture target, both digests,
    both shapes -- never a rendered message; {!pp} renders them. *)

module Pair : sig
  type 'a t = { actual : 'a; expected : 'a }
  (** Two values that had to be equal. *)
end

type view =
  | Computed
  | Config
  | Inventory
  | Signature
  | Source
      (** What a capture's expected value was read from: the bytes just
          prepared, a payload config, the producer's [captures.json], the graph
          signature or a source file's own header. *)

module Clash : sig
  type 'a t = { actual : 'a; against : view; expected : 'a; target : string }
  (** One capture's disagreeing views: [expected] is the graph-side or earlier
      view, [actual] what the map says. *)
end

module Cast : sig
  type t = { from : Dtype.t; target : string; to_ : Dtype.t }
end

module Size_clash : sig
  type t = { actual : int64; expected : int64; target : string }
end

module Key_ref : sig
  type t = { file : string; key : string; target : string }
end

module Source_digest : sig
  type t = {
    actual : Pt2_sha256.Digest.t;
    expected : Pt2_sha256.Digest.t;
    name : string;
  }
end

module Stored_dtype : sig
  type t = { actual : string; expected : Dtype.t; target : string }
  (** [actual] is the checkpoint's own dtype code, which need not be one of the
      map's. *)
end

module Over_limit : sig
  type t = { actual : int64; limit : int64; what : Limits.which }
end

type domain = Capture | Checkpoint_file | Pack_key

(** What a duplicated name was a name of. *)
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

val pp_error : error Fmt.t
