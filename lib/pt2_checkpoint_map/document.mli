(** A version-2 checkpoint map: for every tensor a graph captures, where its
    bytes come from and what they must hash to. Everything here is decided from
    the map's own bytes; checking it against a graph is {!Validate}, and reading
    the sources it names is the host's. *)

module Pin : sig
  type t = {
    name : string;
        (** A file name, never a path: no separator, not [.] or [..]. *)
    sha256 : Pt2_sha256.Digest.t;
    size : int64;  (** At least one byte. *)
    url : string;  (** [https://] only. *)
  }
end

module Derived : sig
  type t = {
    file : string;
    repo_id : string;
    revision : string;
    sha256 : Pt2_sha256.Digest.t;
    tool : string;
  }
  (** The upstream file a converted checkpoint was made from. *)
end

module Upstream : sig
  type t = { repo_id : string; revision : string }
  (** [revision] is a 40-digit commit. *)
end

module Source : sig
  type provenance = Converted of Derived.t | Upstream of Upstream.t
  type t = { pin : Pin.t; provenance : provenance }
end

module Origin : sig
  type convert = Cast of { from : Dtype.t; to_ : Dtype.t } | Identity

  module Checkpoint : sig
    type t = {
      convert : convert;
      file : string;  (** A {!Pin.t.name} of a checkpoint source. *)
      key : string;
      tied_aliases : string list;  (** Information only. *)
    }
  end

  type t =
    | Checkpoint of Checkpoint.t
    | Empty
    | Fill of string
        (** One element's little-endian bytes, exactly the dtype's width. *)
    | Inline of string  (** The whole value's bytes, decoded. *)
    | Pack of string  (** A key in the graph-owned file. *)
end

module Entry : sig
  type t = {
    dtype : Dtype.t;  (** The final dtype, after any conversion. *)
    origin : Origin.t;
    sha256 : Pt2_sha256.Digest.t;  (** Of the final raw little-endian bytes. *)
    shape : int64 list;  (** Extents of at least zero; scalar is [[]]. *)
  }

  val element_count : t -> int64
  (** The product of the extents. A decoded entry's product and byte size are
      already proved to fit, so this cannot overflow on one. *)

  val byte_count : t -> int64
end

type t = {
  artifact_id : string;
  checkpoint_files : Source.t list;  (** Declared order; names are unique. *)
  graph_owned : Pin.t option;
  graph_sha256 : Pt2_sha256.Digest.t;  (** Of the raw [model.json] bytes. *)
  model_id : string;
  tensors : Entry.t Schema_runtime.String_map.t;
      (** Keyed by capture target, which is also the key of the graph's weights
          and constants configs. *)
}

val pin_of_fields :
  name:string ->
  sha256:string ->
  size:int64 ->
  url:string ->
  (Pin.t, [> Fault.error ]) Err.t
(** The checks every pinned file passes: a bare file name, a 64-digit lower-case
    digest, at least one byte, an [https://] URL. Shared by the layers that pin
    files (the map, the publication index, the manifests). *)

val supported_schema_version : int

val of_string : ?limits:Limits.t -> string -> (t, [> Fault.error ]) Err.t
(** Decode and check the map against itself: schema version, empty [unmapped],
    unique names, well-formed pins, dtype/shape/byte-size bounds before any
    allocation, and each origin consistent with its entry (an [empty] has no
    elements, a [fill] element is one dtype wide, [inline] bytes are exactly the
    value's size, a cast produces the entry's dtype, a [checkpoint] or [pack]
    names a declared source). Unknown members are rejected where the schema
    forbids them. A conversion is only parsed here; whether it is implemented is
    {!Validate.supported_conversions}. *)

val find_source : t -> string -> Source.t option
(** The checkpoint source of that file name. *)
