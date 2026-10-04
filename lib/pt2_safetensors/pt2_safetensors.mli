(** Run a model whose graph and weight configs are committed as JSON, with the
    weights in a Hugging Face safetensors checkpoint instead of a [.pt2] zip.

    [models/safetensors.json] pins the checkpoint ([Checkpoint_map.Source]) and
    maps each captured tensor's config name to a checkpoint key, dtype and
    shape. {!of_parts} checks that map against the graph's configs and the
    checkpoint header up front, so a loaded tensor can only be the one the graph
    asked for; the result is a {!Pt2_archive.t} whose tensor bytes are zero-copy
    views into the checkpoint. *)

module Checkpoint_map : sig
  module Source : sig
    type t = {
      filename : string;
      repo_id : string;
      revision : string;
      sha256 : string;
      size : int64;
      url : string;
    }
  end

  module Entry : sig
    type t = { dtype : string; key : string; shape : int list }
  end

  type t = {
    schema_version : int;
    source : Source.t;
    tensors : Entry.t Schema_runtime.String_map.t;
        (** Captured-tensor config name -> checkpoint entry. *)
    unmapped : string list;
        (** Captured tensors the checkpoint does not hold (non-persistent
            buffers); any makes the model unrunnable from it. *)
  }

  val jsont : t Jsont.t
end

module Mismatch : sig
  (** What disagreed for one tensor. The three views of a tensor -- the map, the
      graph's weight config and the checkpoint header -- must all name the same
      dtype and shape, and the graph's tensor must be a dense row-major buffer
      at offset 0, which is all a checkpoint tensor can be. *)
  type t =
    | Checkpoint_dtype of { checkpoint : Safetensors.Dtype.t; map : string }
    | Checkpoint_shape of { checkpoint : int64 list; map : int list }
    | Graph_dtype of { graph : Pt2_dtype.t; map : string }
    | Graph_layout
    | Graph_shape of { graph : int list; map : int list }
end

module Source_mismatch : sig
  type t =
    | Sha256 of { actual : string option; expected : string }
        (** The downloaded blob's etag (a git-LFS file's sha256) is not the
            pinned one; [None] when the blob has no etag. *)
    | Size of { actual : int64; expected : int64 }
end

type error =
  [ `Map_decode of string
  | `Mismatch of string * Mismatch.t
  | `Missing_in_checkpoint of string * string
    (** config name, checkpoint key *)
  | `Schema_version of int
  | `Source_mismatch of Source_mismatch.t
  | `Unmapped_constants of string list
  | `Unmapped_tensor of string
  | Pt2_tensor.error ]

val pp_error : Format.formatter -> error -> unit

val map_of_string : string -> (Checkpoint_map.t, [> error ]) Err.t
(** Decode [safetensors.json] and check [schema_version = 1]. *)

val check_source :
  Checkpoint_map.Source.t ->
  etag:string option ->
  size:int64 ->
  (unit, [> error ]) Err.t
(** The checkpoint a download produced is the one [source] pins: its etag equals
    the pinned sha256 and its size the pinned size. Shared by the native cache
    and the JavaScript drivers, which differ only in where the bytes landed. *)

val check_graph :
  map:Checkpoint_map.t ->
  weights:Pytorch_weights_config.ModelWeightsConfig.t ->
  constants:Pytorch_weights_config.ModelWeightsConfig.t ->
  (unit, [> error ]) Err.t
(** The checks that need no checkpoint: nothing {!Checkpoint_map.t.unmapped},
    every captured tensor of the graph mapped, and the map's dtype and shape
    equal to the graph's, with the graph's tensor a dense row-major buffer. *)

val of_parts :
  map:Checkpoint_map.t ->
  program:Pytorch_types.ExportedProgram.t ->
  weights:Pytorch_weights_config.ModelWeightsConfig.t ->
  constants:Pytorch_weights_config.ModelWeightsConfig.t ->
  Safetensors.Memory.t ->
  (Pt2_archive.t, [> error ]) Err.t
(** Validate everything described above, then build the archive. Nothing is read
    from the checkpoint until a tensor is loaded. *)
