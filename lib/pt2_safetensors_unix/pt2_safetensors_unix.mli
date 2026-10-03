(** [open_dir model_dir] opens a model from its committed JSON, with the weights
    mapped from the pinned Hub checkpoint.

    [model_dir] is a [models/<name>] directory of the
    devcontainer.pytorch-image-models submodule: [models/model.json],
    [models/safetensors.json] and [data/{weights,constants}/*_config.json]. The
    checkpoint is the one [safetensors.json] pins, resolved at its commit, so a
    cached copy needs no network. *)

module Source_mismatch : sig
  type t =
    | Sha256 of { actual : string option; expected : string }
        (** The cached blob's etag (a git-LFS file's sha256) is not the pinned
            one; [None] when the blob has no etag. *)
    | Size of { actual : int64; expected : int64 }
end

type error =
  [ Pt2_archive.error
  | Pt2_safetensors.error
  | `Checkpoint of Hf_hub_safetensors.error
  | `Source of Hf_hub.Error.t  (** [source] in [safetensors.json] is invalid *)
  | `Source_mismatch of Source_mismatch.t ]

val pp_error : Format.formatter -> error -> unit

val open_dir :
  ?env:Hf_hub.Env.t ->
  ?http:Hf_hub_unix.http ->
  string ->
  (Pt2_archive.t, [> error ]) Err.t
