(** [open_dir model_dir] opens a model from its committed JSON, with the weights
    mapped from the pinned Hub checkpoint.

    [model_dir] is a [models/<name>] directory of the
    devcontainer.pytorch-image-models submodule: [models/model.json],
    [models/safetensors.json] and [data/{weights,constants}/*_config.json]. The
    checkpoint is the one [safetensors.json] pins, resolved at its commit, so a
    cached copy needs no network. *)

type error =
  [ Pt2_archive.error
  | Pt2_safetensors.error
  | `Checkpoint of Hf_hub_safetensors.error
  | `Source of Hf_hub.Error.t  (** [source] in [safetensors.json] is invalid *)
  ]

val pp_error : Format.formatter -> error -> unit

val open_dir :
  ?env:Hf_hub.Env.t ->
  ?http:Hf_hub_unix.http ->
  string ->
  (Pt2_archive.t, [> error ]) Err.t

val checkpoint_path :
  ?env:Hf_hub.Env.t ->
  ?http:Hf_hub_unix.http ->
  string ->
  (string, [> error ]) Err.t
(** The path of the pinned checkpoint in the cache, fetched if absent and
    checked against the pin like {!open_dir}. For a consumer that cannot use the
    Hub driver, such as a js_of_ocaml run handed the file. *)
