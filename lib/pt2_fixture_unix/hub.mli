(** Checkpoint sources that are files of a Hugging Face Hub repository, fetched
    through [hf-hub] into its huggingface_hub-layout cache ([HF_HUB_CACHE]), not
    into the content-addressed fixture cache.

    A source takes this route only when the map says it {e is} the upstream file
    ([Upstream] provenance) and its pinned URL is exactly the Hub resolve URL of
    that repository, revision and file name. A converted source (a producer's
    own release asset) or a non-Hub URL keeps the transport route. The pin is
    still the authority: the blob's size and SHA-256 must equal it, whatever the
    Hub answered. *)

type t

val make :
  ?http:Hf_hub_unix.http -> ?hasher:Hf_hub_unix.hasher -> Hf_hub.Env.t -> t
(** [http] and [hasher] default to the [curl] and [sha256sum] drivers. *)

val of_environment : unit -> t
(** [HF_HUB_CACHE], [HF_HOME], [HF_TOKEN], [HF_HUB_OFFLINE], [HF_ENDPOINT]. *)

val env : t -> Hf_hub.Env.t

val ensure :
  ?hash:Cache.file_hasher ->
  online:bool ->
  t ->
  Pt2_checkpoint_map.Document.Source.t ->
  (string option, [> Fault.error ]) Err.t
(** [None] when the source is not a Hub file (use the transport). Otherwise the
    path of the verified cached file. With [online:false] (or an offline
    environment) nothing is requested: a missing file is [`Offline]. A URL that
    claims the Hub endpoint but disagrees with the source's repository, revision
    or name is [`Hub_url_mismatch]. *)
