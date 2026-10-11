(** Where a model comes from, told by its name: a timm model of the producer
    submodule (a directory with [models/model.json] and a weight map) or a
    released Transformers fixture of a cohort. *)

type t =
  | Fixture of { id : string; json : string option }
      (** [json] is the graph in the fixture cache, [None] until it is cached.
      *)
  | Timm of { name : string; dir : string; json : string }

type dirs = { cache : string; cohort : string; models : string }
(** The timm models directory, the cohort file and the fixture cache. *)

val resolve : dirs -> string -> (t, string) result
(** A name that is a directory of timm models with a [models/model.json] is that
    model. Otherwise it is a cohort artifact: an exact id, or the dotted words
    that single one out ([smolvlm-256m.connector]) -- each must be one
    [/]-separated component of exactly one id. An unknown or ambiguous name is
    an error listing the candidates. *)

val checkpoint : string -> (string, string) result
(** The path of a timm model's pinned Hub checkpoint, fetched into
    [HF_HUB_CACHE] if absent ([HF_HUB_OFFLINE=1] forbids the request). *)
