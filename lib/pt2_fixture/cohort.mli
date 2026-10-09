(** The consumer's own record of which released fixtures it trusts, and the root
    of the chain: it pins the publication index by digest, and each artifact's
    archive, manifest, graph, contract, map and source files. The producer's
    documents are only believed where they agree with it. *)

module Pin = Pt2_checkpoint_map.Document.Pin

type entry = {
  artifact_id : string;
  archive : Pin.t;
  cases : string list;  (** Case ids, in contract order. *)
  contract_sha256 : Pt2_sha256.Digest.t;
  graph_sha256 : Pt2_sha256.Digest.t;
  manifest : Pin.t;
  map_member : string;  (** The map's member name inside the archive. *)
  map_sha256 : Pt2_sha256.Digest.t;
  role : string;  (** What the cohort uses it for; informational. *)
  sources : Pin.t list;
      (** The checkpoint files the map must declare, by name. *)
}

type t = {
  entries : entry list;
  publication : Pin.t;
  release_producer_commit : string;
  release_tag : string;
  repository : string;
}

val of_string : string -> (t, [> Fault.error ]) Err.t
val find : t -> string -> entry option
