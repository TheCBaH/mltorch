(** The producer's per-artifact manifest: what the archive contains, member by
    member, and which map it carries. Pinned by the cohort (and the publication
    index) and itself the pin of every file in the archive. *)

module Pin = Pt2_checkpoint_map.Document.Pin

type member = { sha256 : Pt2_sha256.Digest.t; size : int64 }

type t = {
  archive : member * string;  (** Digest and size of the archive; its name. *)
  artifact_id : string;
  cases : string list;
  contract_sha256 : Pt2_sha256.Digest.t;
  graph_sha256 : Pt2_sha256.Digest.t;
  map_assets : Pin.t list;  (** Released checkpoint files the map names. *)
  map_member : string;
  members : member Schema_runtime.String_map.t;  (** By archive member name. *)
}

val of_string : string -> (t, [> Fault.error ]) Err.t
(** Requires a slim bundle: [payload] is [null]. *)

val check : Cohort.entry -> t -> (unit, [> Fault.error ]) Err.t
(** The manifest is the cohort's artifact: same id, graph, contract, case list
    and archive pin; its map member is the cohort's, with the cohort's digest;
    and every released source it lists is pinned identically by the cohort. *)
