(** The producer's publication index for one release: every artifact's archive,
    manifest and released source files, by pin. It is believed only where it
    agrees with the {!Cohort} that pinned its bytes. *)

module Pin = Pt2_checkpoint_map.Document.Pin

type entry = {
  archive : Pin.t;
  artifact_id : string;
  graph_sha256 : Pt2_sha256.Digest.t;
  manifest : Pin.t;
  sources : Pin.t list;  (** The [v2:] assets: converted checkpoint files. *)
}

type t = { entries : entry list; release_tag : string; repository : string }

val of_string : string -> (t, [> Fault.error ]) Err.t

val check : Cohort.t -> t -> Cohort.entry -> (entry, [> Fault.error ]) Err.t
(** The publication is the cohort's release, lists the cohort's artifact, and
    gives the same archive and manifest pins and graph digest. Returns the
    publication's entry. *)
