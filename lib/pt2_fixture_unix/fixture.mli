(** A released fixture opened for execution: the graph and configs from its
    bundle, every checkpoint capture prepared and proved against the map, all
    behind the existing {!Pt2_archive.t} interface.

    Layers, each checked against the one above: the cohort pins (consumer) →
    publication index → manifest → archive members → graph/contract/map digests
    → the map's own source pins → each source file's bytes → each capture's
    final value. Nothing is executed until every layer has passed. *)

type error =
  [ Fault.error
  | Pt2_archive.error
  | Pt2_checkpoint_map_unix.error
  | `Contract_decode of string ]

val pp_error : error Fmt.t

type t = {
  archive : Pt2_archive.t;
      (** Graph, weight/constant configs and prepared captures; it has no [.pt2]
          zip behind it. *)
  bundle : Bundle.t;
  captures : Pt2_checkpoint_map.Prepare.t;
  document : Pt2_checkpoint_map.Document.t;
}

val open_ :
  ?limits:Pt2_checkpoint_map.Limits.t ->
  Bundle.config ->
  Pt2_fixture.Cohort.t ->
  Pt2_fixture.Cohort.entry ->
  (t, [> error ]) Err.t
(** Resolve the bundle ({!Bundle.ensure}), then fetch and verify the sources its
    map declares and prepare every capture. Offline when the config has no
    transport: everything must already be in the cache. *)

val of_bundle :
  ?limits:Pt2_checkpoint_map.Limits.t ->
  Bundle.config ->
  Bundle.t ->
  (t, [> error ]) Err.t
(** As {!open_} from an already verified bundle directory. *)
