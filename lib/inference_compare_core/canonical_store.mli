(** The coordinator-owned store of canonical values. A value enters once, by its
    owner unit or from the pinned initial payloads, with the number of reads the
    plan counted for it; it is dropped as soon as the last read has finished.
    Values are immutable: readers receive the stored value itself and never a
    writable view, and nothing the store holds is shared with an engine. *)

open Inference_compare_schema

type t

val create : unit -> t

val add :
  t ->
  Id.Canonical_value.t ->
  instance:Id.Value_instance.t ->
  uses:int ->
  Inference_compare_codecs.Logical_value.t ->
  (unit, [> `Already_present ]) result
(** Takes ownership. A value with no reads is dropped immediately. *)

val get :
  t -> Id.Canonical_value.t -> Inference_compare_codecs.Logical_value.t option

val instance : t -> Id.Canonical_value.t -> Id.Value_instance.t option
val uses_left : t -> Id.Canonical_value.t -> int option

val finish_uses : t -> Id.Canonical_value.t -> int -> unit
(** Records finished reads; drops the value at zero. Raises [Invalid_argument]
    when more reads finish than were counted. *)

val live : t -> Id.Canonical_value.t list

val clear : t -> unit
(** Drops everything still held. *)
