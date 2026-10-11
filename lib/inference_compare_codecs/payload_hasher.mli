(** The streaming payload-digest seam. An algorithm is registered by its typed
    id; a state accepts bounded chunks and is consumed by {!finish}. The public
    digest type is {!Inference_compare_schema.Digest.Payload.t}, which carries
    the algorithm: which backend computes it is a registration choice that no
    record's type reflects. *)

open Inference_compare_schema

module type BACKEND = sig
  type state

  val algorithm : Id.Digest_algorithm.t
  val create : unit -> state
  val update : state -> string -> pos:int -> len:int -> unit

  val finish : state -> string
  (** The digest as canonical lowercase hexadecimal. *)
end

module Registry : sig
  type t

  val empty : t

  val register :
    t ->
    (module BACKEND) ->
    (t, [> `Duplicate_algorithm of Id.Digest_algorithm.t ]) Err.t

  val algorithms : t -> Id.Digest_algorithm.t list
  (** In registration order. *)
end

type error =
  | Finished
  | Invalid_range of { length : int; pos : int; len : int }
  | Unknown_algorithm of Id.Digest_algorithm.t

val pp_error : Format.formatter -> error -> unit

module State : sig
  type t
end

val create : Registry.t -> Id.Digest_algorithm.t -> (State.t, error) Err.t

val update : State.t -> string -> pos:int -> len:int -> (unit, error) Err.t
(** Fails once the state has been finished. *)

val update_string : State.t -> string -> (unit, error) Err.t

val finish : State.t -> (Digest.Payload.t, error) Err.t
(** Consumes the state: a second [finish] or any later [update] fails. *)

val digest_string :
  Registry.t ->
  Id.Digest_algorithm.t ->
  string ->
  (Digest.Payload.t, error) Err.t

val content_path : Digest.Payload.t -> string
(** [algorithm/hex]: two algorithms never share a content-addressed path, even
    for equal hex. *)
