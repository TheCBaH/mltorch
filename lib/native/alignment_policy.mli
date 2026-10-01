(* The alignment an arena slot is placed at, from its size alone (see .ai/ on
   the tensor arena). A slot up to one page starts on a cache line, so no two
   slots share one's first line; a larger slot starts on a page. Neither ever
   falls below the payload's own element alignment, nor below what the host
   asked for.

   Deterministic: the same policy, size and payload always give the same
   alignment, so a script that records it compares equal exactly when its
   sizes and policy do. *)

open Core.Storage_units

val cache_line_alignment : Byte_alignment.t
(** 64 bytes. *)

val page_alignment : Byte_alignment.t
(** 4096 bytes. *)

val page_size : Byte_size.t
(** [page_alignment] as a size: the largest slot aligned to a cache line. *)

val default : Byte_size.t -> payload_min:Byte_alignment.t -> Byte_alignment.t
(** [cache_line_alignment] up to and including [page_size], [page_alignment]
    above it; then the larger of that and [payload_min]. *)

type t
(** A policy: the size-based default, and the host's request, if any. *)

val standard : t
(** No host request: {!default}. *)

val with_host : Byte_alignment.t -> t
(** A host's request for every slot. It can only strengthen the default: a
    request below it changes nothing. *)

val host : t -> Byte_alignment.t option

val alignment :
  t -> Byte_size.t -> payload_min:Byte_alignment.t -> Byte_alignment.t
(** The larger of {!default} and the host's request. *)

val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit
