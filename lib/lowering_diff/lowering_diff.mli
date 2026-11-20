(** Line diff between two texts, for comparing two lowerings of one node.

    Lines split on ['\n'] only (CRLF input keeps its ['\r'] in the line); a
    final newline does not make an extra empty line. *)

(* @alphabetical *)
type op = Delete of string | Equal of string | Insert of string

type t = {
  ops : op list;  (** the edit script, in order *)
  minimal : bool;
      (** [false] when the inputs differed by more edits than the search bound:
          [ops] is then a correct but unminimised delete-all, insert-all script
      *)
}

val lines : string -> string list

val diff : string -> string -> t
(** [diff old_text new_text]; applying [ops] to [old_text]'s lines gives
    [new_text]'s. *)

val identical : t -> bool
(** no [Delete] or [Insert] *)

(** A run of changes with its surrounding context, 1-based line starts as
    unified diff prints them (a side with no lines starts at the line before).
*)
module Hunk : sig
  type t = {
    old_start : int;
    old_count : int;
    new_start : int;
    new_count : int;
    ops : op list;
  }
end

val hunks : context:int -> t -> Hunk.t list
(** Empty when {!identical}. *)

val unified : context:int -> old_label:string -> new_label:string -> t -> string
(** [--- old]/[+++ new] header and [@@] hunks; the empty string when
    {!identical}. *)
