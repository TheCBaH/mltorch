(* The lowering inspector's answer, as the protocol carries it.

   A PROTOCOL-LEVEL record: it names no backend type, so the worker maps
   whatever the lowering view produced onto it and the browser decodes it
   without the lowering libraries. What an answer contains is therefore data
   with a tag, never prose: a refusal is [{kind; data}] and so is each note, and
   the page chooses the words.

   The text is bounded by [Me_limits.Limits.max_artifact_bytes] at construction,
   so a [t] cannot exceed the ceiling it was built under, and [truncated]
   records the cut rather than leaving it to be inferred from a length.

   See the `.ai/` directory for the lowering inspector design. *)

module Tag : sig
  type t = { kind : string; data : string list }
  (** A stable tag name and its arguments, each rendered as text. [kind] is the
      producer's own vocabulary (a refusal reason, a note category); this layer
      does not enumerate it, only carries it. *)

  val jsont : t Jsont.t
end

module Body : sig
  (* @alphabetical *)
  type t =
    | Refused of Tag.t  (** no artifact, and why *)
    | Text of { language : string; text : string; truncated : bool }
        (** [language] names the artifact's syntax ("js", "c", "wat", ...) so
            the page can choose a highlighter. *)

  val jsont : t Jsont.t
end

type t = { key : Lowering_key.t; body : Body.t; notes : Tag.t list }

val text : limits:Me_limits.Limits.t -> language:string -> string -> Body.t
(** [Text] of at most [limits.max_artifact_bytes] bytes. A longer text is cut at
    the last UTF-8 boundary at or before the ceiling, with [truncated = true]; a
    text that fits is unchanged and [truncated = false]. *)

val jsont : t Jsont.t
