(** The producer's canonical JSON:
    [json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)],
    the form its configuration IDs hash. Members are sorted by name in
    code-point order, strings are raw UTF-8 with Python's escapes for the quote,
    the backslash and the control characters, and numbers are integers as
    written.

    A decoded number is a float, so its source spelling is lost: an integral
    value below 2^53 is printed as an integer, and any other number is refused
    rather than guessed. A source float such as [1.0] would print as [1], so the
    ID recomputed from it differs from the declared one and the configuration is
    refused there: the check fails closed. *)

type error =
  [ `Duplicate_member of string  (** A member name twice in one object. *)
  | `Inexact_number of float
    (** Not an integer, or too large to be one exactly. *) ]

val pp_error : error Fmt.t
val to_string : Jsont.json -> (string, [> error ]) Err.t
