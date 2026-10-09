(** Strict byte codecs for the two encodings a map embeds. Both reject what a
    lenient decoder would skip: a map is pinned by digest, so a second spelling
    of the same bytes is a defect, not a convenience. *)

val hex_decode : string -> string option
(** Lower-case pairs only; the empty string is [None] (a fill element has at
    least one byte). *)

val base64_decode : string -> string option
(** RFC 4648 standard alphabet, padded, canonical (unused trailing bits zero),
    no whitespace. The empty string decodes to the empty string. *)

val base64_decoded_length : string -> int option
(** The length {!base64_decode} would produce, without decoding; [None] when the
    string is not well-formed enough to tell (length not a multiple of four, or
    padding misplaced). Lets a caller bound the output before allocating it. *)
