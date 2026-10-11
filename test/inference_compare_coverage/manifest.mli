(** The coverage manifest: rows of
    [id | left | right | environment | expect | checks], with [#] comments (see
    the file's own header). *)

type expect =
  | Agree
  | Disagree
  | Invalid  (** exit 2: the pair is not a pair *)
  | Refused of string  (** exit 3, the text in the output *)

type check = Has of string | Lacks of string

type row = {
  id : string;
  left : string;
  right : string;
  env : (string * string) list;
  expect : expect;
  checks : check list;
  line : int;
}

val parse : string -> (row list, string) result
(** Rejects a malformed row, an unknown [expect], and a duplicate id, naming the
    line. *)

val words : row -> string list
(** The view and every configuration word of the row's two variants. *)
