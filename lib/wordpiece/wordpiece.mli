(** BERT's tokenizer for ASCII text: the basic tokenizer (control-character
    cleanup, lower-casing, splitting on whitespace and ASCII punctuation)
    followed by greedy longest-match WordPiece against a vocabulary.

    It reproduces the documented algorithm of the uncased BERT tokenizer on
    ASCII input. Two spellings are outside it and refused rather than guessed:
    non-ASCII text (accent stripping and CJK handling need Unicode tables), and
    the special-token spellings ([[CLS]] and the like), which a reference
    tokenizer keeps whole where this one would split them. *)

type vocab

val vocab_of_lines : string list -> vocab
(** One token per line, the id being the line number from 0 (the layout of a
    BERT [vocab.txt]). *)

val size : vocab -> int
val token : vocab -> int -> string option
val id : vocab -> string -> int option

type error =
  | Non_ascii of char
  | Special_token_spelling of string
  | Missing_special of string

val pp_error : Format.formatter -> error -> unit

val words : string -> (string list, error) result
(** The basic tokenizer's output: lower-cased words and single punctuation
    characters, in order. *)

val pieces : vocab -> string -> string list
(** One word as WordPiece pieces, or [["[UNK]"]] when some part has no match or
    the word is longer than 100 characters. *)

type encoded = {
  attention_mask : int list;  (** 1 over real tokens, 0 over padding. *)
  input_ids : int list;
  tokens : string list;  (** The pieces, padding included. *)
  token_type_ids : int list;  (** All 0: a single segment. *)
}

val encode : vocab -> max_length:int -> string -> (encoded, error) result
(** [[CLS]] pieces [[SEP]], truncated to [max_length] (pieces are cut, the two
    specials kept) and padded with [[PAD]] to exactly [max_length]. Requires
    [max_length >= 2] and the vocabulary to carry [[CLS]], [[SEP]], [[PAD]] and
    [[UNK]]. *)
