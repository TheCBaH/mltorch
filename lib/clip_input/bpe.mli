(** CLIP's byte-pair-encoding tokenizer for ASCII text, driven by the
    [tokenizer.json] a CLIP repository ships: the vocabulary, the ranked merges
    and the [</w>] end-of-word suffix.

    The pipeline is the file's own for ASCII input: whitespace collapsed,
    lower-cased, split by the pattern
    [<|startoftext|> | <|endoftext|> | 's | 't | 're | 've | 'm | 'll | 'd |
     letters+ | one digit | punctuation+], each piece merged by lowest-ranked
    pair. Non-ASCII text, control characters and the two special-token spellings
    are refused rather than guessed. *)

type t

val of_json : string -> (t, string) result
(** A tokenizer from the text of a [tokenizer.json]. *)

type encoded = { attention_mask : int list; input_ids : int list }

val encode : t -> max_length:int -> string -> (encoded, string) result
(** [<|startoftext|>] pieces [<|endoftext|>], the pieces cut to fit
    [max_length], padded with [<|endoftext|>] (the pad token) and masked 0. *)

val pieces : t -> string -> (string list, string) result
(** The pattern's pieces of an ASCII string, lower-cased, in order. *)

val bpe : t -> string -> string list
(** One piece merged to vocabulary tokens (before id lookup). *)
