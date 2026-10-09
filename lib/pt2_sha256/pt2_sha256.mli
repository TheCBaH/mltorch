(** SHA-256 (FIPS 180-4), incremental, over strings and bigstrings.

    Pure OCaml with [Int32] words and an [int64] length, so a digest is the same
    on native and js_of_ocaml. The input position and length of a bigstring are
    [int]s: callers on js_of_ocaml bound them by [Sys.max_string_length], the
    same ceiling as the bigstrings themselves. *)

type bigstring =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

module Digest : sig
  type t
  (** Thirty-two bytes. *)

  val equal : t -> t -> bool
  val to_hex : t -> string

  val of_hex : string -> t option
  (** Exactly 64 lower-case hexadecimal characters, the form every pin in a
      checkpoint map takes; anything else is [None]. *)

  val pp : t Fmt.t
  (** The hex form. *)
end

type t
(** A running hash. Mutable; {!finish} consumes it. *)

val create : unit -> t

val add_string : t -> ?pos:int -> ?len:int -> string -> unit
(** Raises [Invalid_argument] on a range outside the string, or after {!finish}.
*)

val add_bigstring : t -> ?pos:int -> ?len:int -> bigstring -> unit
(** As {!add_string}. *)

val finish : t -> Digest.t
(** Pads and returns the digest. The state is unusable afterwards. *)

val string : string -> Digest.t
val bigstring : bigstring -> Digest.t
