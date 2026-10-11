(** Strict schema-v1 wire helpers over the generic JSON value.

    Records are objects, variants are objects carrying a ["tag"] member, 64-bit
    counts and indices are decimal strings, and an absent optional field is an
    explicit [null]. Decoding rejects duplicate keys, unknown members, unknown
    tags and malformed numbers; nothing here supplies a default for a missing
    field. A decoder reports where in the document it failed. *)

type json = Jsont.json

module Constraint : sig
  (** A closed set of structural rules a decoder or validator enforces. *)
  type t =
    | Count_overflow
    | Duplicate_element
    | Empty_list
    | Malformed_document
    | Negative_value
    | Nonfinite_number
    | Out_of_range
end

type sort = [ `Array | `Bool | `Null | `Number | `Object | `String ]

type reason =
  | Constraint of Constraint.t
  | Duplicate_key of string
  | Expected of sort
  | Invalid_decimal of string
  | Invalid_digest of Digest_error.t
  | Invalid_text of Text_error.t
  | Missing_member of string
  | Unknown_member of string
  | Unknown_tag of { variant : string; tag : string }
  | Unsupported_schema_version of string

type error = { path : string list; reason : reason }
(** [path] runs from the document root to the failing member; list positions
    appear as decimal indices. *)

val pp_error : Format.formatter -> error -> unit
val fail : reason -> ('a, error) Err.t

val at : string -> ('a, error) Err.t -> ('a, error) Err.t
(** [at component r] prefixes [component] to the path of a failure in [r]. *)

val ( let* ) :
  ('a, error) Err.t -> ('a -> ('b, error) Err.t) -> ('b, error) Err.t

val return : 'a -> ('a, error) Err.t

type 'a decoder = json -> ('a, error) Err.t
type 'a encoder = 'a -> json

(** {1 Scalars} *)

val string : string decoder
val bool : bool decoder

val int64 : int64 decoder
(** A decimal string; [-9223372036854775808] and up to its positive maximum. *)

val int : int decoder
(** A decimal string that fits the native [int] on every backend (31 bits). *)

val float : float decoder
(** A finite JSON number. *)

val enc_string : string encoder
val enc_bool : bool encoder
val enc_int64 : int64 encoder
val enc_int : int encoder

val enc_float : float encoder
(** Raises [Invalid_argument] on a nonfinite value: a measurement that can be
    nonfinite must be tagged, never encoded as a number. *)

val enc_null : json

(** Decoder and encoder for a validated-text module. *)
module Text (T : Text_value.S) : sig
  val decode : T.t decoder
  val encode : T.t encoder
end

(** {1 Containers} *)

val list : 'a decoder -> 'a list decoder
val enc_list : 'a encoder -> 'a list encoder

val option : 'a decoder -> 'a option decoder
(** [null] is [None]; every other value goes to the element decoder. *)

val enc_option : 'a encoder -> 'a option encoder

val pair : 'a decoder -> 'b decoder -> ('a * 'b) decoder
(** A two-element array. *)

val enc_pair : 'a encoder -> 'b encoder -> ('a * 'b) encoder

(** {1 Objects} *)

module Obj : sig
  type t

  val required : t -> string -> 'a decoder -> ('a, error) Err.t

  val optional : t -> string -> 'a decoder -> ('a option, error) Err.t
  (** The member must be present; [null] decodes to [None]. *)

  val tag : t -> (string, error) Err.t
end

val record : (Obj.t -> ('a, error) Err.t) -> 'a decoder
(** Decodes an object, runs the reader, then rejects any member the reader did
    not consume. *)

val variant : (string -> Obj.t -> ('a, error) Err.t) -> 'a decoder
(** [variant f] reads the ["tag"] member and passes it to [f] with the remaining
    members; [f] returns {!unknown_tag} for a tag it does not know. *)

val unknown_tag : kind:string -> string -> ('a, error) Err.t
val obj : (string * json) list -> json
val tagged : string -> (string * json) list -> json
val field : string -> 'a encoder -> 'a -> string * json

(** {1 Schema version and canonical text} *)

val check_version : Obj.t -> expected:string -> (unit, error) Err.t

val canonical_string : json -> string
(** Compact text with object members sorted by name and finite numbers printed
    with enough digits to roundtrip: the preimage of a digest. *)

(** {1 Closed enumerations} *)

val enum : kind:string -> ('a * string) list -> 'a decoder * 'a encoder
(** A constant-constructor variant carried as a bare string. The association
    list must cover every constructor; the encoder raises [Invalid_argument] on
    one it does not list. *)

val nonneg_float : float decoder
(** A finite number that is not negative. *)

val nonneg_int : int decoder
val nonneg_int64 : int64 decoder

val unique : key:('a -> string) -> 'a list decoder -> 'a list decoder
(** Rejects a list in which two elements share a key, reporting
    [Duplicate_element] at the later position. Order is preserved. *)
