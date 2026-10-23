(** The result record a subsystem guest leaves in DTCM, where the HTIF block
    would be, for the host to read over the SPI bridge.

    The subsystem has no HTIF, and its SPI bridge reads about a kilobyte a
    second, so slot bytes never travel back: the guest hashes each written slot
    ({!hash}) after the timed call and the host compares the hashes with its
    own. The record is {!size} bytes of little-endian 32-bit words:

    {v
    0   magic: "CNPUOUT1" (a result) or "CNPUTRP1" (a trap)
    8   layout digest
    12  result: entry status     trap: mcause
    16          mcycle (u64)           mepc
    20                                 mtval
    24          minstret (u64)
    32          hash count n
    36          n hashes, in written-slot order
    252 done: {!done_word}, stored last
    v} *)

val size : int
(** 256 bytes. *)

val done_offset : int
(** 252. The host polls the 16-byte line holding it. *)

val done_word : int32
(** ["DONE"] as a little-endian word. *)

val hashes_offset : int
(** 36. *)

val max_hashes : int
(** 54: the hashes that fit before the done word. *)

val magic_result : string
val magic_trap : string

val hash : string -> int32
(** FNV-1a over 32-bit words, folded: from [0x811C9DC5], each little-endian word
    [w] (a tail byte counts as a word of its own) makes [h]
    [(h xor w) * 0x01000193] mod 2{^ 32}, then [h xor (h lsr 15)]. Without the
    fold a word's top bit never reaches the others, so two sign flips cancel. *)

module Result : sig
  type t = {
    digest : int32;
    hashes : int32 list;
    mcycle : int64;
    minstret : int64;
    status : int32;
  }
end

module Trap : sig
  type t = { mcause : int32; mepc : int32; mtval : int32 }
end

type t = Result of Result.t | Trap of Trap.t

module Malformed : sig
  type t =
    | Hash_count of int  (** More hashes than fit. *)
    | Magic of string
    | Size of int  (** Not {!size} bytes. *)
end

val is_done : string -> int -> bool
(** [is_done line offset]: the done word stands at [offset] of [line]. *)

val decode : string -> (t, Malformed.t) result
(** A record whose done word is set. *)

val pp_malformed : Malformed.t Fmt.t
