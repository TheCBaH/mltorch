(** The subsystem's DMA engine ([bus.DmaEngine]) as data: its registers, and
    descriptor chains built by the host and loaded with the image.

    One channel, one outstanding bus transaction: each beat is read, then
    written. A chain of 32-byte, 32-byte-aligned descriptors runs from one
    start; [STATUS.done] is set when the whole chain is done, and a start clears
    it. Transfers are one-dimensional: a strided panel takes a descriptor per
    row. *)

val base : int64
(** [0x4005_0000]. *)

module Register : sig
  (** Alphabetical. *)
  type t = Ctrl | Cur_desc | Desc_addr | Status | Xfer_remain

  val offset : t -> int
end

val ctrl_start : int
(** [CTRL] = enable and start, 3. *)

val status_done : int
(** [STATUS] bit 1. *)

val status_error : int
(** [STATUS] bit 2; the code is in bits 7:4. *)

module Beat : sig
  (** The bytes a beat moves. In size order, which is the encoding's. *)
  type t = B1 | B2 | B4 | B8 | B16

  val bytes : t -> int
  val log2 : t -> int
  val of_bytes : int -> t option
end

module Poll : sig
  type t = { address : int64; mask : int32; value : int32 }
  (** Before each data beat the engine reads the 32-bit word at [address] and
      waits until [word land mask = value]: on a one-beat descriptor, a gate
      that holds the rest of the chain until the core writes the word. *)
end

module Descriptor : sig
  type t = {
    beat : Beat.t;
    dst : int64;
    dst_fixed : bool;  (** The destination does not advance: a FIFO. *)
    length : int64;  (** Bytes, a multiple of the beat. *)
    poll : Poll.t option;
    src : int64;
    src_fixed : bool;
  }

  val size : int
  (** 32 bytes, which is also the alignment. *)

  val copy :
    ?beat:Beat.t -> ?poll:Poll.t -> src:int64 -> dst:int64 -> int64 -> t
  (** Memory to memory, both advancing; [beat] defaults to [B16], no polling by
      default. *)
end

val len_flags : Descriptor.t -> int32
(** Length in bits 23:0, beat log2 in 26:24, [src_fixed] 27, [dst_fixed] 28,
    [poll] 29. *)

module Misaligned : sig
  type t = { address : int64; alignment : int }
end

module Unreachable : sig
  type t = { address : int64; length : int64 }
end

type error =
  [ `Dma_length of int64
    (** Zero, over 16 MiB - 1, or not a multiple of the beat. *)
  | `Dma_misaligned of Misaligned.t
  | `Dma_unreachable of Unreachable.t
    (** Not within one region the engine reaches. *) ]

val pp_error : error Fmt.t

val check : Coral_profile.t -> Descriptor.t -> (unit, [> error ]) Err.t
(** Lengths, beat alignment of both addresses, and both ranges each within one
    [dma_reachable] region; a polled word, 4-byte aligned, nonzero and
    reachable. *)

val chain :
  Coral_profile.t -> at:int64 -> Descriptor.t list -> (string, [> error ]) Err.t
(** The descriptors' bytes, linked in order, for loading at [at] (32-byte
    aligned, itself within a reachable region): each [next] is the following
    descriptor's address, the last's 0. Every descriptor is {!check}ed. *)
