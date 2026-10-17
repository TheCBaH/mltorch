(** Where a guest image and its slots go in a profile's memory.

    - Executable segments go to ITCM, in order. There is no spill: a program
      larger than ITCM is refused.
    - DTCM holds, from its base and in this order: the data segments, the HTIF
      block (four 8-byte words, each on its own 64 bytes), the HTIF packet
      (eight 8-byte words), the staging buffer, the trap scratch, then slots.
      The stack takes the top of DTCM.
    - A slot that does not fit in DTCM goes to the profile's writable RAM
      region. A slot is never split.

    The digest is 32 bits of a SHA-256 over the configuration ID and every
    address and size, so a host and a guest built for different layouts cannot
    exchange a frame. *)

module Segment : sig
  type t = { align : int64; executable : bool; name : string; size : int64 }
  (** [size] counts the zero-filled tail. *)
end

module Placement : sig
  type t =
    | In  (** Read from the input frame before the entry. *)
    | In_out  (** Read before the entry and written to the result after. *)
    | Out  (** Zeroed before the entry, written to the result after. *)
    | Scratch  (** Zeroed before the entry. *)

  val to_string : t -> string

  val read : t -> bool
  (** Filled from the input frame: [In], [In_out]. *)

  val written : t -> bool
  (** Part of the result frame: [In_out], [Out]. *)
end

module Slot : sig
  type t = { align : int64; placement : Placement.t; size : int64 }
end

module Params : sig
  type t = { stack : int64; staging : int64 }

  val default : t
  (** 4 KiB of stack, 4 KiB of staging. *)
end

module Placed_slot : sig
  type t = {
    address : int64;
    htif_reachable : bool;
        (** HTIF may read or write it directly; otherwise it is staged. *)
    region : string;
    slot : Slot.t;
  }
end

type t = {
  digest : int32;
  htif : int64;
      (** [tohost]; [fromhost], [tohost_ready] and [fromhost_ready] follow at
          64-byte steps. *)
  packet : int64;
  segments : (string * int64) list;  (** For [Image.bind_image]. *)
  slots : Placed_slot.t list;
  stack_top : int64;
  staging : int64;
  staging_size : int64;
  trap : int64;
}

val htif_size : int64
(** The HTIF block, 256 bytes. *)

val packet_size : int64
val trap_size : int64

module Over_capacity : sig
  type t = { available : int64; needed : int64; region : string }
end

type error =
  [ `Missing_region of Coral_profile.Role.t
    (** The profile has no ITCM, no DTCM, or no writable RAM where a slot needs
        it. *)
  | `Over_capacity of Over_capacity.t ]

val pp_error : error Fmt.t

val place :
  ?params:Params.t ->
  Coral_profile.t ->
  Segment.t list ->
  Slot.t list ->
  (t, [> error ]) Err.t

val pp : t Fmt.t
