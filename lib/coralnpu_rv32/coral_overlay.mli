(** Code that runs from a slot of the instruction memory (ITCM) and is loaded
    into it by the DMA when it is needed.

    An overlay is a set of modules with one entry symbol, which may call and
    address what the resident image exports. It is linked on its own for one
    slot and one data address, so its bytes are only right for the slot it was
    linked for and the resident image it imports from. Its [.data] and [.bss] go
    to a data region in DTCM that every overlay shares; whoever loads an overlay
    must zero its [.bss] when another overlay has used the region since, as
    kernels expect their zero-initialised buffers to be zero on entry. *)

module Overlay : sig
  type t = {
    name : string;
    entry : string;  (** An exported symbol of [modules]. *)
    modules : Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_ list;
  }
end

module Linked : sig
  type t = {
    overlay : Overlay.t;
    text : string;
        (** For the slot's base, padded to the DMA's 16-byte beat. *)
    data : string;  (** For the data region's base, likewise padded. *)
    bss : int64;  (** Zero bytes after [data], at its aligned end. *)
    bss_at : int64;  (** Where they start. *)
    entry_offset : int64;  (** The entry's address less the slot's base. *)
  }
end

(** A size against the room for it. *)
module Too_large : sig
  type t = { bytes : int64; capacity : int64 }
end

type error =
  [ Machine_rivet_riscv32.Rv32_image.Error.t
  | `Overlay_data_too_large of Too_large.t
  | `Overlay_entry_missing of string
  | `Overlay_segment of string
    (** A segment other than [.text], [.data] and [.bss]. *)
  | `Overlay_text_too_large of Too_large.t ]

val pp_error : error Fmt.t

val link :
  ?imports:(string * int64) list ->
  rv32_profile:Machine_rivet_riscv32.Rv32_profile.t ->
  slot_base:int64 ->
  slot_size:int64 ->
  data_base:int64 ->
  data_size:int64 ->
  Overlay.t ->
  (Linked.t, [> error ]) Err.t
(** The overlay's text at [slot_base] and its data and bss from [data_base]. A
    reference to a symbol outside [modules] resolves to its address in [imports]
    (the resident image's exports); any other is a [`Rivet_plan] error. A text
    larger than [slot_size], or data and bss larger than [data_size], is
    refused. *)
