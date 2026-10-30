(** The resident side of code overlays ({!Coral_overlay}): a slot of the
    instruction memory, one stub per overlay and the routine that makes an
    overlay resident.

    A stub has its overlay's entry name, so a caller (a code generator's, or
    CompCert's) reaches it as it would the function itself. The stub loads the
    overlay if another is resident and tail-jumps into the slot with the
    caller's argument registers and [ra] intact: it links through [t4]. A load
    starts the overlay's chain, waits for [STATUS.done] (an engine error or a
    wait that does not end is an [ecall] trap), zeroes the overlay's [.bss] and
    executes [fence.i]. An overlay that stays resident across calls keeps its
    data and bss; one reloaded starts from its initial data and a zero bss. *)

val slot_symbol : string
(** The slot's first byte, exported; an overlay is linked for this address. *)

module Entry : sig
  type t = {
    name : string;  (** The stub's symbol, also the overlay's entry. *)
    entry_offset : int64;  (** The entry's address less the slot's. *)
  }
end

module Table : sig
  type entry = { chain : int64; bss_at : int64; bss_len : int64 }
  (** One entry an overlay, in the order of {!module_}'s entries. *)

  val entry_bytes : int64
  (** The size of an encoded entry, 16. *)

  val encode : entry list -> string
end

val module_ :
  slot_bytes:int64 ->
  state:int64 ->
  table:int64 ->
  Entry.t list ->
  (Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_, string) result
(** [state] is 4 zeroed bytes (the resident overlay's number, 0 for none) and
    [table] the encoded {!Table}. The module's size depends only on the entries
    and [slot_bytes], not on any address. *)
