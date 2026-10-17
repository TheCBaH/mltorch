(** The guest's [_start], trap handler and HTIF calls, as one typed module for a
    layout.

    In order: the stack, [mtvec], [mstatus.FS] and a clear [fcsr]; [Out] and
    [Scratch] slots zeroed; the input stream's header checked (magic, version,
    layout digest, length) and each read slot ([In], [In_out]) read, directly or
    through the staging buffer; the entry called between two reads of [mcycle]
    and [minstret]; the result frame written, the written slots ([In_out],
    [Out]) directly or staged; an HTIF exit of 0, then [mpause] forever.

    A trap writes a trap frame and exits with 1. An input stream the guest
    refuses is reported the same way, with [mcause] {!refused} and the reason in
    [mepc] ({!Refusal}).

    Every address is loaded by a [lui]/[addi] pair, even where one instruction
    would do, so the module's size does not depend on the layout: a draft built
    from any layout has the size of the final one. *)

val refused : int32
(** [0xFFFF_FFFF], an [mcause] no trap has. *)

module Refusal : sig
  type t = Digest | End_of_input | Length | Magic | Trailer | Version

  val of_code : int32 -> t option
  val to_string : t -> string
end

val entry_symbol : string
(** [_start]. *)

val module_ :
  Coral_layout.t ->
  abi:Machine_rivet_riscv32.Rv32_harness.Abi.t ->
  entry:string ->
  (Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_, string) result
(** [Error] names a form Rivet refused. *)
