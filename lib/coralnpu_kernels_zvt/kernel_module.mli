(** A list of typed kernel items as an assembler module for the Zve32f profile,
    with the Zvt extension when the items use it: the route every generated
    function on the matrix engine takes from items to bytes. *)

val of_items :
  ?zvt:bool ->
  name:string ->
  Machine_rivet_riscv32.Rv32_kernel_item.t list ->
  (Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_, string) result
(** [name] is the kernel's (and its module's) name; [zvt] (default true) gives
    the Zvt encoder, so a function using no tile instruction may pass [false]
    and need no matrix configuration. A refusal is rendered. *)
