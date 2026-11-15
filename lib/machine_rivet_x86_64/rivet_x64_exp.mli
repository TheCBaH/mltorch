(** The project-owned [exp] as typed Rivet instructions (see the implementation
    for the register contract). Its constants-block offsets are private. *)

val module_ :
  ( X86_family_encode.Instruction.t Asm_core.Normalized_ast.module_,
    Rivet_x64_refusal.t )
  Err.t

val symbol : string
(** The exported function's name. *)
