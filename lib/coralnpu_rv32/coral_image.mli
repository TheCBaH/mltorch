(** A kernel and the runtime around it, laid out for a profile and written as
    ELF.

    The runtime's code depends on the layout (which slots are staged) and the
    layout on the code's size, so the two are iterated from a layout with no
    staging until the layout repeats; DTCM placement does not depend on code
    size, so this settles on the second round. *)

type t = {
  elf : string;
  image : Image.t;
      (** The bound image the ELF was written from: its segments and exported
          symbols at their addresses. *)
  layout : Coral_layout.t;
  profile : Coral_profile.t;
}

type error =
  [ Coral_layout.error
  | Coral_elf.error
  | Machine_rivet_riscv32.Rv32_image.Error.t
  | `Layout_unsettled  (** No fixed point within four rounds. *)
  | `Runtime_form of string  (** A form Rivet refused in the runtime. *) ]

val pp_error : error Fmt.t

val build :
  ?params:Coral_layout.Params.t ->
  ?abi:Machine_rivet_riscv32.Rv32_harness.Abi.t ->
  ?lowered:Riscv32_encode.fixup_kind Asm_core.Lowered_ast.module_ list ->
  ?rv32_profile:Machine_rivet_riscv32.Rv32_profile.t ->
  Coral_profile.t ->
  entry:string ->
  kernel:Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_ list ->
  Coral_layout.Slot.t list ->
  (t, [> error ]) Err.t
(** [abi] defaults to [Table]. [kernel] and [lowered] together hold every module
    the entry needs, the C library included; [lowered] are modules already
    lowered under {!Machine_rivet_riscv32.Rv32_profile.default} (CompCert's
    assembly, through Rivet's text pipeline). *)
