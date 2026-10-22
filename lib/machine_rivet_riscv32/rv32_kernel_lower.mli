(** A finalized kernel as a Rivet module, built through the encoder's surface
    constructor and its profile-aware simplification, so a form the profile
    excludes is refused here, at the kernel item that wrote it, rather than
    rendered later by the image planner.

    Exported labels are spelled by their export name, in their definition and in
    every reference; the others as [.L<kernel>_<name>]. Non-global labels are
    local to their module, so the qualification only keeps a kernel's private
    names off its own exports, which cannot start with a dot. *)

module Rivet_form : sig
  type t = { at : Rv32_kernel_item.At.t; error : Riscv32_encode.error }
  (** The encoder refused an instruction. [error] is the encoder's own, with its
      detection origin. *)

  val pp : t Fmt.t
end

type error = [ `Rivet_form of Rivet_form.t ]

val pp_error : error Fmt.t

val module_ :
  profile:Rv32_profile.t ->
  Rv32_kernel.t ->
  (Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_, error) Err.t
