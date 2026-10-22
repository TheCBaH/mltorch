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

module Extension_form : sig
  type t = { at : Rv32_kernel_item.At.t; feature : string; message : string }
  (** The extension's encoder refused an instruction. [message] is Rivet's
      diagnostic, rendered: it belongs to another encoder instance. *)

  val pp : t Fmt.t
end

module Extension_profile : sig
  type t = {
    at : Rv32_kernel_item.At.t;
    feature : string;
    profile : Rv32_profile.t;
    missing : string list;  (** Rivet features, in its order. *)
  }
  (** The extension needs features the profile does not have, which Rivet would
      otherwise enable along with it. *)

  val pp : t Fmt.t
end

module Extension_missing : sig
  type t = { at : Rv32_kernel_item.At.t; feature : string }
  (** An instruction of a vendor extension the lowering was not given. *)

  val pp : t Fmt.t
end

type error =
  [ `Extension_form of Extension_form.t
  | `Extension_missing of Extension_missing.t
  | `Extension_profile of Extension_profile.t
  | `Rivet_form of Rivet_form.t ]

val pp_error : error Fmt.t

(** A Rivet vendor extension: rows the standard RV32 encoder lacks, such as
    Coral NPU's Zvt. Its instructions are encoded one at a time by an encoder
    over the standard extensions and these rows, under the profile's features
    and the extension's own, and placed as words: they have no fixups. *)
module Extension : sig
  type t

  val make : feature:string -> Riscv_family_encode.Extensions.Validated.t -> t
  (** [feature] is the Rivet feature that enables [definition]'s rows, the one
      {!Rv32_kernel_insn.extension} names. *)

  val feature : t -> string
end

val module_ :
  ?extensions:Extension.t list ->
  profile:Rv32_profile.t ->
  Rv32_kernel.t ->
  (Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_, error) Err.t
