(** A bound Coral image as the ELF the simulators load.

    One PT_LOAD per memory region, holding that region's sections in address
    order: the layout packs DTCM, so its data segments and the HTIF block can
    share a page, which separate PT_LOADs could not. Every byte is explicit,
    zero-filled tails included, because the RTL testbench loads by copying the
    file's bytes. A segment outside a [loadable] region is refused before any
    ELF is built.

    The HTIF block is a [.htif] section of zeros at the layout's address, with
    the simulators' four symbols over it, plus [_start] at the entry. The ABI is
    ILP32F on RV32IMF with Zicsr and Zifencei, with no compressed instructions.
*)

val abi : Elf.Riscv.abi

module Unloadable : sig
  type t = { address : int64; name : string; size : int64 }
end

type error =
  [ `Elf of Elf.Exec.error | `No_entry | `Unloadable_segment of Unloadable.t ]

val pp_error : error Fmt.t

val write :
  Coral_profile.t -> Coral_layout.t -> Image.t -> (string, [> error ]) Err.t
