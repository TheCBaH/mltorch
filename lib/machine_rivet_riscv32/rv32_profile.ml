(* The ISA and floating-point ABI an RV32 image is built for: the one choice the
   assembler's feature set, the ELF flags, the C library and the emulator's CPU
   are all read from, so they cannot disagree.

   [Rv32imf_ilp32f] is the default: binary32 only, as on an RV32 core without
   the D extension (Coral NPU among them). Rivet assembles it with its [d]
   component disabled, so a D instruction in a typed module is refused when the
   module is lowered, and qemu runs it with D disabled, so one that reached
   execution by another way would be an illegal instruction rather than a
   silent success. [Rv32imafd_ilp32d] is the binary64 profile, for code that
   computes in [double]. *)

type t = Rv32imafd_ilp32d | Rv32imf_ilp32f

let default = Rv32imf_ilp32f

let name = function
  | Rv32imafd_ilp32d -> "rv32imafd-ilp32d"
  | Rv32imf_ilp32f -> "rv32imf-ilp32f"

(* Rivet's optional components: everything, or everything but [d]. *)
let features : t -> Target_config.spec = function
  | Rv32imafd_ilp32d -> Target_config.default_spec
  | Rv32imf_ilp32f -> [ Target_config.Disable "d" ]

(* The assembler state lowering starts from. The specs are fixed, so a failure
   means the installed Rivet predates its [d] component. *)
let state p =
  match Target_config.resolve Riscv32_encode.components (features p) with
  | Ok config -> Riscv32_encode.initial_state config
  | Error m -> invalid_arg ("Rv32_profile.state: " ^ m)

(* The ISA string is the one GNU as 2.44 writes for [-march=rv32imf] under its
   default ISA specification, as for the named profiles of [Elf.Riscv]. *)
let elf_abi = function
  | Rv32imafd_ilp32d -> Elf.Riscv.rv32imafd_ilp32d
  | Rv32imf_ilp32f ->
      {
        Elf.Riscv.rv32im_ilp32 with
        float_abi = Elf.Riscv.Single;
        arch = "rv32i2p0_m2p0_f2p0_zmmul1p0";
      }

(* The [-cpu] of qemu-riscv32, where qemu's default (which has every extension
   the binary64 profile uses) is too wide. *)
let qemu_cpu = function
  | Rv32imafd_ilp32d -> None
  | Rv32imf_ilp32f -> Some "rv32,d=false"
