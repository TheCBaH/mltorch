(* The ISA and floating-point ABI an RV32 image is built for: the one choice the
   assembler's feature set, the ELF flags, the C library and the emulator's CPU
   are all read from, so they cannot disagree.

   [Rv32imf_ilp32f] is the default: binary32 only, as on an RV32 core without
   the D extension (Coral NPU among them). Rivet assembles it with exactly M and
   F, so a D or vector instruction in a typed module is refused when the module
   is lowered, and qemu runs it with D disabled, so one that reached execution
   by another way would be an illegal instruction rather than a silent success.
   [Rv32imf_zve32f_ilp32f] adds the embedded vector profile with binary32
   elements, as on Coral NPU's vector core (VLEN 128). [Rv32imafd_ilp32d] is the
   binary64 profile, for code that computes in [double]. *)

type t = Rv32imafd_ilp32d | Rv32imf_ilp32f | Rv32imf_zve32f_ilp32f

let default = Rv32imf_ilp32f

let name = function
  | Rv32imafd_ilp32d -> "rv32imafd-ilp32d"
  | Rv32imf_ilp32f -> "rv32imf-ilp32f"
  | Rv32imf_zve32f_ilp32f -> "rv32imf_zve32f-ilp32f"

(* Rivet's components, named from an empty set: its default enables every
   component it has (C, B, V and more), which no profile here claims. Each set
   is the profile's ISA string, closed under Rivet's requirements (F brings
   Zicsr, M Zmmul, Zve32f Zve32x). *)
let features : t -> Target_config.spec =
  let open Target_config in
  function
  | Rv32imafd_ilp32d ->
      [ None_; Enable "m"; Enable "a"; Enable "f"; Enable "d" ]
  | Rv32imf_ilp32f -> [ None_; Enable "m"; Enable "f" ]
  | Rv32imf_zve32f_ilp32f -> [ None_; Enable "m"; Enable "f"; Enable "zve32f" ]

(* The resolved feature set. The specs are fixed, so a failure means the
   installed Rivet predates its component model. *)
let config p =
  match Target_config.resolve Riscv32_encode.components (features p) with
  | Ok config -> config
  | Error m -> invalid_arg ("Rv32_profile.config: " ^ m)

(* The encoder state typed modules are lowered from. Rivet's text pipeline has
   a state of its own, made from the same [config]. *)
let state p = Riscv32_encode.initial_state (config p)

(* The ISA string is the one GNU as 2.44 writes for [-march=rv32imf] under its
   default ISA specification, as for the named profiles of [Elf.Riscv]. The
   vector one adds Zve32f and what binutils' implication table derives from it
   (Zve32x, Zvl32b, Zicsr); it has not been compared with an assembler's
   output. *)
let elf_abi = function
  | Rv32imafd_ilp32d -> Elf.Riscv.rv32imafd_ilp32d
  | Rv32imf_ilp32f ->
      {
        Elf.Riscv.rv32im_ilp32 with
        float_abi = Elf.Riscv.Single;
        arch = "rv32i2p0_m2p0_f2p0_zmmul1p0";
      }
  | Rv32imf_zve32f_ilp32f ->
      {
        Elf.Riscv.rv32im_ilp32 with
        float_abi = Elf.Riscv.Single;
        arch =
          "rv32i2p0_m2p0_f2p0_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zvl32b1p0";
      }

(* The [-cpu] of qemu-riscv32, where qemu's default (which has every extension
   the binary64 profile uses) is too wide. The vector profile is Coral NPU's
   shape: VLEN 128, 32-bit elements. *)
let qemu_cpu = function
  | Rv32imafd_ilp32d -> None
  | Rv32imf_ilp32f -> Some "rv32,d=false"
  | Rv32imf_zve32f_ilp32f -> Some "rv32,d=false,zve32f=true,vlen=128,elen=32"
