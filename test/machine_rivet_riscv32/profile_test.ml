open Machine_rivet_riscv32

(* Each profile's resolved Rivet features and the ELF ISA string written from
   them. The BF16 profile has no Zfhmin: Coral NPU lacks it, and Zfbfmin brings
   [flh] on its own. *)
let%expect_test "profile features and ELF ISA strings" =
  List.iter
    (fun p ->
      Fmt.pr "%s: %s@.  %s@." (Rv32_profile.name p)
        (String.concat "," (Target_config.features (Rv32_profile.config p)))
        (Rv32_profile.elf_abi p).Elf.Riscv.arch)
    Rv32_profile.
      [
        Rv32imafd_ilp32d;
        Rv32imf_ilp32f;
        Rv32imf_zve32f_ilp32f;
        Rv32imf_zve32f_zvfbfwma_ilp32f;
      ];
  [%expect
    {|
    rv32imafd-ilp32d: a,d,f,m,zicsr,zmmul
      rv32i2p1_m2p0_a2p1_f2p2_d2p2_zicsr2p0_zmmul1p0
    rv32imf-ilp32f: f,m,zicsr,zmmul
      rv32i2p1_m2p0_f2p2_zicsr2p0_zmmul1p0
    rv32imf_zve32f-ilp32f: f,m,zicsr,zmmul,zve32f,zve32x
      rv32i2p1_m2p0_f2p2_zicsr2p0_zmmul1p0_zve32f1p0_zve32x1p0_zvl32b1p0
    rv32imf_zve32f_zvfbfwma-ilp32f: f,m,zfbfmin,zicsr,zmmul,zve32f,zve32x,zvfbfmin,zvfbfwma
      rv32i2p1_m2p0_f2p2_zicsr2p0_zmmul1p0_zfbfmin1p0_zve32f1p0_zve32x1p0_zvfbfmin1p0_zvfbfwma1p0_zvl32b1p0 |}]
