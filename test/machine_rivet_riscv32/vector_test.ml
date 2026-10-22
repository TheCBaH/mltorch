(* The vector profile. A strip-mined loop whose step is whatever [vsetvli]
   grants computes [out[i] = a[i] * 2] and then checks every cell with scalar
   F instructions, exiting with the number of wrong cells; the same image runs
   at several VLENs. The other profiles claim no vector extension, so the same
   module is refused when it is lowered. *)

open Machine_rivet_riscv32
open Rv32_asm

let n = 37
let vtype lmul = [ sym "e32"; sym lmul; sym "ta"; sym "ma" ]

(* exits with VLMAX at e32, m1 *)
let vlmax () =
  module_ "vlmax" (fun () ->
      [
        text;
        global "_start";
        label "_start";
        insn "vsetvli" ([ reg "a0"; reg "zero" ] @ vtype "m1");
        insn "li" [ reg "a7"; imm 94 ];
        insn "ecall" [];
      ])

let doubled lmul =
  module_ "doubled" (fun () ->
      [
        data;
        label "a";
        words
          (List.init n (fun i ->
               Int32.to_int (Int32.bits_of_float (float_of_int i))));
        label "out";
        zero (4 * n);
        text;
        global "_start";
        label "_start";
        insn "la" [ reg "a0"; sym "a" ];
        insn "la" [ reg "a1"; sym "out" ];
        insn "li" [ reg "a2"; imm n ];
      ]
      @ load_imm "t1" 0x40000000
      @ [
          insn "fmv.w.x" [ reg "ft0"; reg "t1" ];
          label "strip";
          insn "vsetvli" ([ reg "t0"; reg "a2" ] @ vtype lmul);
          insn "vle32.v" [ reg "v8"; mem "a0" 0 ];
          insn "vfmul.vf" [ reg "v16"; reg "v8"; reg "ft0" ];
          insn "vse32.v" [ reg "v16"; mem "a1" 0 ];
          insn "sub" [ reg "a2"; reg "a2"; reg "t0" ];
          insn "slli" [ reg "t2"; reg "t0"; imm 2 ];
          insn "add" [ reg "a0"; reg "a0"; reg "t2" ];
          insn "add" [ reg "a1"; reg "a1"; reg "t2" ];
          insn "bnez" [ reg "a2"; sym "strip" ];
          insn "la" [ reg "a1"; sym "out" ];
          insn "li" [ reg "a3"; imm 0 ];
          insn "li" [ reg "a4"; imm 0 ];
          insn "li" [ reg "a5"; imm n ];
          label "check";
          insn "flw" [ reg "ft1"; mem "a1" 0 ];
          insn "fcvt.s.w" [ reg "ft2"; reg "a3" ];
          insn "fadd.s" [ reg "ft2"; reg "ft2"; reg "ft2" ];
          insn "feq.s" [ reg "t3"; reg "ft1"; reg "ft2" ];
          insn "xori" [ reg "t3"; reg "t3"; imm 1 ];
          insn "add" [ reg "a4"; reg "a4"; reg "t3" ];
          insn "addi" [ reg "a1"; reg "a1"; imm 4 ];
          insn "addi" [ reg "a3"; reg "a3"; imm 1 ];
          insn "bne" [ reg "a3"; reg "a5"; sym "check" ];
          insn "mv" [ reg "a0"; reg "a4" ];
          insn "li" [ reg "a7"; imm 94 ];
          insn "ecall" [];
        ])

let run ?cpu profile m =
  let cpu =
    match cpu with Some _ -> cpu | None -> Rv32_profile.qemu_cpu profile
  in
  match m with
  | Error what -> Fmt.pr "no form for %s@." what
  | Ok m -> (
      match Rv32_harness.elf_of ~profile ~entry:"_start" [ m ] with
      | Error e -> Fmt.pr "%s: %s@." (Rv32_profile.name profile) e
      | Ok elf -> (
          match Rv32_process.execute ?cpu elf with
          | Error e -> Fmt.pr "run: %s@." e
          | Ok { Rv32_process.exit_code; _ } -> Fmt.pr "exit %d@." exit_code))

let vector = Rv32_profile.Rv32imf_zve32f_ilp32f

let%expect_test "VLMAX at the profile's VLEN" =
  run vector (vlmax ());
  [%expect {| exit 4 |}]

let%expect_test "the strip-mined loop at every VLEN and LMUL" =
  List.iter
    (fun vlen ->
      List.iter
        (fun lmul ->
          Fmt.pr "vlen %d %s: " vlen lmul;
          run
            ~cpu:
              (Fmt.str
                 "rv32,d=false,zve32f=true,vlen=%d,elen=32,rvv_ta_all_1s=true"
                 vlen)
            vector (doubled lmul))
        [ "m1"; "m8" ])
    [ 128; 256; 1024 ];
  [%expect
    {|
    vlen 128 m1: exit 0
    vlen 128 m8: exit 0
    vlen 256 m1: exit 0
    vlen 256 m8: exit 0
    vlen 1024 m1: exit 0
    vlen 1024 m8: exit 0 |}]

let%expect_test "profiles without a vector extension refuse it" =
  run Rv32_profile.Rv32imf_ilp32f (doubled "m1");
  run Rv32_profile.Rv32imafd_ilp32d (doubled "m1");
  [%expect
    {|
    rv32imf-ilp32f: <synthesized by riscv32 from <synthesized by machine_rivet_riscv32>>: error[riscv32.feature]: vsetvli requires feature zve32x, which is not enabled
    <synthesized by riscv32 from <synthesized by machine_rivet_riscv32>>: error[riscv32.feature]: vle32.v requires feature zve32x, which is not enabled
    <synthesized by riscv32 from <synthesized by machine_rivet_riscv32>>: error[riscv32.feature]: vfmul.vf requires feature zve32f, which is not enabled
    <synthesized by riscv32 from <synthesized by machine_rivet_riscv32>>: error[riscv32.feature]: vse32.v requires feature zve32x, which is not enabled
    rv32imafd-ilp32d: <synthesized by riscv32 from <synthesized by machine_rivet_riscv32>>: error[riscv32.feature]: vsetvli requires feature zve32x, which is not enabled
    <synthesized by riscv32 from <synthesized by machine_rivet_riscv32>>: error[riscv32.feature]: vle32.v requires feature zve32x, which is not enabled
    <synthesized by riscv32 from <synthesized by machine_rivet_riscv32>>: error[riscv32.feature]: vfmul.vf requires feature zve32f, which is not enabled
    <synthesized by riscv32 from <synthesized by machine_rivet_riscv32>>: error[riscv32.feature]: vse32.v requires feature zve32x, which is not enabled |}]
