(* The smallest process: [_start] exits with a given status through the Linux
   exit_group system call. *)

open Machine_rivet_riscv32
open Rv32_asm

let exit_with status =
  module_ "smoke" (fun () ->
      [
        text;
        global "_start";
        label "_start";
        insn "li" [ reg "a0"; imm status ];
        insn "li" [ reg "a7"; imm 94 ];
        insn "ecall" [];
      ])

let run status =
  match exit_with status with
  | Error what -> Fmt.pr "no form for %s@." what
  | Ok m -> (
      match Rv32_harness.elf_of ~entry:"_start" [ m ] with
      | Error e -> Fmt.pr "elf: %s@." e
      | Ok elf -> (
          match
            Rv32_process.execute
              ?cpu:(Rv32_profile.qemu_cpu Rv32_profile.default)
              elf
          with
          | Error e -> Fmt.pr "run: %s@." e
          | Ok { Rv32_process.exit_code; _ } -> Fmt.pr "exit %d@." exit_code))

let%expect_test "exit status" =
  run 42;
  run 0;
  [%expect {|
    exit 42
    exit 0 |}]
