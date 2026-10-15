(* An RV32 ELF run as a static process under qemu-riscv32. There is no RV32
   hardware to run it on, so every run is emulation: it checks instruction
   bytes and control flow, not an RV32 core's timing or memory ordering. A
   missing qemu-riscv32 is an error, never a skip. *)

type outcome = { exit_code : int; stdout : string }

let qemu = "qemu-riscv32"

(* Runs [elf] with [args], returning its exit code and standard output. *)
let execute ?(timeout = 120) ?(args = []) elf =
  let path = Filename.temp_file "mltorch_rv32_" ".elf" in
  let out = Filename.temp_file "mltorch_rv32_" ".out" in
  Fun.protect
    ~finally:(fun () ->
      (try Sys.remove path with Sys_error _ -> ());
      try Sys.remove out with Sys_error _ -> ())
    (fun () ->
      Out_channel.with_open_bin path (fun oc -> output_string oc elf);
      (* a developer's hook: keep the last process for a disassembler *)
      Option.iter
        (fun keep ->
          Out_channel.with_open_bin keep (fun oc -> output_string oc elf))
        (Sys.getenv_opt "MLTORCH_RV32_ELF");
      Unix.chmod path 0o755;
      let fd = Unix.openfile out [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
      let devnull = Unix.openfile "/dev/null" [ Unix.O_RDONLY ] 0 in
      let command =
        Array.of_list ([ "timeout"; string_of_int timeout; qemu; path ] @ args)
      in
      match Unix.create_process "timeout" command devnull fd Unix.stderr with
      | exception Unix.Unix_error (e, _, _) ->
          Unix.close fd;
          Unix.close devnull;
          Error (Fmt.str "cannot start %s: %s" qemu (Unix.error_message e))
      | pid -> (
          Unix.close fd;
          Unix.close devnull;
          match snd (Unix.waitpid [] pid) with
          | Unix.WEXITED 127 -> Error (Fmt.str "%s is not installed" qemu)
          | Unix.WEXITED 124 -> Error "the process timed out"
          | Unix.WEXITED n ->
              Ok
                {
                  exit_code = n;
                  stdout = In_channel.with_open_bin out In_channel.input_all;
                }
          | Unix.WSIGNALED n | Unix.WSTOPPED n ->
              Error (Fmt.str "the process was stopped by signal %d" n)))
