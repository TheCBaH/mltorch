let source = "long entry(void *io) { return 42; }\n"

let first_instruction asm =
  String.split_on_char '\n' asm
  |> List.find_opt (fun l ->
      String.length l > 0 && l.[0] = '\t' && l.[1] <> '.')
  |> Option.value ~default:"?"

let () =
  (match Compcert_embed_aarch64.compile_to_asm source with
  | Ok asm -> Printf.printf "aarch64: %s\n" (first_instruction asm)
  | Error e -> Format.printf "aarch64: %a\n" Compcert_embed.pp_error e);
  match Compcert_embed_riscv32_fp32.compile_to_asm source with
  | Ok asm -> Printf.printf "riscv32_fp32: %s\n" (first_instruction asm)
  | Error e -> Format.printf "riscv32_fp32: %a\n" Compcert_embed.pp_error e
