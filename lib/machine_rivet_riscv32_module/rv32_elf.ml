(* A bound image as a static RV32 ELF executable, built by [Elf.Exec]: one
   PT_LOAD per segment, no section headers, flagged with the profile's ABI. Nothing is linked: Rivet has already
   placed every byte. *)

let perms (p : Asm_core.Perms.t) =
  {
    Elf.Exec.read = p.Asm_core.Perms.read;
    write = p.Asm_core.Perms.write;
    execute = p.Asm_core.Perms.execute;
  }

(* The initialized bytes, then the zero-filled tail at the next address; the
   sections are named [name].data and [name].bss. *)
let segment name (s : Image.segment) : Elf.Exec.segment =
  let len = String.length s.Image.bytes in
  let data =
    if len = 0 then []
    else
      [
        {
          Elf.Exec.name = name ^ ".data";
          addr = s.Image.address;
          align = 1L;
          content = Elf.Exec.Bytes s.Image.bytes;
        };
      ]
  and bss =
    if s.Image.zero_fill = 0 then []
    else
      [
        {
          Elf.Exec.name = name ^ ".bss";
          addr = Int64.add s.Image.address (Int64.of_int len);
          align = 1L;
          content = Elf.Exec.Zero (Int64.of_int s.Image.zero_fill);
        };
      ]
  in
  { Elf.Exec.perms = perms s.Image.perms; paddr = None; sections = data @ bss }

let write ?(profile = Rv32_profile.default) (image : Image.t) :
    (string, string) result =
  match image.Image.entry with
  | None -> Error "image has no entry"
  | Some entry ->
      let segments =
        List.filter
          (fun (s : Image.segment) ->
            String.length s.Image.bytes > 0 || s.Image.zero_fill > 0)
          image.Image.segments
      in
      let spec =
        {
          Elf.Exec.target = Elf.Target.riscv (Rv32_profile.elf_abi profile);
          entry;
          segments =
            List.mapi
              (fun i s -> segment (Printf.sprintf ".seg%d" i) s)
              segments;
          headers = Elf.Exec.Unmapped;
          gnu_stack = None;
          section_headers = false;
          symbols = [];
        }
      in
      Result.map_error
        (Fmt.str "%a" Elf.Exec.pp_error)
        (Err.payload (Elf.Exec.build spec))
