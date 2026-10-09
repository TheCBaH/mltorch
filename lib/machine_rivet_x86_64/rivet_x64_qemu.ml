(* An artifact run as a static x86-64 process under qemu-user. This is
   emulation: it checks the instruction bytes and the control flow against the
   interpreter's, and says nothing about an x86-64 CPU's timing, its flag
   corner cases beyond what qemu implements, or its memory ordering.

   The process is built from the artifact's table-bound module and a harness:
   one data block holding every mutable region at its caller's initial bytes,
   the table of their addresses, and a [_start] that calls the image's entry,
   stores the status word at the front of the block and writes the whole block
   to standard output. The runner reads it back. *)

open Machine_ir
module Art = Machine_model.Mir_artifact
module F = Rivet_x64_form
module M = Rivet_x64_module
module N = Asm_core.Normalized_ast
module D = Asm_core.Directive
module Fam = X86_family_encode
module Table = Machine_rivet_common.Rivet_table

let ( let* ) = Result.bind
let block = "harness_block"
let table = "harness_table"
let origin = F.origin
let insn i = N.Instruction { insn = i; origin }
let dir directive = N.Directive { directive; origin }
let lbl name = N.Label { name; origin }
let const n = Asm_core.Expr.Const (Foundation.Bigint.of_int64 n)

type layout = { slots : (Table.slot * int) list; total : int }

(* The status word, then each slot at 16-byte alignment (or its own, if more). *)
let layout slots =
  let align_up n a = (n + a - 1) / a * a in
  let at = ref 8 in
  let placed =
    List.map
      (fun (s : Table.slot) ->
        let off = align_up !at (max 16 (Int64.to_int s.Table.align)) in
        at := off + Int64.to_int s.Table.size;
        (s, off))
      slots
  in
  { slots = placed; total = align_up !at 16 }

let harness ~bound { slots; total } =
  Err.Escape.with_escape @@ fun esc ->
  let env =
    {
      F.mutation = None;
      esc;
      reference = (fun _ -> None);
      table_slot = (fun _ -> None);
    }
  in
  let reg n = Fam.Operand.Reg (F.find env n) in
  let i mnemonic ops = insn (F.make env mnemonic ops) in
  let rip name =
    Fam.Operand.Mem
      {
        Fam.Mem.base = Some Fam.rip_reg;
        index = None;
        scale = 1;
        disp = Fam.Disp.Sym (Asm_core.Expr.Symbol name);
      }
  in
  let initial =
    let b = Bytes.make total '\000' in
    List.iter
      (fun ((s : Table.slot), off) ->
        match (s.Table.section, bound s.Table.region) with
        | Art.Section.Bound, Some bytes ->
            Bytes.blit_string bytes 0 b off
              (min (String.length bytes) (Int64.to_int s.Table.size))
        | _ -> ())
      slots;
    b
  in
  let data =
    [
      dir
        (D.Section { name = ".data"; perms = Asm_core.Perms.rw; nobits = false });
      dir (D.Align { boundary = 16 });
      dir (D.Global { name = block });
      lbl block;
      dir
        (D.Data
           {
             width = 1;
             values =
               List.init total (fun k ->
                   const (Int64.of_int (Char.code (Bytes.get initial k))));
           });
      dir (D.Align { boundary = 8 });
      dir (D.Global { name = table });
      lbl table;
      dir
        (D.Data
           {
             width = 8;
             values =
               List.map
                 (fun (_, off) ->
                   Asm_core.Expr.Binary
                     ( Asm_core.Expr.Add,
                       Asm_core.Expr.Symbol block,
                       const (Int64.of_int off) ))
                 slots;
           });
    ]
  in
  let text =
    [
      dir
        (D.Section { name = ".text"; perms = Asm_core.Perms.rx; nobits = false });
      dir (D.Align { boundary = 16 });
      dir (D.Global { name = "_start" });
      lbl "_start";
      i "leaq" [ rip table; reg "rdi" ];
      i "call" [ Fam.Operand.Sym (Asm_core.Expr.Symbol Table.entry) ];
      i "movq" [ reg "rax"; rip block ];
      i "movl" [ F.imm 1L; reg "eax" ];
      i "movl" [ F.imm 1L; reg "edi" ];
      i "leaq" [ rip block; reg "rsi" ];
      i "movl" [ F.imm (Int64.of_int total); reg "edx" ];
      i "syscall" [];
      i "movl" [ F.imm 60L; reg "eax" ];
      i "xorl" [ reg "edi"; reg "edi" ];
      i "syscall" [];
    ]
  in
  {
    N.unit_name = "harness";
    items =
      text @ data @ [ dir (D.Declared_section { name = ".note.GNU-stack" }) ];
  }

let base = 0x400000L

(* Typed modules laid out at {!base} and written as a static ELF entered at
   [entry]. *)
let elf_of ~entry modules =
  let render pp e = Fmt.str "%a" pp e in
  let* laid =
    Result.map_error
      (render Rivet_x64_image.Error.pp)
      (Err.payload (Rivet_x64_image.plan ~entry modules))
  in
  let* image =
    Result.map_error
      (render Rivet_x64_image.Error.pp)
      (Err.payload (Rivet_x64_image.bind ~base laid))
  in
  Ok (Rivet_x64_elf.write image)

(* The ELF of the artifact and its harness, with the layout that reads its output. *)
let process ?(runtime = Machine_rivet_common.Rivet_runtime.Dependency_free)
    ~bound artifact =
  let render pp e = Fmt.str "%a" pp e in
  let* () =
    Result.map_error
      (fun h -> Fmt.str "helper %s has no implementation in this image" h)
      (Machine_rivet_common.Rivet_runtime.admit runtime artifact)
  in
  let slots = Table.of_artifact artifact in
  let lay = layout slots in
  let* modul =
    Result.map_error
      (render Rivet_x64_refusal.pp)
      (Err.payload (M.of_artifact ~binding:M.Table artifact))
  in
  let* host =
    Result.map_error
      (render Rivet_x64_refusal.pp)
      (Err.payload (harness ~bound lay))
  in
  let* elf = elf_of ~entry:"_start" [ modul; host ] in
  Ok (elf, lay)

let qemu = "qemu-x86_64"

type outcome = { status : int64; regions : (Mir_id.Region.t * string) list }

(* Runs the ELF, returning its standard output. *)
let execute ?(timeout = 120) elf =
  let path = Filename.temp_file "mltorch_x64_" ".elf" in
  let out = Filename.temp_file "mltorch_x64_" ".out" in
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
        (Sys.getenv_opt "MLTORCH_X64_ELF");
      Unix.chmod path 0o755;
      let fd = Unix.openfile out [ Unix.O_WRONLY; Unix.O_TRUNC ] 0o600 in
      let devnull = Unix.openfile "/dev/null" [ Unix.O_RDONLY ] 0 in
      let pid =
        Unix.create_process "timeout"
          [| "timeout"; string_of_int timeout; qemu; path |]
          devnull fd Unix.stderr
      in
      Unix.close fd;
      Unix.close devnull;
      match snd (Unix.waitpid [] pid) with
      | Unix.WEXITED 0 -> Ok (In_channel.with_open_bin out In_channel.input_all)
      | Unix.WEXITED n -> Error (Fmt.str "the emulated process exited %d" n)
      | Unix.WSIGNALED n | Unix.WSTOPPED n ->
          Error (Fmt.str "the emulated process was stopped by signal %d" n))

let decode lay out =
  if String.length out <> lay.total then
    Error
      (Fmt.str "the emulated process wrote %d bytes, expected %d"
         (String.length out) lay.total)
  else
    Ok
      {
        status = String.get_int64_le out 0;
        regions =
          List.map
            (fun ((s : Table.slot), off) ->
              (s.Table.region, String.sub out off (Int64.to_int s.Table.size)))
            lay.slots;
      }

let run ?runtime ?timeout ~bound artifact =
  let* elf, lay = process ?runtime ~bound artifact in
  let* out = execute ?timeout elf in
  decode lay out
