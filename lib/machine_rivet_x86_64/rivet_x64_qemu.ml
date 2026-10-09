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

(* The probe: the registers the System V ABI makes the callee preserve are set
   to sentinels and MXCSR to a rounding mode, flush-to-zero and denormals-are-
   zero setting the kernel must neither use nor lose, then the image's entry
   is called; the helpers it calls are stubs that record a misaligned stack
   and clobber every register a callee may. *)
module Abi = struct
  type t = {
    callee_saved : (string * int64 * int64) list;  (** name, before, after *)
    rsp_before : int64;
    rsp_after : int64;
    mxcsr_before : int64;
    mxcsr_after : int64;
    misaligned : int64;  (** the OR of every helper entry's [(rsp+8) land 15] *)
  }

  let altered_mxcsr = 0xFFC0L

  let sentinels =
    [
      ("rbx", 0x0B0B_0B0B_0B0B_0B0BL);
      ("rbp", 0x0505_0505_0505_0505L);
      ("r12", 0x1212_1212_1212_1212L);
      ("r13", 0x1313_1313_1313_1313L);
      ("r14", 0x1414_1414_1414_1414L);
      ("r15", 0x1515_1515_1515_1515L);
    ]

  let violations t =
    List.filter_map
      (fun (n, b, a) ->
        if Int64.equal a b then None
        else Some (Fmt.str "%s changed from %Lx to %Lx" n b a))
      t.callee_saved
    @ (if Int64.equal t.rsp_before t.rsp_after then []
       else [ Fmt.str "rsp changed from %Lx to %Lx" t.rsp_before t.rsp_after ])
    @ (if Int64.equal t.mxcsr_before t.mxcsr_after then []
       else
         [
           Fmt.str "mxcsr changed from %Lx to %Lx" t.mxcsr_before t.mxcsr_after;
         ])
    @
    if Int64.equal t.misaligned 0L then []
    else [ "a helper was entered with a misaligned stack" ]
end

type layout = { slots : (Table.slot * int) list; total : int }

(* The status word, the ABI probe's record, then each slot at 16-byte alignment
   (or its own, if more). *)
let probe_record = 8
let first_slot = 128

let layout slots =
  let align_up n a = (n + a - 1) / a * a in
  let at = ref first_slot in
  let placed =
    List.map
      (fun (s : Table.slot) ->
        let off = align_up !at (max 16 (Int64.to_int s.Table.align)) in
        at := off + Int64.to_int s.Table.size;
        (s, off))
      slots
  in
  { slots = placed; total = align_up !at 16 }

let harness ?(probe = false) ~helpers { slots; total } =
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
  let rip_at name off =
    Fam.Operand.Mem
      {
        Fam.Mem.base = Some Fam.rip_reg;
        index = None;
        scale = 1;
        disp =
          Fam.Disp.Sym
            (Asm_core.Expr.Binary
               ( Asm_core.Expr.Add,
                 Asm_core.Expr.Symbol name,
                 const (Int64.of_int off) ));
      }
  in
  let data =
    [
      dir
        (D.Section { name = ".data"; perms = Asm_core.Perms.rw; nobits = false });
      dir (D.Align { boundary = 16 });
      dir (D.Global { name = block });
      lbl block;
      dir (D.Zero { length = total });
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
  let probe_regs = [ "rbx"; "rbp"; "r12"; "r13"; "r14"; "r15" ] in
  let prologue =
    if not probe then []
    else
      List.map (fun (n, v) -> i "movq" [ F.imm v; reg n ]) Abi.sentinels
      @ [
          i "subq" [ F.imm 16L; reg "rsp" ];
          i "movl"
            [
              F.imm Abi.altered_mxcsr;
              F.mem_op ~base:(F.find env "rsp") ~disp:0L ();
            ];
          i "ldmxcsr" [ F.mem_op ~base:(F.find env "rsp") ~disp:0L () ];
          i "addq" [ F.imm 16L; reg "rsp" ];
          i "movq" [ reg "rsp"; rip_at block 56 ];
        ]
  in
  let epilogue =
    if not probe then []
    else
      List.mapi
        (fun k n -> i "movq" [ reg n; rip_at block (8 + (8 * k)) ])
        probe_regs
      @ [
          i "movq" [ reg "rsp"; rip_at block 64 ];
          i "subq" [ F.imm 16L; reg "rsp" ];
          i "stmxcsr" [ F.mem_op ~base:(F.find env "rsp") ~disp:0L () ];
          i "movl" [ F.mem_op ~base:(F.find env "rsp") ~disp:0L (); reg "ecx" ];
          i "movq" [ reg "rcx"; rip_at block 72 ];
          i "addq" [ F.imm 16L; reg "rsp" ];
        ]
  in
  let stubs =
    if not probe then []
    else
      List.concat_map
        (fun name ->
          [
            dir (D.Align { boundary = 16 });
            dir (D.Global { name });
            lbl name;
            i "leaq"
              [ F.mem_op ~base:(F.find env "rsp") ~disp:8L (); reg "rax" ];
            i "andq" [ F.imm 15L; reg "rax" ];
            i "orq" [ reg "rax"; rip_at block 80 ];
          ]
          @ List.map
              (fun n -> i "movq" [ F.imm 0xDEADL; reg n ])
              [ "rax"; "rcx"; "rdx"; "rsi"; "rdi"; "r8"; "r9"; "r10"; "r11" ]
          @ List.init 16 (fun k ->
              let x = reg (Printf.sprintf "xmm%d" k) in
              i "xorps" [ x; x ])
          @ [ i "ret" [] ])
        helpers
  in
  let text =
    [
      dir
        (D.Section { name = ".text"; perms = Asm_core.Perms.rx; nobits = false });
      dir (D.Align { boundary = 16 });
      dir (D.Global { name = "_start" });
      lbl "_start";
      i "leaq" [ rip table; reg "rdi" ];
    ]
    @ prologue
    @ [ i "call" [ Fam.Operand.Sym (Asm_core.Expr.Symbol Table.entry) ] ]
    @ [ i "movq" [ reg "rax"; rip block ] ]
    @ epilogue
    @ [
        i "movl" [ F.imm 1L; reg "eax" ];
        i "movl" [ F.imm 1L; reg "edi" ];
        i "leaq" [ rip block; reg "rsi" ];
        i "movl" [ F.imm (Int64.of_int total); reg "edx" ];
        i "syscall" [];
        i "movl" [ F.imm 60L; reg "eax" ];
        i "xorl" [ reg "edi"; reg "edi" ];
        i "syscall" [];
      ]
    @ stubs
  in
  {
    N.unit_name = "harness";
    items =
      text @ data @ [ dir (D.Declared_section { name = ".note.GNU-stack" }) ];
  }

let base = 0x400000L

(* Typed modules laid out at {!base}. *)
let image_of ~entry modules =
  let render pp e = Fmt.str "%a" pp e in
  let* laid =
    Result.map_error
      (render Rivet_x64_image.Error.pp)
      (Err.payload (Rivet_x64_image.plan ~entry modules))
  in
  Result.map_error
    (render Rivet_x64_image.Error.pp)
    (Err.payload (Rivet_x64_image.bind ~base laid))

(* ... and written as a static ELF entered at [entry]. *)
let elf_of ~entry modules =
  let* image = image_of ~entry modules in
  Ok (Rivet_x64_elf.write image)

(* An artifact and its harness laid out once: the code does not change between
   runs, only the bytes of the data block do. *)
type prepared = { image : Image.t; lay : layout; probe : bool }

let prepare ?(runtime = Machine_rivet_common.Rivet_runtime.Dependency_free)
    ?(probe = false) ?entry_mutation artifact =
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
      (Err.payload (M.of_artifact ?entry_mutation ~binding:M.Table artifact))
  in
  let* host =
    Result.map_error
      (render Rivet_x64_refusal.pp)
      (Err.payload
         (harness ~probe
            ~helpers:(Machine_rivet_common.Rivet_runtime.helpers artifact)
            lay))
  in
  let* image = image_of ~entry:"_start" [ modul; host ] in
  Ok { image; lay; probe }

(* The image with its data block holding the caller's bytes for every bound
   region. *)
let with_block (p : prepared) ~bound =
  match List.assoc_opt block p.image.Image.exports with
  | None -> Error "the harness block has no address"
  | Some addr ->
      let patched = ref false in
      let segments =
        List.map
          (fun (s : Image.segment) ->
            let len = Int64.of_int (String.length s.Image.bytes) in
            if
              Int64.compare addr s.Image.address >= 0
              && Int64.compare addr (Int64.add s.Image.address len) < 0
            then begin
              patched := true;
              let at = Int64.to_int (Int64.sub addr s.Image.address) in
              let b = Bytes.of_string s.Image.bytes in
              List.iter
                (fun ((slot : Table.slot), off) ->
                  match (slot.Table.section, bound slot.Table.region) with
                  | Art.Section.Bound, Some bytes ->
                      Bytes.blit_string bytes 0 b (at + off)
                        (min (String.length bytes)
                           (Int64.to_int slot.Table.size))
                  | _ -> ())
                p.lay.slots;
              { s with Image.bytes = Bytes.unsafe_to_string b }
            end
            else s)
          p.image.Image.segments
      in
      if !patched then Ok { p.image with Image.segments }
      else Error "the harness block is in no segment"

let qemu = "qemu-x86_64"

type outcome = {
  status : int64;
  regions : (Mir_id.Region.t * string) list;
  abi : Abi.t option;  (** with a probe *)
}

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

let decode ?(probe = false) lay out =
  if String.length out <> lay.total then
    Error
      (Fmt.str "the emulated process wrote %d bytes, expected %d"
         (String.length out) lay.total)
  else
    let word k = String.get_int64_le out k in
    Ok
      {
        status = word 0;
        abi =
          (if not probe then None
           else
             Some
               {
                 Abi.callee_saved =
                   List.mapi
                     (fun k (n, v) -> (n, v, word (8 + (8 * k))))
                     Abi.sentinels;
                 rsp_before = word 56;
                 rsp_after = word 64;
                 mxcsr_before = Abi.altered_mxcsr;
                 mxcsr_after = word 72;
                 misaligned = word 80;
               });
        regions =
          List.map
            (fun ((s : Table.slot), off) ->
              (s.Table.region, String.sub out off (Int64.to_int s.Table.size)))
            lay.slots;
      }

(* One run of a prepared process over the caller's bound bytes. *)
let launch ?timeout (p : prepared) ~bound =
  let* image = with_block p ~bound in
  let* out = execute ?timeout (Rivet_x64_elf.write image) in
  decode ~probe:p.probe p.lay out

let run ?runtime ?timeout ?probe ?entry_mutation ~bound artifact =
  let* p = prepare ?runtime ?probe ?entry_mutation artifact in
  launch ?timeout p ~bound
