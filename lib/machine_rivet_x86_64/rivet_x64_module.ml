(* A published x86-64 artifact as one typed Rivet module: the functions'
   text, each data symbol in its section, and nothing else. Blocks are labels
   in function order; a branch whose target follows falls through. A host
   helper's trampoline is a second module ({!helpers}), because its address
   belongs to a process and the artifact's must not. *)

open Machine_ir
open Machine_target_x86_64
module Art = Machine_model.Mir_artifact
module Loc = Mir_phys.Loc
module R = Rivet_x64_refusal
module F = Rivet_x64_form
module Fam = X86_family_encode
module Rivet_cfi = Machine_rivet_common.Rivet_cfi
module Rivet_table = Machine_rivet_common.Rivet_table
module N = Asm_core.Normalized_ast
module D = Asm_core.Directive

let origin = F.origin

let label_of (f : (_, _) Mir_phys.Func.t) b =
  Printf.sprintf ".L%s_b%d" f.Mir_phys.Func.name (Mir_id.Block.to_int b)

let insn i = N.Instruction { insn = i; origin }
let dir directive = N.Directive { directive; origin }
let lbl name = N.Label { name; origin }
let section name perms ~nobits = dir (D.Section { name; perms; nobits })
let const_expr n = Asm_core.Expr.Const (Foundation.Bigint.of_int64 n)

(* {1 Data} *)

let data_items (s : Art.Symbol.t) =
  match s.Art.Symbol.kind with
  | Art.Symbol.Data { size; align; section = sec; _ } ->
      let name = s.Art.Symbol.name in
      let header sect =
        [
          sect;
          dir (D.Align { boundary = Int64.to_int align });
          dir (D.Global { name });
          dir (D.Sym_type { name; kind = D.Object });
          lbl name;
        ]
      and footer = [ dir (D.Sym_size { name; size = const_expr size }) ] in
      Some
        (match sec with
        | Art.Section.Bss | Art.Section.Bound ->
            header (section ".bss" Asm_core.Perms.rw ~nobits:true)
            @ [ dir (D.Zero { length = Int64.to_int size }) ]
            @ footer
        | Art.Section.Rodata bytes ->
            header (section ".rodata" Asm_core.Perms.ro ~nobits:false)
            @ [
                dir
                  (D.Data
                     {
                       width = 1;
                       values =
                         List.init (String.length bytes) (fun i ->
                             const_expr (Int64.of_int (Char.code bytes.[i])));
                     });
              ]
            @ footer)
  | Art.Symbol.External_function _ | Art.Symbol.Function _ -> None

(* {1 Text} *)

let relocation_index artifact =
  let t = Hashtbl.create 64 in
  List.iter
    (fun (r : Art.Relocation.t) ->
      Hashtbl.add t
        ( Mir_id.Func.to_int r.Art.Relocation.func,
          Mir_id.Block.to_int r.Art.Relocation.block,
          Option.map Mir_id.Instr.to_int r.Art.Relocation.instr )
        r)
    (Art.relocations artifact);
  t

let instrs_of_func ~table_slot esc relocations
    (f : (X64_op.t, X64_op.test) Mir_phys.Func.t) =
  let fid = Mir_id.Func.to_int f.Mir_phys.Func.id in
  let blocks = f.Mir_phys.Func.blocks in
  let next =
    let rec go = function
      | a :: (b :: _ as rest) -> (a, Some b) :: go rest
      | [ a ] -> [ (a, None) ]
      | [] -> []
    in
    go blocks
  in
  let env_at bid instr =
    {
      F.mutation = None;
      esc;
      table_slot;
      reference =
        (fun reference ->
          List.find_map
            (fun (r : Art.Relocation.t) ->
              if r.Art.Relocation.reference = reference then
                Some (r.Art.Relocation.symbol, r.Art.Relocation.addend)
              else None)
            (Hashtbl.find_all relocations (fid, bid, instr)));
    }
  in
  let cfi = Rivet_cfi.create ~entry_cfa:8L in
  let entry_jump =
    match blocks with
    | b :: _
      when not (Mir_id.Block.equal b.Mir_phys.Block.id f.Mir_phys.Func.entry) ->
        [ insn (F.jump (env_at (-1) None) (label_of f f.Mir_phys.Func.entry)) ]
    | _ -> []
  in
  entry_jump
  @ List.concat_map
      (fun ((b : (_, _) Mir_phys.Block.t), following) ->
        let bid = Mir_id.Block.to_int b.Mir_phys.Block.id in
        let env = env_at bid in
        let reg (v : Mir_target.View.t) = "%" ^ v.Mir_target.View.name in
        let items_of (i : X64_op.t Mir_phys.Instr.t) =
          List.map insn
            (match i with
            | Mir_phys.Instr.Exec { instr; uses; defs } -> (
                match instr.Mir_instr.op with
                | Mir_sel.Op.Machine op ->
                    F.instructions
                      (env (Some (Mir_id.Instr.to_int instr.Mir_instr.id)))
                      op ~uses ~defs
                | Mir_sel.Op.Event _ -> []
                | Mir_sel.Op.Undef _ -> [])
            | Mir_phys.Instr.Late { op; uses; defs } ->
                F.instructions (env None) op ~uses ~defs
            | Mir_phys.Instr.Remat { instr; defs } -> (
                match instr.Mir_instr.op with
                | Mir_sel.Op.Machine op ->
                    F.instructions
                      (env (Some (Mir_id.Instr.to_int instr.Mir_instr.id)))
                      op ~uses:[] ~defs
                | Mir_sel.Op.Event _ | Mir_sel.Op.Undef _ -> [])
            | Mir_phys.Instr.Move { dst; src; _ } ->
                F.transfer (env None) ~dst ~src
            | Mir_phys.Instr.Save { dst; src } ->
                F.transfer (env None) ~dst ~src
            | Mir_phys.Instr.Sp delta -> F.stack_step (env None) delta)
          @ List.map
              (fun (c : Rivet_cfi.directive) ->
                dir
                  (D.Cfi
                     {
                       name = c.Rivet_cfi.name;
                       argument = c.Rivet_cfi.argument;
                     }))
              (Rivet_cfi.after cfi ~reg i)
        in
        let returns =
          match b.Mir_phys.Block.terminator with
          | Mir_phys.Term.Return _ -> true
          | _ -> false
        in
        let split = Rivet_cfi.epilogue_start b.Mir_phys.Block.body in
        let before, epilogue =
          ( List.filteri (fun k _ -> k < split) b.Mir_phys.Block.body,
            List.filteri (fun k _ -> k >= split) b.Mir_phys.Block.body )
        in
        let cfi_state name = dir (D.Cfi { name; argument = "" }) in
        let saved = Rivet_cfi.cfa cfi in
        let body_items = List.concat_map items_of before in
        let epilogue_items =
          if returns && epilogue <> [] then
            cfi_state ".cfi_remember_state" :: List.concat_map items_of epilogue
          else List.concat_map items_of epilogue
        in
        let restore_after =
          if returns && epilogue <> [] then (
            Rivet_cfi.set_cfa cfi saved;
            [ cfi_state ".cfi_restore_state" ])
          else []
        in
        let falls_to t =
          match following with
          | Some (n : (_, _) Mir_phys.Block.t) ->
              Mir_id.Block.equal n.Mir_phys.Block.id t
          | None -> false
        in
        let terminator =
          match b.Mir_phys.Block.terminator with
          | Mir_phys.Term.Return _ -> [ F.ret (env None) ]
          | Mir_phys.Term.Jump t ->
              if falls_to t then [] else [ F.jump (env None) (label_of f t) ]
          | Mir_phys.Term.Branch { test; uses = _; then_; else_ } ->
              let env = env None in
              if falls_to then_ then
                [ F.branch env test ~label:(label_of f else_) ~inverted:true ]
              else
                F.branch env test ~label:(label_of f then_) ~inverted:false
                ::
                (if falls_to else_ then []
                 else [ F.jump env (label_of f else_) ])
        in
        (N.Label { name = label_of f b.Mir_phys.Block.id; origin } :: body_items)
        @ epilogue_items @ List.map insn terminator @ restore_after)
      next

(* Where a mutable region's bytes live. *)
type binding =
  | Image_resident  (** in the image's own [.bss]: a host reads and writes it *)
  | Table  (** at the caller's address in {!Rivet_table}'s table *)

(* Fault injection for the evidence suite: each is one deliberate defect in the
   entry wrapper that the ABI probe must detect. No consumer passes one. *)
module Entry_mutation = struct
  type t =
    | Clobber_rbx  (** a callee-saved register left changed *)
    | Keep_caller_mxcsr  (** the kernel runs under the caller's MXCSR *)
    | Skip_mxcsr_restore  (** the caller's MXCSR is not put back *)
    | Misalign_stack  (** the kernel is entered with the stack 8 bytes off *)
end

(* The entry of a table-bound image: rbp holds the caller's table for the
   kernel, which keeps no state of its own. rbp is callee-saved and the target
   reserves it, so the kernel and every helper leave it alone; the entry saves
   and restores the caller's. MXCSR is set to the default (round to nearest,
   every exception masked) for the call and put back, because the frame leaves
   it alone. The three pushes keep the kernel's entry at the 8 mod 16 the
   System V ABI gives a callee. *)
let table_entry ~entry_mutation ~kernel =
  let open Asm_core in
  Err.Escape.with_escape @@ fun esc ->
  let env =
    {
      F.mutation = None;
      esc;
      reference = (fun _ -> None);
      table_slot = (fun _ -> None);
    }
  in
  let i mnemonic ops = insn (F.make env mnemonic ops) in
  let cfi name argument = dir (D.Cfi { name; argument }) in
  let reg n = Fam.Operand.Reg (F.find env n) in
  let rsp = F.find env "rsp" in
  let slot disp = F.mem_op ~base:rsp ~disp () in
  let name = Rivet_table.entry in
  let frame =
    if entry_mutation = Some Entry_mutation.Misalign_stack then 24L else 16L
  in
  [
    section ".text" Perms.rx ~nobits:false;
    dir (D.Align { boundary = 16 });
    dir (D.Global { name });
    dir (D.Sym_type { name; kind = D.Function });
    lbl name;
    cfi ".cfi_startproc" "";
    i "pushq" [ reg "rbp" ];
    cfi ".cfi_adjust_cfa_offset" "8";
    cfi ".cfi_offset" "%rbp, -16";
    i "subq" [ F.imm frame; reg "rsp" ];
    cfi ".cfi_adjust_cfa_offset" (Int64.to_string frame);
    i "stmxcsr" [ slot 0L ];
  ]
  @ (if entry_mutation = Some Entry_mutation.Keep_caller_mxcsr then []
     else [ i "movl" [ F.imm 0x1F80L; slot 4L ]; i "ldmxcsr" [ slot 4L ] ])
  @ [
      i "movq" [ reg "rdi"; reg "rbp" ];
      i "call" [ Fam.Operand.Sym (Expr.Symbol kernel) ];
    ]
  @ (if entry_mutation = Some Entry_mutation.Skip_mxcsr_restore then []
     else [ i "ldmxcsr" [ slot 0L ] ])
  @ (if entry_mutation = Some Entry_mutation.Clobber_rbx then
       [ i "movq" [ F.imm 1L; reg "rbx" ] ]
     else [])
  @ [
      i "addq" [ F.imm frame; reg "rsp" ];
      cfi ".cfi_adjust_cfa_offset" (Int64.to_string (Int64.neg frame));
      i "popq" [ reg "rbp" ];
      cfi ".cfi_restore" "%rbp";
      cfi ".cfi_adjust_cfa_offset" "-8";
      i "ret" [];
      cfi ".cfi_endproc" "";
      dir
        (D.Sym_size
           {
             name;
             size =
               Expr.Binary (Expr.Sub, Expr.Current_location, Expr.Symbol name);
           });
    ]

let of_artifact ?entry_mutation ?(binding = Image_resident) artifact =
  Err.Escape.with_escape @@ fun esc ->
  let program = Art.program artifact in
  let relocations = relocation_index artifact in
  let slots = Rivet_table.of_artifact artifact in
  let table_slot =
    match binding with
    | Image_resident -> fun _ -> None
    | Table -> Rivet_table.slot_of_symbol slots
  in
  let text =
    List.concat_map
      (fun (f : (X64_op.t, X64_op.test) Mir_phys.Func.t) ->
        let name = f.Mir_phys.Func.name in
        [
          section ".text" Asm_core.Perms.rx ~nobits:false;
          dir (D.Align { boundary = 16 });
          dir (D.Global { name });
          dir (D.Sym_type { name; kind = D.Function });
          lbl name;
          dir (D.Cfi { name = ".cfi_startproc"; argument = "" });
        ]
        @ instrs_of_func ~table_slot esc relocations f
        @ [
            dir (D.Cfi { name = ".cfi_endproc"; argument = "" });
            dir
              (D.Sym_size
                 {
                   name;
                   size =
                     Asm_core.Expr.Binary
                       ( Asm_core.Expr.Sub,
                         Asm_core.Expr.Current_location,
                         Asm_core.Expr.Symbol name );
                 });
          ])
      program.Mir_phys.Program.funcs
  in
  let data =
    List.concat_map
      (fun (s : Art.Symbol.t) ->
        match (binding, s.Art.Symbol.kind) with
        | ( Table,
            Art.Symbol.Data { section = Art.Section.Bound | Art.Section.Bss; _ }
          ) ->
            []
        | _ -> Option.value ~default:[] (data_items s))
      (Art.symbols artifact)
  in
  let entry =
    match binding with
    | Image_resident -> []
    | Table -> (
        let main =
          (List.find
             (fun (f : (_, _) Mir_phys.Func.t) ->
               Mir_id.Func.equal f.Mir_phys.Func.id
                 program.Mir_phys.Program.main)
             program.Mir_phys.Program.funcs)
            .Mir_phys.Func.name
        in
        match table_entry ~entry_mutation ~kernel:main with
        | Ok items -> items
        | Error e -> Err.Escape.throw_error esc e)
  in
  {
    N.unit_name = "artifact";
    items =
      text @ entry @ data
      @ [ dir (D.Declared_section { name = ".note.GNU-stack" }) ];
  }

(* {1 Host helpers}

   The image linker has no imports, so a helper the artifact calls is defined
   by a stub that loads its address in this process and jumps there. The callee
   returns to the artifact's call site; r11 is the scratch the target reserves
   and the System V ABI leaves caller-saved. *)

let helpers ~host_symbol names =
  Err.Escape.with_escape @@ fun esc ->
  let env =
    {
      F.mutation = None;
      esc;
      reference = (fun _ -> None);
      table_slot = (fun _ -> None);
    }
  in
  let r11 = Fam.Operand.Reg (F.find env "r11") in
  let stub name =
    let addr =
      match host_symbol name with
      | Some a -> a
      | None -> Err.Escape.throw esc (R.Helper name)
    in
    [
      section ".text" Asm_core.Perms.rx ~nobits:false;
      dir (D.Align { boundary = 16 });
      dir (D.Global { name });
      dir (D.Sym_type { name; kind = D.Function });
      lbl name;
      insn (F.make env "movq" [ F.imm addr; r11 ]);
      insn (F.make env "jmp" [ Fam.Operand.Reg (F.find env "r11") ]);
    ]
  in
  {
    N.unit_name = "host";
    items =
      List.concat_map stub names
      @ [ dir (D.Declared_section { name = ".note.GNU-stack" }) ];
  }
