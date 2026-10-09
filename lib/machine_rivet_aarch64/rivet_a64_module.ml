(* A published AArch64 artifact as one typed Rivet module: the functions'
   text, each data symbol in its section, and nothing else. Blocks are labels
   in function order; a branch whose target follows falls through. A host
   helper's trampoline is a second module ({!helpers}), because its address
   belongs to a process and the artifact's must not. *)

open Machine_ir
open Machine_target_aarch64
module Art = Machine_model.Mir_artifact
module Loc = Mir_phys.Loc
module R = Rivet_a64_refusal
module F = Rivet_a64_form
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

let instrs_of_func ?mutation ~table_slot esc relocations
    (f : (A64_op.t, A64_op.test) Mir_phys.Func.t) =
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
  let entry_jump =
    match blocks with
    | b :: _
      when not (Mir_id.Block.equal b.Mir_phys.Block.id f.Mir_phys.Func.entry) ->
        [ insn (F.jump (label_of f f.Mir_phys.Func.entry)) ]
    | _ -> []
  in
  entry_jump
  @ List.concat_map
      (fun ((b : (_, _) Mir_phys.Block.t), following) ->
        let bid = Mir_id.Block.to_int b.Mir_phys.Block.id in
        let env instr =
          {
            F.mutation;
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
        let body =
          List.concat_map
            (fun (i : A64_op.t Mir_phys.Instr.t) ->
              match i with
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
                  F.transfer ?mutation (env None) ~dst ~src
              | Mir_phys.Instr.Save { dst; src } ->
                  F.transfer ?mutation ~save:true (env None) ~dst ~src
              | Mir_phys.Instr.Sp delta -> F.stack_step (env None) delta)
            b.Mir_phys.Block.body
        in
        let falls_to t =
          match following with
          | Some (n : (_, _) Mir_phys.Block.t) ->
              Mir_id.Block.equal n.Mir_phys.Block.id t
          | None -> false
        in
        let terminator =
          match b.Mir_phys.Block.terminator with
          | Mir_phys.Term.Return _ -> [ F.ret ]
          | Mir_phys.Term.Jump t ->
              if falls_to t then [] else [ F.jump (label_of f t) ]
          | Mir_phys.Term.Branch { test; uses; then_; else_ } ->
              let env = env None in
              if falls_to then_ then
                [
                  F.branch env test ~uses ~label:(label_of f else_)
                    ~inverted:(mutation <> Some F.Mutation.Branch_sense);
                ]
              else
                F.branch env test ~uses ~label:(label_of f then_)
                  ~inverted:false
                ::
                (if falls_to else_ then [] else [ F.jump (label_of f else_) ])
        in
        N.Label { name = label_of f b.Mir_phys.Block.id; origin }
        :: List.map insn (body @ terminator))
      next

(* Where a mutable region's bytes live. *)
type binding =
  | Image_resident  (** in the image's own [.bss]: a host reads and writes it *)
  | Table  (** at the caller's address in {!Rivet_a64_table}'s table *)

(* The entry of a table-bound image: x18 holds the caller's table for the
   kernel, which keeps no state of its own; x18 and the link register are put
   back on the way out. *)
let table_entry ~kernel =
  let open Asm_core in
  let i op ops = insn (F.ins op ops) in
  let sp =
    Aarch64.Operand.Reg { Aarch64.Reg.num = 31; width = 64; is_sp = true }
  in
  let x n =
    Aarch64.Operand.Reg { Aarch64.Reg.num = n; width = 64; is_sp = false }
  in
  let slot n =
    Aarch64.Operand.Mem
      {
        Aarch64.Mem.base = { Aarch64.Reg.num = 31; width = 64; is_sp = true };
        offset = Aarch64.Disp.Const (Int64.of_int n);
        writeback = false;
        pre = true;
      }
  in
  let name = Rivet_a64_table.entry in
  [
    section ".text" Perms.rx ~nobits:false;
    dir (D.Align { boundary = 4 });
    dir (D.Global { name });
    dir (D.Sym_type { name; kind = D.Function });
    lbl name;
    i Aarch64.Opcode.Sub [ sp; sp; F.imm 32L ];
    i Aarch64.Opcode.Str [ x 30; slot 0 ];
    i Aarch64.Opcode.Str [ x 18; slot 8 ];
    i Aarch64.Opcode.Mov [ x 18; x 0 ];
    i Aarch64.Opcode.Bl [ Aarch64.Operand.Sym (Expr.Symbol kernel) ];
    i Aarch64.Opcode.Ldr [ x 18; slot 8 ];
    i Aarch64.Opcode.Ldr [ x 30; slot 0 ];
    i Aarch64.Opcode.Add [ sp; sp; F.imm 32L ];
    i Aarch64.Opcode.Ret [];
    dir
      (D.Sym_size
         {
           name;
           size = Expr.Binary (Expr.Sub, Expr.Current_location, Expr.Symbol name);
         });
  ]

let of_artifact ?mutation ?(binding = Image_resident) artifact =
  Err.Escape.with_escape @@ fun esc ->
  let program = Art.program artifact in
  let relocations = relocation_index artifact in
  let slots = Rivet_a64_table.of_artifact artifact in
  let table_slot =
    match binding with
    | Image_resident -> fun _ -> None
    | Table -> Rivet_a64_table.slot_of_symbol slots
  in
  let text =
    List.concat_map
      (fun (f : (A64_op.t, A64_op.test) Mir_phys.Func.t) ->
        let name = f.Mir_phys.Func.name in
        [
          section ".text" Asm_core.Perms.rx ~nobits:false;
          dir (D.Align { boundary = 4 });
          dir (D.Global { name });
          dir (D.Sym_type { name; kind = D.Function });
          lbl name;
        ]
        @ instrs_of_func ?mutation ~table_slot esc relocations f
        @ [
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
    | Table ->
        let main =
          (List.find
             (fun (f : (_, _) Mir_phys.Func.t) ->
               Mir_id.Func.equal f.Mir_phys.Func.id
                 program.Mir_phys.Program.main)
             program.Mir_phys.Program.funcs)
            .Mir_phys.Func.name
        in
        table_entry ~kernel:main
  in
  {
    N.unit_name = "artifact";
    items =
      text @ entry @ data
      @ [ dir (D.Declared_section { name = ".note.GNU-stack" }) ];
  }

(* {1 Host helpers}

   The image linker has no imports, so a helper the artifact calls is defined
   by a stub that loads its address in this process and jumps there. The
   callee returns to the artifact's caller-saved link register unchanged;
   x16 is the scratch the AAPCS64 reserves for exactly this. *)

let helpers ~host_symbol names =
  Err.Escape.with_escape @@ fun esc ->
  let x16 =
    Aarch64.Operand.Reg { Aarch64.Reg.num = 16; width = 64; is_sp = false }
  in
  let stub name =
    let addr =
      match host_symbol name with
      | Some a -> a
      | None -> Err.Escape.throw esc (R.Host_symbol name)
    in
    let quarter k =
      Int64.logand (Int64.shift_right_logical addr (16 * k)) 0xffffL
    in
    let ins = F.ins in
    [
      section ".text" Asm_core.Perms.rx ~nobits:false;
      dir (D.Align { boundary = 4 });
      dir (D.Global { name });
      dir (D.Sym_type { name; kind = D.Function });
      lbl name;
      insn (ins Aarch64.Opcode.Movz [ x16; F.imm (quarter 0) ]);
    ]
    @ List.map
        (fun k ->
          insn
            (ins Aarch64.Opcode.Movk
               [
                 x16;
                 F.imm (quarter k);
                 Aarch64.Operand.Shift
                   { Aarch64.Shift.kind = "lsl"; amount = 16 * k };
               ]))
        [ 1; 2; 3 ]
    @ [ insn (ins Aarch64.Opcode.Br [ x16 ]) ]
  in
  {
    N.unit_name = "host";
    items =
      List.concat_map stub names
      @ [ dir (D.Declared_section { name = ".note.GNU-stack" }) ];
  }
