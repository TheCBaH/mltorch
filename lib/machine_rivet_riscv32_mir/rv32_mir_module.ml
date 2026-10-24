(* A published RV32 artifact as one typed Rivet module: every function's text,
   the entry the process harness calls, and the constant regions. The
   artifact is table-bound: each mutable region is addressed through the table
   of region addresses s11 holds, which the entry takes from a0, so the image
   holds code and constants only and the process maps the tensors.

   RV32 has no short and long encoding of one branch, and Rivet no branch
   relaxation for it, so each function is laid out here: every item has a
   known size, a conditional branch whose target is beyond a B-type's reach
   becomes the inverted branch over a [jal], and the layout is repeated until
   no branch changes. *)

open Machine_ir
open Machine_target_riscv32
module Art = Machine_model.Mir_artifact
module Table = Machine_rivet_common.Rivet_table
module A = Machine_rivet_riscv32.Rv32_asm
module F = Rv32_mir_form
module Item = F.Item

let label_of (f : (_, _) Mir_phys.Func.t) b =
  Printf.sprintf ".L%s_b%d" f.Mir_phys.Func.name (Mir_id.Block.to_int b)

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

(* A B-type branch reaches [-4096, 4094] bytes; the margin keeps a promotion
   from pushing a neighbour out of reach within the same round. *)
let reach = 4000

(* The items with every conditional branch resolved, short or long. *)
let lay_out items =
  let long = Hashtbl.create 16 in
  let rec round () =
    let at = Hashtbl.create 64 in
    let pos = ref 0 in
    List.iteri
      (fun k it ->
        (match it with Item.Label l -> Hashtbl.replace at l !pos | _ -> ());
        pos :=
          !pos
          +
          match it with
          | Item.Branch _ when Hashtbl.mem long k -> 8
          | it -> Item.bytes it)
      items;
    let pos = ref 0 and changed = ref false in
    List.iteri
      (fun k it ->
        (match it with
        | Item.Branch { label; _ } when not (Hashtbl.mem long k) -> (
            match Hashtbl.find_opt at label with
            | Some target when abs (target - !pos) <= reach -> ()
            | _ ->
                Hashtbl.replace long k ();
                changed := true)
        | _ -> ());
        pos :=
          !pos
          +
          match it with
          | Item.Branch _ when Hashtbl.mem long k -> 8
          | it -> Item.bytes it)
      items;
    if !changed then round ()
  in
  round ();
  List.concat
    (List.mapi
       (fun k it ->
         match it with
         | Item.Branch { mnemonic; regs; label } when Hashtbl.mem long k ->
             let over = Printf.sprintf "%s_far%d" label k in
             [
               Item.Branch
                 { mnemonic = F.invert_mnemonic mnemonic; regs; label = over };
               Item.Insn { mnemonic = "j"; ops = [ A.sym label ] };
               Item.Label over;
             ]
         | it -> [ it ])
       items)

let render items =
  List.map
    (function
      | Item.Label l -> A.label l
      | Item.Insn { mnemonic; ops } -> A.insn mnemonic ops
      | Item.Branch { mnemonic; regs; label } ->
          A.insn mnemonic (List.map A.reg regs @ [ A.sym label ]))
    items

(* Whether an instruction is a call, which leaves the vector configuration
   undefined. *)
let calls (i : Rv32_op.t Mir_phys.Instr.t) =
  match i with
  | Mir_phys.Instr.Exec
      { instr = { Mir_instr.op = Mir_sel.Op.Machine (Rv32_op.Call _); _ }; _ }
    ->
      true
  | _ -> false

(* Where a block's items need a [vsetivli]: for each item's configuration
   requirement and whether it is a call, the configuration to establish just
   before it. The configuration is unknown at the block's start and after a
   call; an item needing one that is not known to hold establishes it (every
   form that changes it on its way restores LMUL 1, its own requirement). *)
let configurations ?mutation (items : (F.Vconfig.t option * bool) list) =
  let state = ref None and set_once = ref false in
  List.map
    (fun (requirement, call) ->
      let set =
        match requirement with
        | Some c when !state <> Some c ->
            if mutation = Some F.Mutation.Unset_vtype then None
            else if mutation = Some F.Mutation.Stale_vtype && !set_once then
              None
            else (
              state := Some c;
              set_once := true;
              Some c)
        | Some _ | None -> None
      in
      if call then state := None;
      set)
    items

let func_items env relocations (f : (Rv32_op.t, Rv32_op.test) Mir_phys.Func.t) =
  let outer = env in
  let fid = Mir_id.Func.to_int f.Mir_phys.Func.id in
  let blocks = f.Mir_phys.Func.blocks in
  let rec pairs = function
    | a :: (b :: _ as rest) -> (a, Some b) :: pairs rest
    | [ a ] -> [ (a, None) ]
    | [] -> []
  in
  let entry_jump =
    match blocks with
    | b :: _
      when not (Mir_id.Block.equal b.Mir_phys.Block.id f.Mir_phys.Func.entry) ->
        [ F.i "j" [ A.sym (label_of f f.Mir_phys.Func.entry) ] ]
    | _ -> []
  in
  entry_jump
  @ List.concat_map
      (fun ((b : (_, _) Mir_phys.Block.t), following) ->
        let bid = Mir_id.Block.to_int b.Mir_phys.Block.id in
        let env instr =
          {
            env with
            F.reference =
              (fun reference ->
                List.find_map
                  (fun (r : Art.Relocation.t) ->
                    if r.Art.Relocation.reference = reference then
                      Some (r.Art.Relocation.symbol, r.Art.Relocation.addend)
                    else None)
                  (Hashtbl.find_all relocations (fid, bid, instr)));
          }
        in
        let of_instr (i : Rv32_op.t Mir_phys.Instr.t) =
          match i with
          | Mir_phys.Instr.Exec { instr; uses; defs } -> (
              match instr.Mir_instr.op with
              | Mir_sel.Op.Machine op ->
                  F.items
                    (env (Some (Mir_id.Instr.to_int instr.Mir_instr.id)))
                    op ~uses ~defs
              | Mir_sel.Op.Event _ | Mir_sel.Op.Undef _ -> [])
          | Mir_phys.Instr.Late { op; uses; defs } ->
              F.items (env None) op ~uses ~defs
          | Mir_phys.Instr.Remat { instr; defs } -> (
              match instr.Mir_instr.op with
              | Mir_sel.Op.Machine op ->
                  F.items
                    (env (Some (Mir_id.Instr.to_int instr.Mir_instr.id)))
                    op ~uses:[] ~defs
              | Mir_sel.Op.Event _ | Mir_sel.Op.Undef _ -> [])
          | Mir_phys.Instr.Move { dst; src; _ }
          | Mir_phys.Instr.Save { dst; src } ->
              F.transfer (env None) ~dst ~src
          | Mir_phys.Instr.Sp delta -> F.stack_step delta
        in
        let falls_to t =
          match following with
          | Some (n : (_, _) Mir_phys.Block.t) ->
              Mir_id.Block.equal n.Mir_phys.Block.id t
          | None -> false
        in
        let terminator =
          match b.Mir_phys.Block.terminator with
          | Mir_phys.Term.Return _ -> [ F.i "ret" [] ]
          | Mir_phys.Term.Jump t ->
              if falls_to t then [] else [ F.i "j" [ A.sym (label_of f t) ] ]
          | Mir_phys.Term.Branch { test; uses; then_; else_ } ->
              let env = env None in
              if falls_to then_ then
                [
                  F.branch env test ~uses ~label:(label_of f else_)
                    ~inverted:true;
                ]
              else
                F.branch env test ~uses ~label:(label_of f then_)
                  ~inverted:false
                ::
                (if falls_to else_ then []
                 else [ F.i "j" [ A.sym (label_of f else_) ] ])
        in
        let body = b.Mir_phys.Block.body in
        let settings =
          configurations ?mutation:outer.F.mutation
            (List.map (fun i -> (F.requirement i, calls i)) body)
        in
        let body =
          List.concat
            (List.map2
               (fun i set ->
                 (match set with
                   | Some c -> [ F.vsetivli ~config:c () ]
                   | None -> [])
                 @ of_instr i)
               body settings)
        in
        (Item.Label (label_of f b.Mir_phys.Block.id) :: body) @ terminator)
      (pairs blocks)

(* The entry the harness calls with the table in a0: s11 and the link
   register kept, s11 set to the table, the kernel called, its status left in
   a0. *)
let entry_items ~kernel =
  let open A in
  [
    text;
    align 4;
    global Table.entry;
    label Table.entry;
    insn "addi" [ reg "sp"; reg "sp"; imm (-16) ];
    insn "sw" [ reg "ra"; mem "sp" 12 ];
    insn "sw" [ reg "s11"; mem "sp" 8 ];
    insn "mv" [ reg "s11"; reg "a0" ];
    insn "call" [ sym kernel ];
    insn "lw" [ reg "s11"; mem "sp" 8 ];
    insn "lw" [ reg "ra"; mem "sp" 12 ];
    insn "addi" [ reg "sp"; reg "sp"; imm 16 ];
    insn "ret" [];
  ]

let const_expr n = Asm_core.Expr.Const (Foundation.Bigint.of_int64 n)

let data_items (s : Art.Symbol.t) =
  match s.Art.Symbol.kind with
  | Art.Symbol.Data { align; section = Art.Section.Rodata bytes; _ } ->
      let name = s.Art.Symbol.name in
      [
        A.section ".rodata" Asm_core.Perms.ro false;
        A.align (Int64.to_int align);
        A.global name;
        A.label name;
        A.dir
          (Asm_core.Directive.Data
             {
               width = 1;
               values =
                 List.init (String.length bytes) (fun i ->
                     const_expr (Int64.of_int (Char.code bytes.[i])));
             });
      ]
  | Art.Symbol.Data _ | Art.Symbol.Function _ -> []
  | Art.Symbol.External_function _ | Art.Symbol.Generated_function _ -> []

(* The artifact's slots, the table its entry receives. *)
let slots artifact = Table.of_artifact artifact

(* Every function's items, laid out, under one escape: what [of_artifact]
   renders and what a census counts. *)
let laid_out ?mutation esc artifact =
  let program = Art.program artifact in
  List.iter
    (fun (s : Art.Symbol.t) ->
      match s.Art.Symbol.kind with
      | Art.Symbol.External_function n ->
          Err.Escape.throw esc (F.Refusal.External n)
      | _ -> ())
    (Art.symbols artifact);
  let relocations = relocation_index artifact in
  let table = slots artifact in
  let labels = ref 0 in
  let env =
    {
      F.mutation;
      esc;
      reference = (fun _ -> None);
      table_slot = Table.slot_of_symbol table;
      fresh_label =
        (fun () ->
          incr labels;
          Printf.sprintf ".Lsel%d" !labels);
    }
  in
  List.map
    (fun (f : (Rv32_op.t, Rv32_op.test) Mir_phys.Func.t) ->
      (f.Mir_phys.Func.name, lay_out (func_items env relocations f)))
    program.Mir_phys.Program.funcs

let function_items ?mutation artifact =
  Err.Escape.with_escape @@ fun esc -> laid_out ?mutation esc artifact

let of_artifact ?mutation artifact =
  Err.Escape.with_escape @@ fun esc ->
  let program = Art.program artifact in
  let laid = laid_out ?mutation esc artifact in
  (* rendering makes the typed instructions, so a form Rivet refuses is
     caught by [module_] *)
  let text () =
    List.concat_map
      (fun (name, items) ->
        [ A.text; A.align 4; A.global name; A.label name ] @ render items)
      laid
  in
  let main =
    (List.find
       (fun (f : (_, _) Mir_phys.Func.t) ->
         Mir_id.Func.equal f.Mir_phys.Func.id program.Mir_phys.Program.main)
       program.Mir_phys.Program.funcs)
      .Mir_phys.Func.name
  in
  let data = List.concat_map data_items (Art.symbols artifact) in
  match
    A.module_ "artifact" (fun () -> text () @ entry_items ~kernel:main @ data)
  with
  | Ok m -> m
  | Error what -> Err.Escape.throw esc (F.Refusal.Unencodable what)
