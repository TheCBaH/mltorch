(* One form's vectors as one static process: a loop over records, each loading
   its operands into the registers the form's locations name, seeding the flags
   and the destination, running the form's Rivet instructions, and storing the
   results, the flags and the buffer back into the record. The whole block goes
   to standard output. *)

open Machine_ir
open Machine_target_x86_64
module Loc = Mir_phys.Loc
module F = Machine_rivet_x86_64.Rivet_x64_form
module N = Asm_core.Normalized_ast
module D = Asm_core.Directive
module Fam = X86_family_encode
open X64_forms
open X64_model

let stride = 256
let off_in k = 16 * k
let off_seed = 64
let off_flags = 80
let off_buf = 96
let off_out k = 128 + (16 * k)
let off_out_flags = 160
let off_out_base = 168
let records = "records"
let origin = F.origin
let insn i = N.Instruction { insn = i; origin }
let dir directive = N.Directive { directive; origin }
let lbl name = N.Label { name; origin }

let le64 b off v =
  for i = 0 to 7 do
    Bytes.set b (off + i)
      (Char.chr
         (Int64.to_int
            (Int64.logand (Int64.shift_right_logical v (8 * i)) 0xffL)))
  done

let get64 s off = String.get_int64_le s off

(* The pools: the registers inputs and results live in. rax and rdx are the
   fixed registers of IDIV; r12 and r13 the loop's. *)
let gpr_pool = [ 6; 8; 9; 10; 11 ]
let xmm_pool = [ 8; 9; 10 ]
let dest_gpr = 7
let dest_xmm = 11

type placed = {
  locs : Loc.t list;
  kinds : [ `Flags | `Gpr of int | `Xmm of int ] list;
}

let is_xmm (t : Mir_type.t) =
  match t with
  | Mir_type.F32 | Mir_type.F64 | Mir_type.Mask _ | Mir_type.Vec _ -> true
  | _ -> false

(* Registers for the form's uses, in order, with the fixed ones honoured. *)
let place_uses (f : X64_forms.t) op =
  let fixed =
    List.filter_map
      (function
        | Mir_target.Constraint.Fixed_use { use; view } -> Some (use, view)
        | _ -> None)
      (X64_op.constraints op)
  in
  let gp = ref gpr_pool and xp = ref xmm_pool in
  let take r =
    match !r with
    | x :: rest ->
        r := rest;
        x
    | [] -> invalid_arg "X64_batch: out of registers"
  in
  let placed =
    List.mapi
      (fun i role ->
        match role with
        | Flags -> (Loc.Reg X64_reg.rflags, `Flags)
        | Base | Index -> (
            match List.assoc_opt i fixed with
            | Some v ->
                (Loc.Reg v, `Gpr (Mir_id.Unit.to_int v.Mir_target.View.unit))
            | None ->
                let k = take gp in
                (Loc.Reg (X64_reg.q k), `Gpr k))
        | Val ty -> (
            if is_xmm ty then
              let k = take xp in
              (Loc.Reg (X64_reg.xmm k), `Xmm k)
            else
              match List.assoc_opt i fixed with
              | Some v ->
                  (Loc.Reg v, `Gpr (Mir_id.Unit.to_int v.Mir_target.View.unit))
              | None ->
                  let k = take gp in
                  (Loc.Reg (X64_reg.q k), `Gpr k)))
      f.roles
  in
  { locs = List.map fst placed; kinds = List.map snd placed }

let tied_use op =
  List.find_map
    (function
      | Mir_target.Constraint.Tied { result = 0; use } -> Some use | _ -> None)
    (X64_op.constraints op)

(* The result registers: flags, the tied use's, a fixed one, or the scratch
   destination. *)
let place_defs op tys (u : placed) =
  let fixed =
    List.filter_map
      (function
        | Mir_target.Constraint.Fixed_result { result; view } ->
            Some (result, view)
        | _ -> None)
      (X64_op.constraints op)
  in
  let defs =
    List.mapi
      (fun j (ty : Mir_type.t) ->
        match ty with
        | Mir_type.Flags -> (Loc.Reg X64_reg.rflags, `Flags)
        | _ -> (
            match List.assoc_opt j fixed with
            | Some v ->
                (Loc.Reg v, `Gpr (Mir_id.Unit.to_int v.Mir_target.View.unit))
            | None -> (
                match (j, tied_use op) with
                | 0, Some use -> (List.nth u.locs use, List.nth u.kinds use)
                | _ ->
                    if is_xmm ty then
                      (Loc.Reg (X64_reg.xmm dest_xmm), `Xmm dest_xmm)
                    else (Loc.Reg (X64_reg.q dest_gpr), `Gpr dest_gpr))))
      tys
  in
  { locs = List.map fst defs; kinds = List.map snd defs }

type built = { modul : Fam.Instruction.t N.module_; count : int }

let build_inner ?mutation (f : X64_forms.t) (vs : vector list) =
  Err.Escape.with_escape @@ fun esc ->
  let env =
    {
      F.mutation;
      esc;
      reference = (fun _ -> None);
      table_slot = (fun _ -> None);
    }
  in
  let reg n = Fam.Operand.Reg (F.find env n) in
  let r12 = F.find env "r12" in
  let mem off = F.mem_op ~base:r12 ~disp:(Int64.of_int off) () in
  let r mnemonic ops = F.make env mnemonic ops in
  let i mnemonic ops = insn (r mnemonic ops) in
  let op = f.make (roles_values f) in
  let tys =
    match X64_op.typing op with
    | Ok t -> t
    | Error e -> invalid_arg ("X64_batch.build: " ^ e)
  in
  let u = place_uses f op in
  let d = place_defs op tys u in
  let gname k = X64_reg.names64.(k) in
  let xname k = Printf.sprintf "xmm%d" k in
  let loads_i =
    List.concat
      (List.mapi
         (fun slot (role, kind) ->
           match (role, kind) with
           | _, `Flags -> []
           | Base, `Gpr k ->
               [
                 r "leaq" [ mem (off_buf + 8); reg (gname k) ];
                 r "movq" [ reg (gname k); mem off_out_base ];
               ]
           | _, `Gpr k -> [ r "movq" [ mem (off_in slot); reg (gname k) ] ]
           | _, `Xmm k -> [ r "movups" [ mem (off_in slot); reg (xname k) ] ])
         (List.combine f.roles u.kinds))
  in
  let seeds_i =
    let tied = tied_use op in
    List.concat
      (List.mapi
         (fun j (kind : [ `Flags | `Gpr of int | `Xmm of int ]) ->
           let from_use =
             match (j, tied) with 0, Some _ -> true | _ -> false
           in
           let fixed =
             List.exists
               (function
                 | Mir_target.Constraint.Fixed_result { result; _ } ->
                     result = j
                 | _ -> false)
               (X64_op.constraints op)
           in
           if from_use || fixed then []
           else
             match kind with
             | `Flags -> []
             | `Gpr k -> [ r "movq" [ mem off_seed; reg (gname k) ] ]
             | `Xmm k -> [ r "movups" [ mem off_seed; reg (xname k) ] ])
         d.kinds)
  in
  let body = F.instructions env op ~uses:u.locs ~defs:d.locs in
  let saves_i =
    let j = ref 0 in
    List.concat
      (List.map
         (fun (kind : [ `Flags | `Gpr of int | `Xmm of int ]) ->
           match kind with
           | `Flags -> []
           | `Gpr k ->
               let o = off_out !j in
               incr j;
               [ r "movq" [ reg (gname k); mem o ] ]
           | `Xmm k ->
               let o = off_out !j in
               incr j;
               [ r "movups" [ reg (xname k); mem o ] ])
         d.kinds)
  in
  let n = List.length vs in
  let data = Bytes.make (n * stride) '\000' in
  List.iteri
    (fun r (v : vector) ->
      let at = r * stride in
      List.iteri
        (fun slot (role, inp) ->
          match (role, inp) with
          | Index, _ -> le64 data (at + off_in slot) v.index
          | Val _, Some (o : operand) ->
              le64 data (at + off_in slot) o.lo;
              le64 data (at + off_in slot + 8) o.hi
          | _ -> ())
        (List.combine f.roles v.inputs);
      le64 data (at + off_seed) v.seed_lo;
      le64 data (at + off_seed + 8) v.seed_hi;
      le64 data (at + off_flags) (rflags_image v.flags);
      Bytes.blit v.buf 0 data (at + off_buf) buf_bytes)
    vs;
  let total = n * stride in
  let text =
    [
      dir
        (D.Section { name = ".text"; perms = Asm_core.Perms.rx; nobits = false });
      dir (D.Align { boundary = 16 });
      dir (D.Global { name = "_start" });
      lbl "_start";
      i "leaq"
        [
          Fam.Operand.Mem
            {
              Fam.Mem.base = Some Fam.rip_reg;
              index = None;
              scale = 1;
              disp = Fam.Disp.Sym (Asm_core.Expr.Symbol records);
            };
          reg "r12";
        ];
      i "movq" [ F.imm (Int64.of_int n); reg "r13" ];
      lbl "loop";
      i "movq" [ mem off_flags; reg "rax" ];
      i "pushq" [ reg "rax" ];
      i "popfq" [];
    ]
    @ List.map insn (loads_i @ seeds_i @ body @ saves_i)
    @ [
        i "pushfq" [];
        i "popq" [ reg "rax" ];
        i "movq" [ reg "rax"; mem off_out_flags ];
        i "addq" [ F.imm (Int64.of_int stride); reg "r12" ];
        i "subq" [ F.imm 1L; reg "r13" ];
        i "jne" [ Fam.Operand.Sym (Asm_core.Expr.Symbol "loop") ];
        i "movl" [ F.imm 1L; reg "eax" ];
        i "movl" [ F.imm 1L; reg "edi" ];
        i "leaq"
          [
            Fam.Operand.Mem
              {
                Fam.Mem.base = Some Fam.rip_reg;
                index = None;
                scale = 1;
                disp = Fam.Disp.Sym (Asm_core.Expr.Symbol records);
              };
            reg "rsi";
          ];
        i "movl" [ F.imm (Int64.of_int total); reg "edx" ];
        i "syscall" [];
        i "movl" [ F.imm 60L; reg "eax" ];
        i "xorl" [ reg "edi"; reg "edi" ];
        i "syscall" [];
      ]
  in
  let words =
    List.init (total / 8) (fun w ->
        Asm_core.Expr.Const
          (Foundation.Bigint.of_int64 (Bytes.get_int64_le data (w * 8))))
  in
  let data_items =
    [
      dir
        (D.Section { name = ".data"; perms = Asm_core.Perms.rw; nobits = false });
      dir (D.Align { boundary = 16 });
      dir (D.Global { name = records });
      lbl records;
      dir (D.Data { width = 8; values = words });
    ]
  in
  let modul =
    {
      N.unit_name = "batch";
      items =
        text @ data_items
        @ [ dir (D.Declared_section { name = ".note.GNU-stack" }) ];
    }
  in
  Ok { modul; count = n }

let build ?mutation f vs =
  match Err.payload (build_inner ?mutation f vs) with
  | Ok (Ok b) -> Ok b
  | Ok (Error e) -> Error e
  | Error r -> Error (Fmt.str "%a" Machine_rivet_x86_64.Rivet_x64_refusal.pp r)

let elf built =
  Machine_rivet_x86_64.Rivet_x64_process.elf_of ~entry:"_start" [ built.modul ]

(* The observed record [r] of the output block. *)
type observed = {
  outs : (int64 * int64) list;
  rflags : int64;
  base : int64;
  buffer : int array;
}

let observe (d : placed) (out : string) r =
  let at = r * stride in
  let nres =
    List.length (List.filter (function `Flags -> false | _ -> true) d.kinds)
  in
  {
    outs =
      List.init nres (fun j ->
          (get64 out (at + off_out j), get64 out (at + off_out j + 8)));
    rflags = get64 out (at + off_out_flags);
    base = get64 out (at + off_out_base);
    buffer = Array.init buf_bytes (fun j -> Char.code out.[at + off_buf + j]);
  }

let result_places (f : X64_forms.t) =
  let op = f.make (roles_values f) in
  let tys = Result.get_ok (X64_op.typing op) in
  (tys, place_defs op tys (place_uses f op))
