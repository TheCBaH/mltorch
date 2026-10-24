(* What a published RV32 artifact emits, counted statically: its vector
   instructions by mnemonic, its vector configurations ([vsetivli]/[vsetvli]),
   the allocator's spills, reloads and rematerializations, the realized frame
   bytes and the helper calls. A pure function of the artifact, so a
   selection change shows up here without a simulator. *)

open Machine_ir
open Machine_target_riscv32
module Art = Machine_model.Mir_artifact
module Item = Rv32_mir_form.Item

module Census = struct
  type t = {
    instructions : int;  (** every emitted instruction, branches included *)
    vector : (string * int) list;
        (** vector instructions other than configurations, by mnemonic,
            alphabetical *)
    configurations : int;
    spills : int;  (** moves to frame memory *)
    reloads : int;  (** moves from frame memory *)
    remats : int;
    frame_bytes : int64;  (** summed over the functions *)
    helper_calls : int;
  }

  let pp fmt t =
    Fmt.pf fmt
      "@[<v>instructions %d, configurations %d, spills %d, reloads %d, remats \
       %d, frame %Ld bytes, helper calls %d@,\
       vector %a@]"
      t.instructions t.configurations t.spills t.reloads t.remats t.frame_bytes
      t.helper_calls
      Fmt.(
        list ~sep:(any ", ") (fun fmt (m, n) -> Fmt.pf fmt "%s %d" m n)
        |> fun pp fmt -> function
        | [] -> Fmt.string fmt "none" | l -> pp fmt l)
      t.vector
end

let configuration m = String.starts_with ~prefix:"vset" m

let vector_mnemonic m =
  String.length m > 0 && m.[0] = 'v' && not (configuration m)

let in_memory (l : Mir_phys.Loc.t) =
  match l with Mir_phys.Loc.Reg _ -> false | Slot _ | Mem _ -> true

let helper_call (i : Rv32_op.t Mir_phys.Instr.t) =
  match i with
  | Mir_phys.Instr.Exec
      {
        instr =
          {
            Mir_instr.op =
              Mir_sel.Op.Machine
                (Rv32_op.Call { callee = Mir_op.Callee.Helper _; _ });
            _;
          };
        _;
      } ->
      true
  | _ -> false

let of_artifact artifact =
  let open Err.Syntax in
  let+ laid = Rv32_mir_module.function_items artifact in
  let program = Art.program artifact in
  let mnemonics =
    List.concat_map
      (fun (_, items) ->
        List.filter_map
          (function
            | Item.Insn { mnemonic; _ } | Item.Branch { mnemonic; _ } ->
                Some mnemonic
            | Item.Label _ -> None)
          items)
      laid
  in
  let tally = Hashtbl.create 16 in
  List.iter
    (fun m ->
      if vector_mnemonic m then
        Hashtbl.replace tally m
          (1 + Option.value ~default:0 (Hashtbl.find_opt tally m)))
    mnemonics;
  let instrs =
    List.concat_map
      (fun (f : (_, _) Mir_phys.Func.t) ->
        List.concat_map
          (fun (b : (_, _) Mir_phys.Block.t) -> b.Mir_phys.Block.body)
          f.Mir_phys.Func.blocks)
      program.Mir_phys.Program.funcs
  in
  let count p = List.length (List.filter p instrs) in
  {
    Census.instructions = List.length mnemonics;
    vector =
      List.sort
        (fun (a, _) (b, _) -> String.compare a b)
        (List.of_seq (Hashtbl.to_seq tally));
    configurations = List.length (List.filter configuration mnemonics);
    spills =
      count (function
        | Mir_phys.Instr.Move { dst; _ } -> in_memory dst
        | _ -> false);
    reloads =
      count (function
        | Mir_phys.Instr.Move { src; _ } -> in_memory src
        | _ -> false);
    remats = count (function Mir_phys.Instr.Remat _ -> true | _ -> false);
    frame_bytes =
      List.fold_left
        (fun n (f : (_, _) Mir_phys.Func.t) ->
          Int64.add n (Option.value ~default:0L f.Mir_phys.Func.frame))
        0L program.Mir_phys.Program.funcs;
    helper_calls = count helper_call;
  }
