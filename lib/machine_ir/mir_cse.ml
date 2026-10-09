(* Repeated pure computations within a block computed once. An instruction
   with no effect and no order state whose opcode and operands (after earlier
   merges) equal an earlier one's in the same block reads the earlier result
   instead; the duplicate is then dead and a later clean-up removes it. A
   duplicate whose result another block or a terminator reads is kept, so no
   rewrite reaches past the block. The result is verified again. *)

(* Fault injection for the evidence suite. No consumer passes one. *)
module Mutation = struct
  type t =
    | Ignored_constant  (** constants of one type merged whatever the value *)
end

let vkey (v : Mir_value.t) = Mir_id.Value.to_int v.Mir_value.id

(* The values read outside the block that defines them: by a terminator, or by
   any other block. *)
let escaping (blocks : (Mir_op.t, Mir_terminator.t) Mir_block.t list) =
  let escapes = Hashtbl.create 64 in
  List.iter
    (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
      List.iter
        (fun v -> Hashtbl.replace escapes (vkey v) ())
        (Mir_terminator.operands b.Mir_block.terminator);
      let own = Hashtbl.create 16 in
      List.iter
        (fun (i : Mir_op.t Mir_instr.t) ->
          List.iter
            (fun v -> Hashtbl.replace own (vkey v) ())
            i.Mir_instr.results)
        b.Mir_block.body;
      List.iter
        (fun (i : Mir_op.t Mir_instr.t) ->
          List.iter
            (fun v ->
              if not (Hashtbl.mem own (vkey v)) then
                Hashtbl.replace escapes (vkey v) ())
            (Mir_op.operands i.Mir_instr.op))
        b.Mir_block.body)
    blocks;
  escapes

let func ?mutation (f : (Mir_op.t, Mir_terminator.t) Mir_func.t) =
  let escapes = escaping f.Mir_func.blocks in
  let block (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) =
    let subst = Hashtbl.create 16 and seen = Hashtbl.create 16 in
    let rename v =
      match Hashtbl.find_opt subst (vkey v) with Some r -> r | None -> v
    in
    let body =
      List.map
        (fun (i : Mir_op.t Mir_instr.t) ->
          let op = Mir_op.map_operands rename i.Mir_instr.op in
          let i = { i with Mir_instr.op } in
          match i.Mir_instr.results with
          | [ r ]
            when Mir_op.effect_class op = Mir_op.Effect.Pure
                 && i.Mir_instr.order = None -> (
              let key =
                match (mutation, op) with
                | Some Mutation.Ignored_constant, Mir_op.Const c ->
                    Mir_op.Const { c with Mir_const.bits = 0L }
                | _ -> op
              in
              match Hashtbl.find_opt seen key with
              | Some earlier when not (Hashtbl.mem escapes (vkey r)) ->
                  Hashtbl.replace subst (vkey r) earlier;
                  i
              | Some _ -> i
              | None ->
                  Hashtbl.replace seen key r;
                  i)
          | _ -> i)
        b.Mir_block.body
    in
    { b with Mir_block.body }
  in
  {
    f with
    Mir_func.blocks = Mir_offsets.dead_code (List.map block f.Mir_func.blocks);
  }

let program ?mutation (g : Mir_verify.Generic.t) =
  let p = Mir_verify.Generic.program g in
  Err.or_raise ~pp_error:Mir_diagnostic.pp
    (Mir_verify.generic
       {
         p with
         Mir_program.funcs = List.map (func ?mutation) p.Mir_program.funcs;
       })
