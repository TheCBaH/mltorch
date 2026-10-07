(* Frame realization: abstract slots become stack-pointer-relative memory.
   Layout assigns each slot and save word an aligned offset in a frame whose
   size keeps the stack aligned; the prologue moves the stack pointer down in
   encodable steps and saves the link register (when the function calls),
   every callee-saved register the function writes, and — in the entry
   function, when the target has FP control state — the caller's FP controls,
   which it then sets to the modelled state; an epilogue before every return,
   success or failure alike, restores all of it. A frame offset that no access
   encodes is expanded through the target's reserved scratch: its address is
   computed by a late form, and the access uses that register as its base. *)

open Machine_ir
module Loc = Mir_phys.Loc

(* What frame realization needs of a target beyond its selected interface. *)
module type FRAME = sig
  type op

  val scratch : Mir_target.View.t
  (** reserved: a large offset's address *)

  val control_scratch : Mir_target.View.t
  (** reserved: FP controls in transit *)

  val add_imm : Mir_value.t -> int64 -> op
  (** a pointer plus an immediate the target encodes for any 4 KiB multiple *)

  val fp_control :
    (Mir_target.View.t * (Mir_value.t -> op) * (Mir_value.t -> op) * op) option
  (** the control register, a read of it, a write of it, and a zero *)
end

module Refusal = struct
  type t = Frame_too_large of Mir_id.Func.t

  let pp fmt = function
    | Frame_too_large f ->
        Fmt.pf fmt "%a: a frame beyond the supported code model" Mir_id.Func.pp
          f
end

(* Fault injection for the evidence suite; no consumer passes one. *)
module Mutation = struct
  type t =
    | Allocatable_scratch
        (** a large offset's address through an allocatable register *)
    | Epilogue_once  (** only the first return in block order restores *)
    | Misalign  (** the frame size not rounded to the stack alignment *)
    | Narrow_save  (** callee-saved registers saved at half width *)
    | No_control_restore  (** the caller's FP controls never restored *)
    | No_link_save  (** the link register not saved around calls *)
    | Unexpanded  (** large offsets left as they are *)
end

module Make (T : Mir_sel.TARGET) (F : FRAME with type op = T.op) = struct
  let placeholder k ty =
    { Mir_value.id = Mir_id.Value.of_int (1_000_000 + k); ty }

  type area = {
    key : [ `Control | `Save of Mir_target.View.t | `Slot of Mir_id.Slot.t ];
    bytes : int64;
    align : int64;
  }

  let realize_func ?mutation ~pad ~main (f : (T.op, T.test) Mir_phys.Func.t) =
    let mutated m = mutation = Some m in
    let body_instrs =
      List.concat_map
        (fun (b : (_, _) Mir_phys.Block.t) -> b.Mir_phys.Block.body)
        f.Mir_phys.Func.blocks
    in
    let written =
      List.concat_map
        (function
          | Mir_phys.Instr.Exec { defs; _ }
          | Mir_phys.Instr.Late { defs; _ }
          | Mir_phys.Instr.Remat { defs; _ } ->
              defs
          | Mir_phys.Instr.Move { dst; _ } -> [ dst ]
          | Mir_phys.Instr.Save _ | Mir_phys.Instr.Sp _ -> [])
        body_instrs
      @ List.map snd f.Mir_phys.Func.params
    in
    let writes_unit (v : Mir_target.View.t) =
      List.exists
        (function
          | Loc.Reg w ->
              Mir_id.Unit.equal w.Mir_target.View.unit v.Mir_target.View.unit
          | Loc.Slot _ | Loc.Mem _ -> false)
        written
    in
    let calls =
      List.exists
        (function
          | Mir_phys.Instr.Exec
              { instr = { Mir_instr.op = Mir_sel.Op.Machine op; _ }; _ } -> (
              match T.link with
              | Some l ->
                  List.exists (Mir_target.View.overlap l) (T.clobbers op)
              | None -> false)
          | _ -> false)
        body_instrs
    in
    let saves =
      List.filter
        (fun (v : Mir_target.View.t) ->
          v.Mir_target.View.bank <> Mir_target.Bank.Control && writes_unit v)
        T.abi.Mir_target.Abi.preserved
      @
      match T.link with
      | Some l when calls && not (mutated Mutation.No_link_save) -> [ l ]
      | _ -> []
    in
    let control = if main then F.fp_control else None in
    let areas =
      List.map
        (fun (s : Mir_phys.Slot.t) ->
          {
            key = `Slot s.Mir_phys.Slot.id;
            bytes = s.Mir_phys.Slot.bytes;
            align = s.Mir_phys.Slot.align;
          })
        f.Mir_phys.Func.slots
      @ List.map
          (fun (v : Mir_target.View.t) ->
            {
              key = `Save v;
              bytes = Int64.of_int (v.Mir_target.View.bits / 8);
              align = 8L;
            })
          saves
      @
      match control with
      | Some _ -> [ { key = `Control; bytes = 8L; align = 8L } ]
      | None -> []
    in
    let areas =
      List.stable_sort (fun a b -> Int64.compare b.align a.align) areas
    in
    let cursor = ref pad and offsets = ref [] in
    List.iter
      (fun a ->
        let at = Option.get (Mir_layout.align_up !cursor ~align:a.align) in
        offsets := (a.key, at) :: !offsets;
        cursor := Int64.add at a.bytes)
      areas;
    let align = T.abi.Mir_target.Abi.stack_align in
    (* the frame and a pushed return address together keep the stack
       aligned *)
    let aligned =
      Int64.sub
        (Option.get
           (Mir_layout.align_up (Int64.add !cursor T.call_push) ~align))
        T.call_push
    in
    (* the mutation moves the frame off alignment by one word *)
    let size =
      if mutated Mutation.Misalign then Int64.add aligned 8L else aligned
    in
    let offset key = List.assoc key !offsets in
    (* stack-pointer steps: a 4 KiB-multiple part, then the rest *)
    let steps_of size sign =
      let hi = Int64.logand size (Int64.lognot 0xFFFL)
      and lo = Int64.logand size 0xFFFL in
      List.filter_map
        (fun x ->
          if Int64.equal x 0L then None
          else Some (Mir_phys.Instr.Sp (Int64.mul sign x)))
        [ hi; lo ]
    in
    let steps = steps_of size in
    (* the code model: the aligned frame's steps must encode *)
    let too_large =
      not
        (List.for_all
           (function Mir_phys.Instr.Sp d -> T.stack_step_ok d | _ -> true)
           (steps_of aligned 1L))
    in
    (* a frame access at [off]: direct, or through the reserved scratch *)
    let access ~bytes off =
      if T.frame_offset_ok ~bytes off || mutated Mutation.Unexpanded then
        ([], Loc.Mem { base = T.stack_pointer; offset = off; bytes })
      else
        let hi = Int64.logand off (Int64.lognot 0xFFFL)
        and lo = Int64.logand off 0xFFFL in
        let base =
          if mutated Mutation.Allocatable_scratch then
            {
              F.scratch with
              Mir_target.View.name = "x9";
              unit = Mir_id.Unit.of_int 9;
            }
          else F.scratch
        in
        ( [
            Mir_phys.Instr.Late
              {
                op = F.add_imm (placeholder 0 Mir_type.Ptr) hi;
                uses = [ Loc.Reg T.stack_pointer ];
                defs = [ Loc.Reg base ];
              };
          ],
          Loc.Mem { base; offset = lo; bytes } )
    in
    let rewrite_loc (l : Loc.t) =
      match l with
      | Loc.Slot { slot; bytes } -> access ~bytes (offset (`Slot slot))
      | Loc.Reg _ | Loc.Mem _ -> ([], l)
    in
    let rewrite = function
      | Mir_phys.Instr.Move { dst; src; value } ->
          let pd, dst = rewrite_loc dst and ps, src = rewrite_loc src in
          pd @ ps @ [ Mir_phys.Instr.Move { dst; src; value } ]
      | i -> [ i ]
    in
    (* a saved register's word: the register whole, or half of it under the
       mutation *)
    let save_word (v : Mir_target.View.t) =
      let pre, mem =
        access
          ~bytes:(Int64.of_int (v.Mir_target.View.bits / 8))
          (offset (`Save v))
      in
      if mutated Mutation.Narrow_save then
        let half =
          { v with Mir_target.View.bits = v.Mir_target.View.bits / 2 }
        in
        match mem with
        | Loc.Mem m ->
            (pre, half, Loc.Mem { m with bytes = Int64.div m.bytes 2L })
        | _ -> (pre, half, mem)
      else (pre, v, mem)
    in
    let x17 = Loc.Reg F.control_scratch in
    let prologue =
      steps (-1L)
      @ List.concat_map
          (fun v ->
            let pre, v, mem = save_word v in
            pre @ [ Mir_phys.Instr.Save { dst = mem; src = Loc.Reg v } ])
          saves
      @
      match control with
      | Some (view, mrs, msr, zero) ->
          let pre, mem = access ~bytes:8L (offset `Control) in
          [
            Mir_phys.Instr.Late
              {
                op = mrs (placeholder 1 Mir_type.i64);
                uses = [ Loc.Reg view ];
                defs = [ x17 ];
              };
          ]
          @ pre
          @ [
              Mir_phys.Instr.Save { dst = mem; src = x17 };
              Mir_phys.Instr.Late { op = zero; uses = []; defs = [ x17 ] };
              Mir_phys.Instr.Late
                {
                  op = msr (placeholder 2 Mir_type.i64);
                  uses = [ x17 ];
                  defs = [ Loc.Reg view ];
                };
            ]
      | None -> []
    in
    let epilogue =
      (match control with
        | Some (view, _, msr, _) when not (mutated Mutation.No_control_restore)
          ->
            let pre, mem = access ~bytes:8L (offset `Control) in
            pre
            @ [
                Mir_phys.Instr.Save { dst = x17; src = mem };
                Mir_phys.Instr.Late
                  {
                    op = msr (placeholder 2 Mir_type.i64);
                    uses = [ x17 ];
                    defs = [ Loc.Reg view ];
                  };
              ]
        | _ -> [])
      @ List.concat_map
          (fun v ->
            let pre, v, mem = save_word v in
            pre @ [ Mir_phys.Instr.Save { dst = Loc.Reg v; src = mem } ])
          saves
      @ steps 1L
    in
    let returns = ref 0 in
    let blocks =
      List.map
        (fun (b : (T.op, T.test) Mir_phys.Block.t) ->
          let body = List.concat_map rewrite b.Mir_phys.Block.body in
          let body =
            if Mir_id.Block.equal b.Mir_phys.Block.id f.Mir_phys.Func.entry then
              prologue @ body
            else body
          in
          let body =
            match b.Mir_phys.Block.terminator with
            | Mir_phys.Term.Return _ ->
                incr returns;
                if mutated Mutation.Epilogue_once && !returns > 1 then body
                else body @ epilogue
            | Mir_phys.Term.Branch _ | Mir_phys.Term.Jump _ -> body
          in
          let entry =
            List.map
              (fun (v, l) -> (v, snd (rewrite_loc l)))
              b.Mir_phys.Block.entry
          in
          { b with Mir_phys.Block.body; entry })
        f.Mir_phys.Func.blocks
    in
    if too_large then Error (Refusal.Frame_too_large f.Mir_phys.Func.id)
    else Ok { f with Mir_phys.Func.blocks; slots = []; frame = Some size }

  let realize ?mutation ?(pad = 0L) (p : (T.op, T.test) Mir_phys.Program.t) =
    let funcs =
      List.map
        (fun (f : (_, _) Mir_phys.Func.t) ->
          realize_func ?mutation ~pad
            ~main:(Mir_id.Func.equal f.Mir_phys.Func.id p.Mir_phys.Program.main)
            f)
        p.Mir_phys.Program.funcs
    in
    match List.find_map (function Error e -> Some e | Ok _ -> None) funcs with
    | Some e -> Error e
    | None ->
        Ok { p with Mir_phys.Program.funcs = List.map Result.get_ok funcs }
end
