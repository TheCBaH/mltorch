(* What an allocation costs, counted from its physical program: the moves
   between registers, the stores to and loads from frame slots (spills and
   reloads, edge transfers included), slot-to-slot copies and the slots. *)

open Machine_ir
module Loc = Mir_phys.Loc

type t = {
  register_moves : int;
  stores : int;
  loads : int;
  slot_moves : int;  (** slot to slot, through the copy scratch *)
  slots : int;
  slot_bytes : int64;
  executed : int;  (** selected instructions *)
  rematerialized : int;  (** re-runs instead of reloads *)
}

let empty =
  {
    register_moves = 0;
    stores = 0;
    loads = 0;
    slot_moves = 0;
    slots = 0;
    slot_bytes = 0L;
    executed = 0;
    rematerialized = 0;
  }

let is_memory = function Loc.Slot _ | Loc.Mem _ -> true | Loc.Reg _ -> false

let of_program (p : (_, _) Mir_phys.Program.t) =
  List.fold_left
    (fun acc (f : (_, _) Mir_phys.Func.t) ->
      let acc =
        {
          acc with
          slots = acc.slots + List.length f.Mir_phys.Func.slots;
          slot_bytes =
            List.fold_left
              (fun n (s : Mir_phys.Slot.t) -> Int64.add n s.Mir_phys.Slot.bytes)
              acc.slot_bytes f.Mir_phys.Func.slots;
        }
      in
      List.fold_left
        (fun acc (b : (_, _) Mir_phys.Block.t) ->
          List.fold_left
            (fun acc -> function
              | Mir_phys.Instr.Move { dst; src; _ } -> (
                  match (is_memory dst, is_memory src) with
                  | false, false ->
                      { acc with register_moves = acc.register_moves + 1 }
                  | true, false -> { acc with stores = acc.stores + 1 }
                  | false, true -> { acc with loads = acc.loads + 1 }
                  | true, true -> { acc with slot_moves = acc.slot_moves + 1 })
              | Mir_phys.Instr.Exec _ ->
                  { acc with executed = acc.executed + 1 }
              | Mir_phys.Instr.Remat _ ->
                  { acc with rematerialized = acc.rematerialized + 1 }
              | Mir_phys.Instr.Late _ | Mir_phys.Instr.Save _
              | Mir_phys.Instr.Sp _ ->
                  acc)
            acc b.Mir_phys.Block.body)
        acc f.Mir_phys.Func.blocks)
    empty p.Mir_phys.Program.funcs

let pp fmt t =
  Fmt.pf fmt
    "%d instructions; %d register moves, %d stores, %d loads, %d slot moves, \
     %d rematerialized; %d slots (%Ld bytes)"
    t.executed t.register_moves t.stores t.loads t.slot_moves t.rematerialized
    t.slots t.slot_bytes
