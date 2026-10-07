(* What a target's semantic function sees while one selected instruction
   executes: its virtual operands, the synthetic memory, view addresses and a
   way to stop on a defect. The target never sees the interpreter's frames or
   control flow. *)

open Machine_ir

type t = {
  get : Mir_value.t -> Mir_datum.t;
  memory : Mir_memory.t;
  view : Mir_id.View.t -> Mir_memory.Pointer.t option;
  defect : 'a. Mir_observation.Defect.t -> 'a;
  call : Mir_op.Callee.t -> Mir_datum.t list -> Mir_datum.t list;
      (** a call under the target's convention: the callee's results, then its
          [i32] status (0 on success; on failure the record is already stored
          and no result is defined) *)
}

let bits env v =
  match env.get v with
  | Mir_datum.Bits b -> b
  | Mir_datum.Flags _ | Mir_datum.Lanes _ | Mir_datum.Order | Mir_datum.Ptr _ ->
      env.defect Mir_observation.Defect.Invalid_program

let ptr env v =
  match env.get v with
  | Mir_datum.Ptr p -> p
  | Mir_datum.Bits _ | Mir_datum.Flags _ | Mir_datum.Lanes _ | Mir_datum.Order
    ->
      env.defect Mir_observation.Defect.Invalid_program

(* The [mask] bits of a condition value; reading a bit its producer left
   undefined is a defect. *)
let flags env v ~mask =
  match env.get v with
  | Mir_datum.Flags { bits; defined } ->
      if Int64.equal (Int64.logand mask (Int64.lognot defined)) 0L then
        Int64.logand bits mask
      else env.defect Mir_observation.Defect.Uninitialized
  | Mir_datum.Bits _ | Mir_datum.Lanes _ | Mir_datum.Order | Mir_datum.Ptr _ ->
      env.defect Mir_observation.Defect.Invalid_program

let fault env = function
  | Mir_memory.Fault.Bad_access -> env.defect Mir_observation.Defect.Bad_access
  | Mir_memory.Fault.Uninitialized ->
      env.defect Mir_observation.Defect.Uninitialized

let load env p ~bytes ~align =
  match Mir_memory.load env.memory p ~bytes ~align with
  | Ok x -> x
  | Error e -> fault env e

let store env p ~bytes ~align x =
  match Mir_memory.store env.memory p ~bytes ~align x with
  | Ok () -> ()
  | Error e -> fault env e
