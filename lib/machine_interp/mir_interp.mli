(** The generic interpreter. It owns its opcode dispatch, state transitions,
    byte addressing, guard branches and edge transfers; it never runs an
    instruction through the SSA interpreter or a generated kernel. Loops run
    through an iterative block dispatcher; calls nest to a bounded depth, and a
    step budget (test fuel) bounds the whole run independently of any language
    meter. Execution stops at the first failure. *)

open Machine_ir

(** Where execution stopped. *)
module Location : sig
  type t = {
    func : Mir_id.Func.t;
    block : Mir_id.Block.t;
    instr : Mir_id.Instr.t option;  (** [None] at the terminator *)
  }

  val pp : Format.formatter -> t -> unit
end

module Outcome : sig
  type t =
    | Defect of Mir_observation.Defect.t * Location.t
    | Failure of Mir_observation.Row.t
    | Fuel_exhausted
    | Success of Mir_datum.t list
    | Unsupported of string

  val pp : Format.formatter -> t -> unit
end

(** The region instances one run addresses. *)
module Binding : sig
  type t

  val instance : t -> Mir_id.Region.t -> Mir_memory.Key.t option
  (** The memory instance a region was given. *)

  val view :
    t -> Mir_program.generic -> Mir_id.View.t -> Mir_memory.Pointer.t option
  (** A pointer to a view's first byte, with its window and permission. *)
end

val instantiate :
  Mir_program.generic ->
  Mir_memory.t ->
  bound:(Mir_id.Region.t -> string option) ->
  (Binding.t, string) result
(** One instance per region: a [Bound] region holds the bytes [bound] supplies
    (or none defined), a [Constant] one its bytes, an [Uninitialized] one
    nothing. [Error] names a region the synthetic address space cannot place or
    whose bound bytes are too many. *)

type run = {
  outcome : Outcome.t;
  events : (Mir_event.t * int64) list;  (** every event, zero counts kept *)
  steps : int64;  (** instructions and terminators executed *)
  last : Location.t option;
}

val run :
  ?fuel:int64 ->
  ?max_depth:int ->
  ?invocation:int32 ->
  ?models:Mir_helper_model.t list ->
  Mir_verify.Generic.t ->
  Mir_memory.t ->
  Binding.t ->
  args:Mir_datum.t list ->
  run
(** Runs the main function. A helper the program declares with no model is
    [Unsupported] before anything executes. *)
