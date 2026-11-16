(** The driver of a band schedule as an SSA program. A stage is a pointwise
    contraction over the pixels of a row ([cin] channels in, [cout] out, weights
    [cin x cout]); the schedule says which rows each stage makes when, and the
    driver issues one {!Ssa_ir.Ssa_view_call} per event, its operands' bases the
    element offsets of the rows the event reads and writes: an intermediate
    value is a rolling window of rows ({!Band_schedule.Window}), the first
    stage's input and the last stage's output are whole tensors. *)

open Ssa_ir

module Stage : sig
  type t = { kernel : Band_schedule.kernel; cin : int64; cout : int64 }
end

(** Why a chain has no driver. *)
type refusal =
  | Not_pointwise of string
      (** the stage's kernel is not extent 1, stride 1, no padding, dilation 1
      *)

val pp_refusal : Format.formatter -> refusal -> unit

(** The repeated part of a banded program: from block [first], [iterations]
    repetitions of [period] blocks, one body of view calls whose bases move by a
    fixed step per repetition. Blocks before and after it are unrolled. *)
module Loop : sig
  type t = {
    first : Band_schedule.Count.t;
    period : Band_schedule.Count.t;
    iterations : Band_schedule.Count.t;
  }
end

module Program : sig
  type t = {
    program : Ssa_program.t;
    input : Ssa_id.Buffer.t;  (** the first stage's whole input, rows x width *)
    weights : Ssa_id.Buffer.t list;  (** per stage *)
    output : Ssa_id.Buffer.t;  (** the last stage's whole output *)
    rings : (Ssa_id.Buffer.t * Band_schedule.Count.t) list;
        (** per non-last stage, its window buffer and its rows *)
    loop : Loop.t option;  (** [None] when no run of blocks repeats *)
  }
end

(** Deliberate defects of the loop for the evidence suite; no consumer applies
    one. Alphabetical. *)
type mutation =
  | Iterations_short  (** the loop stops one repetition early *)
  | Step_dropped  (** the bases do not move between repetitions *)

val program :
  ?looped:bool ->
  ?mutation:mutation ->
  Stage.t list ->
  width:int64 ->
  rows:Band_schedule.Count.t ->
  block:Band_schedule.Span.block Band_schedule.Span.t ->
  (Program.t, refusal) result
(** A run of blocks that repeats becomes a loop over one body, so a view-call
    body is reused per repetition; [~looped:false] keeps every event unrolled.
    Raises [Invalid_argument] as {!Band_schedule.run} does. *)

val whole :
  Stage.t list -> width:int64 -> rows:Band_schedule.Count.t -> Program.t
(** The reference: the same stages as one {!Ssa_op.Contract} each over whole
    tensors, intermediates complete. The buffer ids of [input], [weights] and
    [output] are those of {!program}; [rings] is empty. *)
