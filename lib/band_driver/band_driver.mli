(** The driver of a band schedule as an SSA program. A stage is a contraction
    over the pixels of a row ([cin] channels in, [cout] out) of a square window
    of side [extent] (1 is pointwise) with the kernel's stride, dilation and
    zero padding on both axes: its output has the rows and columns
    {!Band_schedule.output_height} gives, its weights
    [(extent * extent * cin) x cout] with the window's row [t] at row offset
    [t * extent * cin]. A window position in the padding is skipped, not added
    as a zero: the rows outside the producer's extent are not read and
    contribute nothing, so the call making an output pixel's first product seeds
    it with zero and the rest add into it. The schedule says which rows each
    stage makes when, and the driver issues {!Ssa_ir.Ssa_view_call}s, their
    operands' bases the element offsets of the rows the event reads and writes:
    an intermediate value is a rolling window of rows ({!Band_schedule.Window}),
    the first stage's input and the last stage's output are whole tensors. A
    window is one call per output row and row it reads (one per column too when
    dilation or padding keeps the columns from being contiguous, over the output
    pixels that column reaches), each past a pixel's first adding into the
    output (seed [Tensor]), so the sum runs row by row, then column and channel,
    as in {!whole}. *)

open Ssa_ir

module Stage : sig
  type t = { kernel : Band_schedule.kernel; cin : int64; cout : int64 }
end

(** Why a chain has no driver. *)
type refusal =
  | Empty_window of string
      (** the stage has an output pixel whose window lies wholly in the padding,
          which no call would write *)

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
  | Padding_read_edge
      (** a window row in the padding reads the nearest producer row instead of
          being skipped *)
  | Step_dropped  (** the bases do not move between repetitions *)
  | Stride_dropped  (** a window steps by one pixel and row, not its stride *)
  | Window_offset_dropped
      (** a window reads its first row of the ring for every row it covers *)

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
(** The reference: the same stages over whole tensors, intermediates complete:
    one {!Ssa_op.Contract} each, a window one view call per output row and row
    it reads at logical rows. The buffer ids of [input], [weights] and [output]
    are those of {!program}; [rings] is empty. *)
