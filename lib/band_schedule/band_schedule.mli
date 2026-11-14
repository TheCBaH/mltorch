(** Banding of a spine of row-local kernels (design: row ranges propagated
    backward, retention by last use).

    A kernel's output rows [[lo, hi)] depend on a bounded range of its input
    rows, so a chain of kernels can run block by block: the last kernel
    produces [block] rows, each earlier kernel runs ahead in chunks of its own,
    and a produced row stays in its window until the consumer's last use of
    it. This module is the interpreter of that schedule: it computes which rows
    each step produces and consumes and how large each producer's window must
    be. It runs nothing. *)

(** A row position in a tensor, possibly negative before clamping (padding). *)
module Row : sig
  type t = private int

  val of_int : int -> t
  (** The domain's own constructor. *)

  val to_int : t -> int
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

(** A number of rows, never negative. *)
module Count : sig
  type t = private int

  val of_int : int -> t
  (** Raises [Invalid_argument] on a negative count. *)

  val to_int : t -> int
  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit
end

type kernel = {
  name : string;
  extent : Count.t;  (** the window's size [K], at least 1 *)
  stride : Count.t;  (** at least 1 *)
  pad : Count.t;
  dilation : Count.t;  (** at least 1 *)
  chunk : Count.t;  (** rows it produces a call when it runs ahead; at least 1 *)
}

val window : kernel -> Count.t
(** [(K - 1) * d + 1], the rows one output row reads. *)

val output_height : kernel -> input:Count.t -> Count.t
(** The output rows of a kernel over [input] rows:
    [(input + 2 pad - window) / stride + 1], zero when the window does not fit.
*)

val input_rows :
  kernel -> lo:Row.t -> hi:Row.t -> height:Count.t -> Row.t * Row.t
(** The input rows, clamped to [[0, height)], that output rows [[lo, hi)] need:
    [[lo * s - p, (hi - 1) * s - p + window)]. *)

type event = {
  kernel : string;  (** the kernel, by name; names are unique in a chain *)
  produced : Row.t * Row.t;  (** output rows, half open *)
  consumed : Row.t * Row.t;  (** input rows it reads, clamped *)
}

type t = {
  events : event list;  (** in execution order *)
  windows : Count.t list;
      (** per kernel, the rows its output keeps live for its consumer, rounded
          up to a multiple of its chunk; zero for the last kernel, whose rows
          are written out *)
}

val run : kernel list -> input:Count.t -> block:Count.t -> t
(** Runs the chain over an input of [input] rows, the last kernel producing
    [block] rows at a time. Raises [Invalid_argument] on an empty chain, a zero
    block or a kernel that produces no rows. *)

val pp : Format.formatter -> t -> unit

(** Why a segment is not a band segment: one spine, a chain whose kernels feed
    the next through a banded value, with [Full] side inputs and side outputs
    only. Alphabetical. *)
type refusal =
  | Fan_out of string  (** the kernel's output feeds two consumers *)
  | Rows_not_demanded of { kernel : string; row : Row.t }
      (** a side output of the kernel whose row [row] the spine never demands,
          so the write-through would leave it unwritten (a stride-2 1x1 consumer
          skips every other producer row) *)
  | Second_sink of string  (** a kernel other than the last has no consumer *)
  | Unknown_kernel of string

val pp_refusal : Format.formatter -> refusal -> unit

val admit :
  kernel list ->
  consumers:(string * string list) list ->
  side_outputs:string list ->
  input:Count.t ->
  block:Count.t ->
  (t, refusal) result
(** The schedule of the spine, or the reason it is not one. [consumers] lists,
    per kernel, the kernels that read its output inside the segment; each kernel
    but the last must have exactly one, the next of the chain. [side_outputs]
    names kernels whose output is also written out in full: the spine's demand
    must cover every row of each. *)

(** What a schedule gets wrong. Alphabetical. *)
type violation =
  | Consumed_not_produced of { event : Count.t; kernel : string; row : Row.t }
      (** an event reads a row its producer has not made yet *)
  | Consumed_range_wrong of { event : Count.t; kernel : string }
      (** an event's input rows are not those its output rows need *)
  | Row_gap of { kernel : string; row : Row.t }
      (** a kernel skipped to a row past the next unproduced one *)
  | Row_recomputed of { kernel : string; row : Row.t }
      (** a kernel produced a row it had already produced *)
  | Rows_missing of { kernel : string; row : Row.t }
      (** a kernel's output ends before its last row *)
  | Unknown_event_kernel of string

val pp_violation : Format.formatter -> violation -> unit

val verify : kernel list -> input:Count.t -> t -> (unit, violation) result
(** Checks a schedule against the rules it is built by: every kernel produces
    its rows once, in order, with none missing; every event reads exactly the
    input rows its output rows need; and an event reads only rows its producer
    has already made. *)

(** Deliberate defects for the evidence suite: each turns a valid schedule into
    one {!verify} must refuse. No consumer applies one. *)
type mutation =
  | Drop_event  (** the first event of the last kernel is lost *)
  | Repeat_event  (** the first event of the first kernel runs twice *)
  | Run_consumer_first  (** the last kernel's first event moves to the front *)
  | Shift_consumed  (** the last kernel's first read starts one row late *)

val mutate : mutation -> t -> t
