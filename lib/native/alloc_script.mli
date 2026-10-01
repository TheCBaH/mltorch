(* The allocation script of a direct run: what a run allocates for each node's
   outputs and when it releases them, as data. A dry run of the evaluator emits
   it without computing anything, and the arena plan is built from it (see
   .ai/ on the tensor arena).

   Events follow node order, then output order within a node, then
   [Release_schedule.after] order. A [Node] marker opens each node's events, so
   a script also records the order it was produced in.

   What a script records is the evaluator's own allocations: graph inputs and
   constants are bound by the caller and never appear, and neither does an index
   output nothing reads, which the evaluator never allocates. *)

open Graph_common

module Kind : sig
  (** The Bigarray element kind a format is stored in, one constructor per kind:
      [Bool] and the unsigned 8-bit formats share [Int8_unsigned], [F16] and
      [BF16] share [Int16_unsigned]. *)
  type t =
    | Float32
    | Float64
    | Int16_signed
    | Int16_unsigned
    | Int32
    | Int64
    | Int8_signed
    | Int8_unsigned

  val all : t list
  (** Alphabetical. *)

  val cell_bytes : t -> int64
  val of_fmt : Payload.packed_fmt -> t
  val equal : t -> t -> bool
  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit
end

module Alloc : sig
  type t = {
    id : Tensor_id.t;
    signature : Tensor_sig.t;
    kind : Kind.t;
    numel : int64;  (** cells, bounded below [Kernel.Limits.Hard.numel] *)
    bytes : int64;
    eligible : bool;
        (** The arena may back this edge: released after some node, and not
            quantized. Outputs and retained edges are never eligible. *)
  }

  val equal : t -> t -> bool
  (** Every field: identity, signature (shape, format, quantization) and
      eligibility. *)
end

module Event : sig
  type t = Alloc of Alloc.t | Free of Tensor_id.t | Node of Node_id.t

  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

type t = Event.t list

val alloc :
  released:Tensor_id.Set.t ->
  Tensor_sig.t ->
  (Alloc.t, [> `Numel_over_limit of Vec6.Numel_bound.t ]) Err.t
(** The [Alloc] record of one edge; [released] is every edge some node releases
    (from the schedule), which is what makes an edge eligible. *)

module Position : Core.Tagged_int.S
(** Where two scripts first disagree. *)

module Difference : sig
  type t = {
    position : Position.t;
    left : Event.t option;  (** [None]: the left script ended first *)
    right : Event.t option;
  }
end

val first_difference : t -> t -> Difference.t option
(** Compares complete events, in order: [None] iff the scripts are equal. *)

val peak_bytes : t -> (int64, [> `Peak_bytes_overflow of Tensor_id.t ]) Err.t
(** The most bytes the script's own allocations hold at once, measured after a
    node's outputs are allocated and before its releases. Graph inputs are not
    in a script, so [Release_schedule.peak_bytes] is this plus their bytes. *)

val out_of_arena_bytes :
  t -> (int64, [> `Peak_bytes_overflow of Tensor_id.t ]) Err.t
(** The total bytes of the script's ineligible allocations: outputs, retained
    and quantized edges, which stay outside any arena. *)

val pp : Format.formatter -> t -> unit
