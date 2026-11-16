(** A {!Contraction} whose result passes through pointwise operations before it
    is stored (bias, residual add, activation), as one descriptor.

    Each operation reads the running value and, for [Add] and [Mul], one side
    element at the output's own coordinate [(batch, row, col)] through element
    strides (0 broadcasts an axis); every result is rounded to binary32 where it
    is defined. An intermediate the outside needs is a tap, a second output of
    the descriptor. The descriptor holds no buffer: buffers are named and
    supplied to the evaluators. The numerical requirement is the core's own
    [numerics]; the tail adds none.

    Two evaluators exist. {!eval_reference} is the unattached form: the whole
    product, then the operations over its result. {!eval_scratch} is the generic
    realization: the product one block at a time into scratch, then the
    operations over that block. They give the same bits for every legal block;
    an engine that absorbs the tail into its store is a third realization of the
    same descriptor. *)

(** A side operand: the buffer's name and its element strides per batch, row and
    column at the output's coordinates. *)
module Side : sig
  type t = { name : string; batch : int64; row : int64; col : int64 }
end

(** An intermediate stored as a second output. *)
module Tap : sig
  type t = { name : string; strides : Contraction.Output.t }
end

module Op : sig
  (* @alphabetical *)
  type t =
    | Add of Side.t  (** [v + side], rounded to binary32 *)
    | Clamp of { lo : float; hi : float }
        (** [lo] when [v < lo], else [hi] when [v > hi], else [v]; a NaN passes
        *)
    | Mul of Side.t  (** [v * side], rounded to binary32 *)
    | Relu  (** [0] when [v < 0], else [v]; a NaN passes *)
end

module Step : sig
  type t = { op : Op.t; tap : Tap.t option  (** stored after [op] *) }
end

type t = private { core : Contraction.t; steps : Step.t list }

module Invalid : sig
  (* @alphabetical *)
  type t =
    | Clamp_bounds of { lo : float; hi : float }
        (** a NaN bound, or [lo] above [hi] *)
    | Duplicate_tap of string
    | Integer_core of Contraction.Element.t
        (** an integer product's result is a 32-bit integer, not binary32 *)
    | Stride of { name : string; value : int64 }  (** a negative stride *)
end

val make :
  Contraction.t ->
  Step.t list ->
  (t, [> `Invalid_contraction_tail of Invalid.t ]) Err.t
(** The steps run in list order. *)

val pp : t Fmt.t

val pp_error :
  Format.formatter -> [< `Invalid_contraction_tail of Invalid.t ] -> unit

(** A block of the output the generic realization computes at once. *)
module Block : sig
  type t = private { rows : int64; cols : int64 }

  module Invalid : sig
    (* @alphabetical *)
    type t =
      | Extent of { name : string; value : int64 }  (** below 1 *)
      | Too_large of { rows : int64; cols : int64; limit : int64 }
          (** more cells than the limit *)
  end

  val limit : int64
  (** The most cells of scratch a block may name. *)

  val make :
    rows:int64 -> cols:int64 -> (t, [> `Invalid_block of Invalid.t ]) Err.t

  val cells : t -> int64
  (** [rows * cols]: the scratch the realization is charged. *)

  val pp_error : Format.formatter -> [< `Invalid_block of Invalid.t ] -> unit
end

(** Deliberate defects of the generic realization, to show that the comparison
    with the reference can fail. *)
module Fault : sig
  (* @alphabetical *)
  type t =
    | Seed_from_scratch  (** a [Tensor] seed reads the scratch, not C *)
    | Side_block_local
        (** side operands read at the block's local row and column *)
    | Unrounded_intermediate  (** values stay binary64 between operations *)
end

val eval_reference :
  t ->
  a:(int64 -> float) ->
  b:(int64 -> float) ->
  c:(int64 -> float) ->
  side:(string -> int64 -> float) ->
  tap:(string -> int64 -> float -> unit) ->
  write:(int64 -> float -> unit) ->
  unit
(** The product with {!Contraction.eval}, held whole; then, per output in batch,
    row, column order, the steps. [side] reads a named buffer at an element
    offset; [tap] and [write] store at an offset of a named buffer and of C. *)

val eval_scratch :
  ?fault:Fault.t ->
  t ->
  Block.t ->
  a:(int64 -> float) ->
  b:(int64 -> float) ->
  c:(int64 -> float) ->
  side:(string -> int64 -> float) ->
  tap:(string -> int64 -> float -> unit) ->
  write:(int64 -> float -> unit) ->
  unit
(** The generic realization: blocks of at most the given rows and columns in
    batch, row-block, column-block order (edge blocks smaller); each is a
    {!Contraction} over a compact scratch, seeded from C when [Tensor], then the
    steps over the scratch, storing to C's offsets and the taps. *)
