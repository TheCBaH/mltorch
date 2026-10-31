(** A normalization of every row [r < rows] over its [width] elements, in the
    order the generated loops compute it (Native's Region order), with [n] the
    width as a binary32 value, [RN] binary32 round to nearest even and every NaN
    result the canonical [0x7FC00000]:

    {v
    Layer: s    = ((+0 + x0) + x1) + ...           width ascending
           mean = RN(s / n)
           dc   = RN(x_c - mean)
           t    = ((+0 + dc0 * dc0) + dc1 * dc1) + ...
           inv  = RN(1 / RN(sqrt RN(RN(t / n) + eps)))
           y_c  = RN(RN(RN(dc * inv) * w_c) + b_c)
    Rms:   t    = ((+0 + x0 * x0) + x1 * x1) + ...
           inv  = RN(1 / RN(sqrt RN(RN(t / n) + eps)))
           y_c  = RN(RN(x_c * inv) * w_c)
    v}

    An absent weight drops its multiply and an absent bias its add. The
    precision decides the sum of squares alone, the one site a plan that
    contracts dots fuses ({!Attention_block.Precision}): [F32_ordered] rounds
    the product and the sum, [F32_fused] makes each step one [fmaf].
    [Bf16_paired] computes as [F32_ordered] and rounds each output to bfloat16.
    X and Y are addressed by element strides from their own base; the weight and
    the bias are dense vectors of [width]. Every extent and stride is an
    [int64]. *)

(** Alphabetical. *)
module Kind : sig
  type t = Layer | Rms

  val name : t -> string
end

(** An operand's element strides: per row and per column. *)
module Strides : sig
  type t = { row : int64; col : int64 }
end

type t = private {
  kind : Kind.t;
  rows : int64;
  width : int64;
  eps : float;  (** a finite, non-negative binary32 value *)
  weight : bool;
  bias : bool;  (** only for [Layer] *)
  x : Strides.t;
  y : Strides.t;
  precision : Attention_block.Precision.t;
}

(** Why a descriptor is refused. Alphabetical. *)
module Invalid : sig
  type t =
    | Bias_without_layer  (** an RmsNorm has no bias *)
    | Eps of float  (** not a finite, non-negative binary32 value *)
    | Extent of { name : string; value : int64 }
        (** an extent below 1, or a width binary32 does not hold exactly (above
            2^24) *)
    | Stride of { name : string; value : int64 }  (** a negative stride *)
end

val make :
  kind:Kind.t ->
  rows:int64 ->
  width:int64 ->
  eps:float ->
  weight:bool ->
  bias:bool ->
  x:Strides.t ->
  y:Strides.t ->
  precision:Attention_block.Precision.t ->
  (t, [> `Invalid_norm of Invalid.t ]) Err.t

val pp : t Fmt.t
val pp_error : Format.formatter -> [< `Invalid_norm of Invalid.t ] -> unit

val name : t -> string
(** A canonical text naming every field, from which {!of_name} rebuilds the
    descriptor. *)

val of_name : string -> t option

val symbol : t -> string
(** An assembler symbol unique to the descriptor: [mlt_norm_] and a digest of
    {!name}. *)

val eval :
  t ->
  x:(int64 -> float) ->
  weight:(int64 -> float) ->
  bias:(int64 -> float) ->
  write:(int64 -> float -> unit) ->
  unit
(** The reference, rows ascending. [x] reads an element at an offset from X's
    base (a binary32 value), [weight] and [bias] the element of a column (never
    called when absent), [write] stores an output element at an offset from Y's
    base. *)
