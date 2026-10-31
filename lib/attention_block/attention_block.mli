(** [softmax (Q K^T * scale + mask) V] for every batch [b < batch], head
    [h < heads] and query [i < tq], over [tk] keys of [d] features, computed as
    the streaming kernel does: the keys in blocks of [block_keys], each block's
    scores and context products rounded as the matrix engine's bfloat16 tile
    multiply rounds them ({!Contraction.Numerics.Paired}), the exponentials
    rounded to bfloat16 before the context product, the softmax normalizing at
    the end. Q, K and V are bfloat16, the mask and the output binary32. Each
    operand is addressed by element strides from its own base (0 on a broadcast
    axis); every extent and stride is an [int64]. The descriptor holds no
    buffer: the operation that carries it names the operands, and the output is
    disjoint from all of them.

    This is a granted numerics, never a silent substitute for the binary32
    [Sdpa]: the block order, the bfloat16 rounding of the unnormalized
    probabilities and the blocks' padding are part of the contract {!eval}
    states. *)

(** An operand's element strides: per batch, per head, per row (a query, or a
    key) and per column (a feature, or for the mask a key). *)
module Strides : sig
  type t = { batch : int64; head : int64; row : int64; col : int64 }
end

(** How the two matrix products and the probabilities are rounded. The bfloat16
    form is the matrix engine's grant; the binary32 forms read binary32 Q, K and
    V, keep the unnormalized probabilities in binary32 and run each context
    product over the block's valid keys only, with {!Contraction}'s
    [Fused_sequential] or [Ordered] numerics. Alphabetical. *)
module Precision : sig
  type t = Bf16_paired | F32_fused | F32_ordered

  val all : t list
  val name : t -> string
  val format : t -> Contraction.Element.t
  val numerics : t -> Contraction.Numerics.t
end

type t = private {
  batch : int64;
  heads : int64;
  tq : int64;
  tk : int64;
  d : int64;
  q : Strides.t;
  k : Strides.t;
  v : Strides.t;
  o : Strides.t;
  mask : Strides.t option;
      (** the mask's rows are queries and its columns keys *)
  scale : float;  (** a binary32 value *)
  block_keys : int64;
  precision : Precision.t;
}

(** Why a descriptor is refused. *)
module Invalid : sig
  type t =
    | Block_keys of int64  (** not a positive multiple of four *)
    | Extent of { name : string; value : int64 }  (** an extent below 1 *)
    | Scale of float  (** not a finite binary32 value *)
    | Stride of { name : string; value : int64 }  (** a negative stride *)
end

val make :
  batch:int64 ->
  heads:int64 ->
  tq:int64 ->
  tk:int64 ->
  d:int64 ->
  q:Strides.t ->
  k:Strides.t ->
  v:Strides.t ->
  o:Strides.t ->
  ?mask:Strides.t ->
  scale:float ->
  block_keys:int64 ->
  ?precision:Precision.t ->
  unit ->
  (t, [> `Invalid_attention of Invalid.t ]) Err.t

val pp : t Fmt.t
val pp_error : Format.formatter -> [< `Invalid_attention of Invalid.t ] -> unit

(** The scale constant: one binary32 value. *)
module Scale : sig
  val default : d:int64 -> float
  (** [1 / sqrt d] computed in binary64 and rounded to binary32. *)

  val explicit : float -> float
  (** The binary32 value nearest the given scale. *)
end

val name : t -> string
(** A canonical text naming every field, from which {!of_name} rebuilds the
    descriptor. *)

val of_name : string -> t option

val symbol : t -> string
(** An assembler symbol unique to the descriptor: [mlt_attend_] and a digest of
    {!name}. *)

val eval :
  t ->
  q:(int64 -> float) ->
  k:(int64 -> float) ->
  v:(int64 -> float) ->
  mask:(int64 -> float) ->
  write:(int64 -> float -> unit) ->
  unit
(** The reference, in the order the kernel runs it. The readers take an element
    offset from their operand's base and give a binary32 value (a bfloat16 one
    widened); [mask] is never called when the descriptor has none; [write]
    stores an output element. Per query column, with [RN] binary32 round to
    nearest even, every operation unfused, every NaN result the canonical
    [0x7FC00000], over key blocks [j = 0, 1, ...] of [block_keys] keys (the last
    holds [vk <= block_keys] valid keys):

    {v
    S_j  = Paired(K_j Q^T)                      seed +0, over d
    z_j  = RN(RN(S_j * scale) + mask_j)         (no mask: RN(S_j * scale))
    m_j  = max(m_{j-1}, colmax z_j)             RISC-V fmax; m_{-1} = -inf
    a_j  = exp(m_{j-1} - m_j)                   1 where m_j = -inf
    p_j  = exp(z_j - m_j)                       +0 where z_j = -inf
    s_j  = ((+0 + p_j[0]) + p_j[1]) + ...       valid keys, ascending
    l_j  = RN(RN(l_{j-1} * a_j) + s_j)          l_{-1} = +0
    Pb_j = bfloat16 RNE of p_j                  +0 past the valid keys
    O_j  = Paired(Pb_j V_j ; seed RN(a_j * O_{j-1}))   over all block_keys keys
    O    = O_last / l_last                      +0 where l = 0
    v}

    [exp] is {!F32_math.Value.exp}. The context product runs over every key of
    the block, the padded ones included, reading [+0] for both operands past the
    valid keys; the padded pairs are part of the contract because they turn a
    [-0] accumulator into [+0]. A column holding a NaN score outputs NaN in
    every feature. *)

val bf16_round : float -> float
(** The bfloat16 nearest a binary32 value (ties to even on the high half of its
    bit pattern), widened. A NaN stays NaN only when its low half is clear, as
    every arithmetic result's is; a payload NaN near all-ones carries into the
    sign. *)
