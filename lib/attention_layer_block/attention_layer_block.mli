(** One encoder attention layer over [batch] sequences of [tokens] rows of
    [width] features, [heads] heads of [d = width / heads]:

    {v
    Pre:   y = x + gamma * (A(N(x)) W_o^T + b_o)
    Post:  y = N(x + gamma * (A(x) W_o^T + b_o))
    none:  y = x + gamma * (A(x) W_o^T + b_o)
    A(u) = merge_heads (attend (R(u W_q^T + b_q), R(u W_k^T + b_k), u W_v^T + b_v, mask))
    R(z) = z * cos + rotate_half(z) * sin     per head, when the layer has RoPE
    v}

    with [rotate_half(z) = cat(-z[d/2:], z[:d/2])] over the head's features and
    [cos], [sin] shared by every head. Every bias and [gamma] is optional. The
    parts are descriptors of their own ({!Norm_block}, {!Contraction},
    {!Attention_block}), derived by {!make} from the layer's extents so that
    they cannot disagree; head split and merge are strides of the attention,
    never copies.

    Operands are dense: X and Y [batch][tokens][width]; each weight
    [width][width] in Linear's convention (row [j] is output feature [j]); each
    bias, [gamma] and the norm's affine operands [width]; [cos] and [sin] by
    {!Rope.t}'s strides; the mask by the attention's mask strides. Every extent
    and stride is an [int64]. *)

(** Where the normalization stands. Alphabetical. *)
module Placement : sig
  type t = Post | Pre

  val name : t -> string
end

(** The layer's normalization, as {!make} takes it. *)
module Norm_spec : sig
  type t = {
    kind : Norm_block.Kind.t;
    placement : Placement.t;
    eps : float;
    weight : bool;
    bias : bool;
  }
end

(** The derived normalization: its descriptor over the layer's rows and where it
    stands. *)
module Norm : sig
  type t = { block : Norm_block.t; placement : Placement.t }
end

(** The element strides of [cos] and [sin] (the same for both): per sequence (0
    when one table serves every sequence), per token, per feature. *)
module Rope : sig
  type t = { batch : int64; row : int64; col : int64 }
end

(** Which optional operands the layer reads. *)
module Flags : sig
  type t = {
    q_bias : bool;
    k_bias : bool;
    v_bias : bool;
    o_bias : bool;
    layer_scale : bool;
  }

  val none : t
  val all : t
end

type t = private {
  batch : int64;
  tokens : int64;
  width : int64;
  heads : int64;
  d : int64;
  norm : Norm.t option;
  q : Contraction.t;
  k : Contraction.t;
  v : Contraction.t;
  o : Contraction.t;
  flags : Flags.t;
  rope : Rope.t option;
  attend : Attention_block.t;
  precision : Attention_block.Precision.t;
}

(** Why a descriptor is refused. Alphabetical. *)
module Invalid : sig
  type t =
    | Attention of Attention_block.Invalid.t
    | Contraction of Contraction.Invalid.t
    | Extent of { name : string; value : int64 }  (** an extent below 1 *)
    | Norm of Norm_block.Invalid.t
    | Rope_odd of int64  (** a rotary head dimension that is odd *)
    | Rope_stride of { name : string; value : int64 }  (** a negative stride *)
    | Width of { width : int64; heads : int64 }
        (** a width that is not a multiple of the heads *)
end

val make :
  batch:int64 ->
  tokens:int64 ->
  width:int64 ->
  heads:int64 ->
  ?norm:Norm_spec.t ->
  flags:Flags.t ->
  ?rope:Rope.t ->
  ?mask:Attention_block.Strides.t ->
  scale:float ->
  block_keys:int64 ->
  precision:Attention_block.Precision.t ->
  unit ->
  (t, [> `Invalid_attention_layer of Invalid.t ]) Err.t
(** [scale] is the attention's binary32 scale constant. *)

val pp : t Fmt.t

val pp_error :
  Format.formatter -> [< `Invalid_attention_layer of Invalid.t ] -> unit

val name : t -> string
(** A canonical text naming every field that is not derived, from which
    {!of_name} rebuilds the descriptor. *)

val of_name : string -> t option

val symbol : t -> string
(** [mlt_attention_layer_] and a digest of {!name}. *)

(** The operand readers {!eval} takes: each an element at an offset from its
    operand's base, a binary32 value (a bfloat16 one widened for a bfloat16
    layer's weights). A reader the layer does not use is never called. *)
module Readers : sig
  type t = {
    x : int64 -> float;
    norm_weight : int64 -> float;
    norm_bias : int64 -> float;
    wq : int64 -> float;
    bq : int64 -> float;
    wk : int64 -> float;
    bk : int64 -> float;
    wv : int64 -> float;
    bv : int64 -> float;
    wo : int64 -> float;
    bo : int64 -> float;
    gamma : int64 -> float;
    cos : int64 -> float;
    sin : int64 -> float;
    mask : int64 -> float;
  }
end

val eval : t -> Readers.t -> write:(int64 -> float -> unit) -> unit
(** The reference: the part oracles composed, every step rounded to binary32,
    every NaN the canonical one. In order, for the binary32 precisions:

    {v
    u    = N(x) (Pre) or x                     Norm_block.eval, binary32
    q    = RN(Contract(u, W_q) + b_q)          and k, v alike
    q, k = RN(RN(z * cos) + RN(r * sin))       r = RN(z' * -1) or z', exact
    a    = attend(q, k, v, mask)               Attention_block.eval
    r    = RN(RN(RN(Contract(a, W_o) + b_o) * gamma) + x)
    y    = N(r) (Post) or r
    v}

    An absent bias or [gamma] drops its step. A bfloat16 layer ([Bf16_paired])
    rounds each Paired operand to bfloat16: the norm's output (or X itself
    without a Pre norm), Q, K and V after their bias and rotation, the
    attention's output; the projections are Paired over bfloat16 weights; the
    biases, the rotation, the epilogue and a Post norm stay binary32. *)
