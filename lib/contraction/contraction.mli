(** [C[b, i, j] = seed + sum over k of A[b, i, k] * B[b, k, j]] for every batch
    [b < batch], row [i < m], column [j < n], [k] ascending from 0 to [k - 1].
    Each operand is addressed by element strides from its own base, so a
    transposed or batched operand is a choice of strides; every extent and
    stride is an [int64]. The descriptor holds no buffer: the operation that
    carries it names A, B and C, and C is disjoint from both. *)

(** An integer element's signedness. *)
module Sign : sig
  type t = Signed | Unsigned
end

(** The element format of A and B. A floating format's C is binary32 and every
    sum is accumulated in binary32 under the numerical requirement; an integer
    format's C is a 32-bit signed integer and every sum is exact (so any order
    computes it, and the requirement does not apply). *)
module Element : sig
  type t =
    | Bf16  (** bfloat16, each element widened exactly to binary32 *)
    | F32  (** binary32 *)
    | I8 of { a : Sign.t; b : Sign.t }
        (** 8-bit integers, each operand signed or not *)

  val name : t -> string
  (** [bf16], [f32], or [i8] and a letter an operand ([s], [u]). *)

  val bytes : t -> int64
  (** One element's bytes: 1, 2 or 4. *)

  val integer : t -> bool
  val all : t list
end

(** What each output starts from. *)
module Seed : sig
  type t =
    | Tensor  (** C's own value before the operation *)
    | Zero  (** positive zero *)

  val name : t -> string
end

(** How each output's sum is rounded. *)
module Numerics : sig
  type t =
    | Fused_sequential
        (** [acc <- fmaf (a, b, acc)], k ascending from the seed: each step
            rounded once *)
    | Ordered
        (** [acc <- acc + a * b] with the product and the sum each rounded, k
            ascending from the seed *)

  val name : t -> string
end

(** An input's element strides: per batch, per output row or column (A's [i],
    B's [j]), and per [k]. *)
module Operand : sig
  type t = { batch : int64; outer : int64; k : int64 }
end

(** The output's element strides: per batch, row and column. *)
module Output : sig
  type t = { batch : int64; row : int64; col : int64 }
end

type t = private {
  batch : int64;
  m : int64;
  n : int64;
  k : int64;
  a : Operand.t;
  b : Operand.t;
  c : Output.t;
  format : Element.t;
  seed : Seed.t;
  numerics : Numerics.t;
}

(** Why a descriptor is refused. *)
module Invalid : sig
  type t =
    | Extent of { name : string; value : int64 }
        (** a batch, row or column extent below 1, or [k] below 0 *)
    | Overflow of { k : int64 }
        (** an integer [k] at which the largest products' sum passes 2^31 - 1 *)
    | Stride of { name : string; value : int64 }  (** a negative stride *)
end

val make :
  batch:int64 ->
  m:int64 ->
  n:int64 ->
  k:int64 ->
  a:Operand.t ->
  b:Operand.t ->
  c:Output.t ->
  format:Element.t ->
  seed:Seed.t ->
  numerics:Numerics.t ->
  (t, [> `Invalid_contraction of Invalid.t ]) Err.t

val pp : t Fmt.t

val pp_error :
  Format.formatter -> [< `Invalid_contraction of Invalid.t ] -> unit

val name : t -> string
(** A canonical text naming every field, from which {!of_name} rebuilds the
    descriptor:
    [contract.f32.b<batch>.m<m>.n<n>.k<k>.a<batch>_<outer>_<k>.
     b<...>.c<batch>_<row>_<col>.<seed>.<numerics>]. *)

val of_name : string -> t option

val symbol : t -> string
(** An assembler symbol unique to the descriptor: [mlt_contract_] and a digest
    of {!name}. *)

val macs : t -> int64
(** [batch * m * n * k]. *)

val eval :
  t ->
  a:(int64 -> float) ->
  b:(int64 -> float) ->
  c:(int64 -> float) ->
  write:(int64 -> float -> unit) ->
  unit
(** The reference: every output in batch, row, column order, each its own sum
    under [numerics]; [a], [b] read an element at an offset from their base
    (binary32 values, a bfloat16 one widened, an integer one exactly), [c] reads
    the seed when it is [Tensor], [write] stores the result: an integer sum is
    exact and the requirement does not apply. *)

(** Which unit realizes a contraction, from the descriptor and what the core
    has. The rules, in order: the matrix engine for an [Ordered] binary32
    product when the core has one ([vtfmm] rounds each product, so it cannot
    compute a fused chain; its bfloat16 form adds two k steps' products to the
    accumulator with one rounding, which neither requirement is); the matrix
    engine for an integer product, whose exact sum any order computes; else the
    vector unit when the core has one; else none, and the product stays in the
    planned loops. *)
module Select : sig
  module Engine : sig
    type t = Loops | Rvv | Zvt

    val name : t -> string
  end

  (** Why the engine was chosen. *)
  module Reason : sig
    type t =
      | Format_off_matrix
          (** the core has the matrix engine, but the operands are bfloat16 *)
      | Fused_off_matrix
          (** the core has the matrix engine, but the product is fused *)
      | Integer_on_matrix  (** an exact integer sum: any order is the same *)
      | No_vector_unit
      | Ordered_on_matrix
      | Vector_unit  (** the core has no matrix engine *)

    val name : t -> string
  end

  module Capabilities : sig
    type t = { matrix : bool; vector : bool }
  end

  val choose : Capabilities.t -> t -> Engine.t * Reason.t
end
