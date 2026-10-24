(** [C[b, i, j] = seed + sum over k of A[b, i, k] * B[b, k, j]] for every batch
    [b < batch], row [i < m], column [j < n], [k] ascending from 0 to [k - 1].
    Each operand is addressed by element strides from its own base, so a
    transposed or batched operand is a choice of strides; every extent and
    stride is an [int64]. The descriptor holds no buffer: the operation that
    carries it names A, B and C, and C is disjoint from both. *)

(** The element format of A and B, and of C. *)
module Element : sig
  type t = F32  (** binary32, accumulated in binary32 *)

  val name : t -> string
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
    (binary32 values), [c] reads the seed when it is [Tensor], [write] stores
    the result. *)
