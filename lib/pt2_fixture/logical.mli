(** A tensor as it is logically: dtype, shape and the elements in row-major
    order, little-endian. This is the one form in which a reference tensor (a
    [.pt] file with whatever strides and offset it was saved with) and an actual
    result (a Native tensor, whose internal layout is channels-last) are
    compared and hashed, so a layout can never be mistaken for a value. *)

module Dtype = Pt2_checkpoint_map.Dtype

type t = { data : Pt2_storage.t; dtype : Dtype.t; shape : int64 list }
(** [data] has exactly [numel * byte_width] bytes. *)

type error =
  [ `Logical_numel_overflow
  | `Logical_over_limit of int64
  | `Logical_stride_range of int
  | `Logical_unsupported_dtype of Pt2_dtype.t ]

val pp_error : error Fmt.t
val numel : t -> int64
val byte_count : t -> int64

val of_pt2 : ?max_bytes:int -> Pt2_tensor.t -> (t, [> error ]) Err.t
(** Gather a strided tensor into row-major bytes. Every element index is bounded
    before the gather, so a stride or offset that would read outside the storage
    is an error, not a wrapped read. [max_bytes] defaults to 1 GiB. *)

val of_bytes :
  dtype:Dtype.t -> shape:int64 list -> string -> (t, [> error ]) Err.t
(** Already row-major bytes. *)

val get_float : t -> int -> float
(** Element [i] of an [F32] or [F64] tensor as a double (exact). Raises
    [Invalid_argument] for another dtype or an index out of range. *)

val get_int64 : t -> int -> int64
(** Element [i] of an integer or [BOOL] tensor, sign- or zero-extended as its
    dtype says. *)
