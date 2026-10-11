(** Checked extent arithmetic for logical shapes. Products divide into the
    ceiling before multiplying, so a bound is tested on the exact value and
    never on a wrapped one. *)

val element_count : int64 list -> (int64, Codec_error.t) Err.t
(** The product of the extents; every extent must be nonnegative and the product
    must fit an [int64]. A zero extent gives zero. *)

val byte_count : width:int -> int64 list -> (int, Codec_error.t) Err.t
(** Element count times [width], bounded by [Sys.max_string_length]; the only
    point where a payload size is narrowed to [int]. *)

val to_ints : int64 list -> int list
(** The extents of a shape whose byte count has been bounded, so each fits. *)

val strides : int list -> int list
(** Row-major element strides. *)
