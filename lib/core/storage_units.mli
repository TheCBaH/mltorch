(* Byte and element quantities of tensor storage: sizes, offsets, alignments,
   element counts and element widths, each its own abstract type over [int64],
   so a byte size cannot be passed for an element count, an offset for a size,
   or a size for an alignment.

   Every constructor from a raw [int64] is checked and reports the value as
   written; there is no unchecked one. Arithmetic is checked too and lives here,
   so no caller unwraps, computes and rewraps. [to_int64] is the named exit, for
   serialization, reporting and a backend adapter that bounds the value before
   narrowing it. [int64] throughout: js_of_ocaml's [int] is 32 bits.

   Offsets are relative to a pool and say nothing about which pool, or whether
   they are inside it: a pool bound is checked by whoever owns the pool.

   Modules are in dependency order, not alphabetical: each converts only into
   the ones above it. *)

(** Which quantity an error is about. *)
module Quantity : sig
  type t =
    | Byte_alignment
    | Byte_offset
    | Byte_size
    | Element_bytes
    | Element_count
    | Element_offset

  val pp : Format.formatter -> t -> unit
end

(** A raw value a constructor refused, as written. *)
module Invalid : sig
  module Reason : sig
    type t = Negative | Not_power_of_two | Zero
  end

  type t = { quantity : Quantity.t; value : int64; reason : Reason.t }
end

(** The operation whose result did not fit, or was not exact. *)
module Operation : sig
  type t = Add | Advance | Align_up | Distance | Scale | Sub | To_elements
end

(** An operation and its operands, as raw values. *)
module Operands : sig
  type t = { operation : Operation.t; left : int64; right : int64 }
end

type error =
  [ `Inexact_conversion of Operands.t
  | `Invalid_quantity of Invalid.t
  | `Quantity_overflow of Operands.t
  | `Quantity_underflow of Operands.t ]
(** [`Inexact_conversion]: a byte quantity that is not a whole number of
    elements. [`Quantity_underflow]: a result that would be negative, which no
    quantity here admits. *)

val pp_error : Format.formatter -> [< error ] -> unit

(** Storage bytes per element of a format: positive. *)
module Element_bytes : sig
  type t

  val of_int64 : int64 -> (t, [> `Invalid_quantity of Invalid.t ]) Err.t
  val to_int64 : t -> int64
  val equal : t -> t -> bool
  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit
end

(** A payload size, a pool capacity or a live-byte total: zero or more. *)
module Byte_size : sig
  type t

  val zero : t
  val of_int64 : int64 -> (t, [> `Invalid_quantity of Invalid.t ]) Err.t
  val to_int64 : t -> int64
  val equal : t -> t -> bool
  val compare : t -> t -> int
  val max : t -> t -> t
  val pp : Format.formatter -> t -> unit
  val add : t -> t -> (t, [> `Quantity_overflow of Operands.t ]) Err.t

  val sub : t -> t -> (t, [> `Quantity_underflow of Operands.t ]) Err.t
  (** [sub a b] is [a - b]; [b > a] is an underflow, not a negative size. *)

  val succ : t -> (t, [> `Quantity_overflow of Operands.t ]) Err.t
  val pred : t -> (t, [> `Quantity_underflow of Operands.t ]) Err.t

  val midpoint : t -> t -> t
  (** The size halfway between two sizes, rounded towards the smaller: never
      overflows, whatever the order of its arguments. *)

  (** A size known to be positive. Forgetting the refinement cannot fail. *)
  module Nonzero : sig
    type size := t
    type t

    val of_size : size -> (t, [> `Invalid_quantity of Invalid.t ]) Err.t
    val to_size : t -> size
  end
end

(** A positive power of two, in bytes. *)
module Byte_alignment : sig
  type t

  val of_int64 : int64 -> (t, [> `Invalid_quantity of Invalid.t ]) Err.t
  val to_int64 : t -> int64
  val equal : t -> t -> bool
  val compare : t -> t -> int
  val max : t -> t -> t
  val pp : Format.formatter -> t -> unit

  val to_size : t -> Byte_size.t
  (** For comparing a size against a threshold; it does not round anything. *)

  val of_element_bytes :
    Element_bytes.t -> (t, [> `Invalid_quantity of Invalid.t ]) Err.t
  (** The natural alignment of an element: its width, which must be a power of
      two. *)

  val pad :
    Byte_size.t ->
    t ->
    (Byte_size.t, [> `Quantity_overflow of Operands.t ]) Err.t
  (** [pad size t]: the least multiple of [t] at or above [size], the extent of
      a block whose end, not only its start, stays aligned. Computed as
      {!Byte_offset.align_up}. *)
end

(** A position within byte storage, relative to its pool: zero or more. *)
module Byte_offset : sig
  type t

  val zero : t
  val of_int64 : int64 -> (t, [> `Invalid_quantity of Invalid.t ]) Err.t
  val to_int64 : t -> int64
  val equal : t -> t -> bool
  val compare : t -> t -> int
  val max : t -> t -> t
  val pp : Format.formatter -> t -> unit

  val of_size : Byte_size.t -> t
  (** The offset [size] bytes past the pool's start: where a block of that size
      placed at zero ends. *)

  val to_size : t -> Byte_size.t
  (** The bytes from the pool's start up to [t]: the smallest pool a block
      ending at [t] fits in. *)

  val advance :
    t -> Byte_size.t -> (t, [> `Quantity_overflow of Operands.t ]) Err.t

  val align_up :
    t -> Byte_alignment.t -> (t, [> `Quantity_overflow of Operands.t ]) Err.t
  (** The least aligned offset at or above [t]. The padding is computed from the
      remainder and its addition checked, never as [t + alignment - 1]. *)

  val is_aligned : t -> Byte_alignment.t -> bool

  val distance :
    from:t -> t -> (Byte_size.t, [> `Quantity_underflow of Operands.t ]) Err.t
  (** [distance ~from t] is the size from [from] up to [t]; [t < from] is an
      underflow, not a negative size. *)
end

(** A number of elements: zero or more. *)
module Element_count : sig
  type t

  val zero : t
  val of_int64 : int64 -> (t, [> `Invalid_quantity of Invalid.t ]) Err.t
  val to_int64 : t -> int64
  val equal : t -> t -> bool
  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit

  val to_bytes :
    t ->
    Element_bytes.t ->
    (Byte_size.t, [> `Quantity_overflow of Operands.t ]) Err.t

  val of_bytes :
    Byte_size.t ->
    Element_bytes.t ->
    (t, [> `Inexact_conversion of Operands.t ]) Err.t
  (** Exact: a size that is not a whole number of elements is refused. *)

  (** A count known to be positive. Forgetting the refinement cannot fail. *)
  module Nonzero : sig
    type count := t
    type t

    val of_count : count -> (t, [> `Invalid_quantity of Invalid.t ]) Err.t
    val to_count : t -> count
  end
end

(** A position within typed storage: zero or more. *)
module Element_offset : sig
  type t

  val zero : t
  val of_int64 : int64 -> (t, [> `Invalid_quantity of Invalid.t ]) Err.t
  val to_int64 : t -> int64
  val equal : t -> t -> bool
  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit

  val advance :
    t -> Element_count.t -> (t, [> `Quantity_overflow of Operands.t ]) Err.t

  val distance :
    from:t ->
    t ->
    (Element_count.t, [> `Quantity_underflow of Operands.t ]) Err.t
  (** As {!Byte_offset.distance}, in elements. *)

  val to_bytes :
    t ->
    Element_bytes.t ->
    (Byte_offset.t, [> `Quantity_overflow of Operands.t ]) Err.t

  val of_bytes :
    Byte_offset.t ->
    Element_bytes.t ->
    (t, [> `Inexact_conversion of Operands.t ]) Err.t
  (** Exact, as {!Element_count.of_bytes}. *)
end
