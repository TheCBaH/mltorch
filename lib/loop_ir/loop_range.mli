(** Intervals over the concrete loop extents: the proof that lets a backend emit
    plain index arithmetic.

    Every bound is computed in [int64] with SATURATING arithmetic and narrowed
    to [int] by nobody, so an aggregate (a scale times an extent, a chain of
    scales) cannot wrap into a range that looks safe. A check on a wrapped
    result is not a bound; a saturated one is simply "too big". *)

type t = { lo : int64; hi : int64 }
(** Inclusive on both ends. *)

val point : int64 -> t

val span : lo:int -> hi:int -> t
(** The values [lo, lo + 1, ..., hi - 1] a half-open loop takes, or the single
    point [lo] when the loop is empty. *)

val unbounded : t
(** What an index of unknown provenance can be. It is outside the domain. *)

val domain : t
(** The index domain the Loop IR guarantees: [-2^31, 2^31 - 1]. It is the
    [js_of_ocaml] [int] and the admission contract of [Kernel] alike, so emitted
    arithmetic is exact in a [Number] and an OCaml [int] on every backend. *)

val within : inner:t -> outer:t -> bool
val saturating_add : int64 -> int64 -> int64

val saturating_mul : int -> int64 -> int64
(** Exact where the true result is within [-2^62, 2^62], and clamped to those
    ends beyond, so an out-of-domain value is never mistaken for an in-domain
    one by wrapping. *)

module Env : sig
  type range = t
  type t

  val create : unit -> t

  val add_var : Loop_var.t -> range -> t -> t
  (** Scoped: the result holds the variable, the argument does not. *)

  val set_temp : Loop_temp.t -> range -> t -> unit
  (** NOT scoped: an index temporary is assigned once and never reused, so its
      range is recorded in a table every environment derived from this one
      shares. A gather's normalized index is created in the middle of lowering
      an index expression, which has no way to hand a widened environment back
      to its caller. *)
end

val of_index : Env.t -> Loop_index.t -> t
(** The interval an index can take. A variable or temporary the environment does
    not know is [unbounded]. *)

val proven : Env.t -> Loop_index.t -> bool
(** Every checked operation ([Add] and [Scale]) of the index stays inside
    [domain] for every value its variables can take, so no overflow check is
    needed. False means "not proven", not "overflows". *)
