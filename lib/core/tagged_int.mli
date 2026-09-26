(** Domain-typed integers whose domains are only compared, keyed and printed:
    ids, ordinals, arena indices, algorithm-local numbers. Each application of
    {!Make} is a fresh type that no other application's [int] can be passed for.
    See .ai/ (domain-typed integers).

    [type t = private int] makes the exit a free coercion, [(x :> int)], so a
    hot path pays nothing to leave the domain. Entry goes through {!S.of_int},
    which is the owner's builder rather than a validation point: a domain with
    an invariant wraps it in a checked constructor of its own. Keep it off
    per-element paths, as [Dim] does.

    js_of_ocaml's [int] is 32 bits; nothing here widens it. *)

module type S = sig
  type t = private int

  val of_int : int -> t
  val to_int : t -> int
  val equal : t -> t -> bool
  val compare : t -> t -> int

  val succ : t -> t
  (** Not bounded: an owner that allocates past a ceiling checks it first. *)

  val pp : Format.formatter -> t -> unit
  (** [prefix] followed by the number, e.g. [t12]. *)

  module Map : Map.S with type key = t
  module Set : Set.S with type elt = t

  (** The first-free counter of an id space. An id is typed and the counter that
      hands out the next one is not, unless it is given a type too: a tensor
      watermark can otherwise be passed where a node watermark is wanted. *)
  module Next : sig
    type id = t
    type t = private int

    val first : t
    (** Nothing allocated yet. *)

    val of_int : int -> t
    (** The counter as a number, for a test or a stored watermark. *)

    val after : id -> t -> t
    (** [after id n] is [n], raised past [id] if [id] is at or above it. *)

    val alloc : t -> id * t
    (** The next free id, and the counter past it. *)

    val check_room : t -> count:int -> unit
    (** Raises [Invalid_argument] if allocating [count] consecutive ids would
        leave the id space. [count > max_int - next], NOT
        [next + count > max_int]: js_of_ocaml's [int] is 32 bits, so a wrapped
        sum sails straight past the naive comparison.

        This is an INVARIANT, not a reachable resource ceiling, which is why it
        raises rather than returning a row: an id space is exhausted only after
        about 2^31 live entries, and memory goes first by orders of magnitude on
        every backend. It is a builder that cannot keep its own promise, where
        an over-limit graph is one we decline to build. *)

    val alloc_n : t -> int -> id list * t
    (** [count] consecutive ids in ascending order, after [check_room]. *)

    val reaches : t -> id -> bool
    (** [id] is at or above the counter: it was introduced after that point. *)

    val equal : t -> t -> bool
    val compare : t -> t -> int
    val pp : Format.formatter -> t -> unit
  end
end

(** Generative: two applications with the same [prefix] still give distinct
    types. *)
module Make
    (_ : sig
      val prefix : string
    end)
    () : S
