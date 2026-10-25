(** Binary32 [exp], [log], [sin], [cos] and [erf] as sequences of binary32
    operations: add, subtract, multiply, divide, and a select on a comparison.
    Each is specified operation by operation, so an instance that rounds every
    operation to binary32 (and contracts none) computes the same bits whatever
    carries it: the Loop and SSA interpreters, generated C, Wasm, Machine IR.

    The algorithms are the Cephes single-precision ones, with the exponent split
    (for [exp]) and the exponent extraction (for [log]) done by staged
    comparisons against powers of two, which are exact binary32 operations, so
    no bit reinterpretation exists anywhere in them. Accuracy is measured in the
    tests against the correctly rounded binary64 libm result and frozen there.
    [sin] and [cos] reduce by the three-part Cody-Waite constants and are
    accurate for [|x|] up to a few thousand; beyond, they stay defined and
    deterministic but lose accuracy. *)

(** A binary32 arithmetic. [const] receives a constant as written and gives the
    binary32 value nearest to it. *)
module type OPS = sig
  type t

  val const : float -> t
  val add : t -> t -> t
  val sub : t -> t -> t
  val mul : t -> t -> t
  val div : t -> t -> t

  val select_lt : t -> t -> t -> t -> t
  (** [select_lt a b c d] is [c] when [a < b], else [d]: false for a NaN. *)

  val select_eq : t -> t -> t -> t -> t
  (** [select_eq a b c d] is [c] when [a = b], else [d]: false for a NaN. *)

  val share : t -> t
  (** The value of an operation used more than once: an instance that builds
      text or statements binds it to a name here, one that computes values
      returns it. Called on every intermediate the sequences reuse. *)
end

module Make (O : OPS) : sig
  val cos : O.t -> O.t
  val erf : O.t -> O.t
  val exp : O.t -> O.t
  val log : O.t -> O.t
  val sin : O.t -> O.t
end

val round32 : float -> float
(** The binary32 value nearest [x], as a binary64. *)

val fma32 : float -> float -> float -> float
(** [fmaf a b c] for binary32 [a], [b], [c]: [a * b + c] rounded once to the
    nearest binary32 (ties to even), as a binary64. *)

val add_odd32 : float -> float -> float
(** [add_odd32 a b] is [a + b] rounded to odd in binary32, as a binary64: the
    exact sum when binary32 holds it, else the neighbour toward zero or away
    from it whose last bit is odd. The matrix engine's bfloat16 pair: [a], [b]
    are products of two bfloat16 values each, exact in binary64. *)

(** The instance over binary64 numbers that each operation rounds to binary32:
    the oracle's meaning of the five functions. Arguments need not be binary32
    values; they are rounded first. *)
module Value : sig
  val cos : float -> float
  val erf : float -> float
  val exp : float -> float
  val log : float -> float
  val sin : float -> float
end
