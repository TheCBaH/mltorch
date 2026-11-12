(* Concrete evaluation: values are floats, indices are ['role Dim.t] (see
   direct.ml), an input is a real tensor. See the `.ai/` native compute design §3.

   The two numeric policies below are part of the evaluator instance, never
   ambient: [Make (P)] is the evaluator under policy [P], and the module
   itself is the instance under {!Policy.default}. Two policies coexist in one
   process; nothing is set and restored. *)

(** How {!dot} accumulates. [Binary64] (the default) sums the exact products in
    binary64, so one rounding to binary32 happens on store.
    [Binary32_sequential] is a sequential chain of binary32 fused multiply-adds,
    each rounded once, which reproduces a reference that accumulates in binary32
    -- and with it that reference's rounding noise. *)
type dot_accumulation = Binary64 | Binary32_sequential

(** How a float is cast to int64. [Checked] (the default) rejects NaN, the
    infinities and out-of-range values, as the engine's design requires.
    [Saturating] is aarch64's conversion: NaN is 0 and an out-of-range value the
    nearest limit. C++ leaves that cast undefined, so a graph that relies on it
    (T5's relative-position buckets cast [log 0 = -inf] and then discard the
    result) is only reproducible by choosing the platform's behaviour, which a
    caller opts into and a report names. *)
type float_to_int = Checked | Saturating

module Policy : sig
  type t = { dot_accumulation : dot_accumulation; float_to_int : float_to_int }

  val default : t
  (** [Binary64] dots, [Checked] casts. *)
end

module type POLICY = sig
  val policy : Policy.t
end

module type S = sig
  include
    Semantics.SEMANTICS
      with type t = float
       and type b = bool
       and type 'role index = 'role Dim.t
       and type input = Tensor.packed

  include
    Semantics.TYPED_SEMANTICS
      with type 'a repr = 'a
       and type 'role index := 'role Dim.t
       and type input := Tensor.packed
       and type b := bool

  (* [i64_load]'s [Bool] counterpart -- not part of
     [Semantics.TYPED_SEMANTICS] since only [Eval_direct]'s [To_copy(Long)]
     dispatch reads a Bool operand today; no [Symbolic] twin exists yet. *)
  val bool_load : Tensor.packed -> Semantics.position Dim.t Vec6.t -> bool
end

(** The evaluator under a policy. Applied once per policy, not per run: a
    functor application builds the whole evaluator. *)
module Make (_ : POLICY) : S

include S
(** The instance under {!Policy.default}. *)
