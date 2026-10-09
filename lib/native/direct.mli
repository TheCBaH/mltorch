(* Concrete evaluation: values are floats, indices are ['role Dim.t] (see
   direct.ml), an input is a real tensor. See .ai/native_compute_design.md §3. *)

include
  Semantics.SEMANTICS
    with type t = float
     and type 'role index = 'role Dim.t
     and type input = Tensor.packed

(** How {!dot} accumulates. [Binary64] (the default) sums the exact products in
    binary64, so one rounding to binary32 happens on store.
    [Binary32_sequential] rounds to binary32 after each fused multiply-add of a
    sequential chain, which reproduces a reference that accumulates in binary32
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

val with_float_to_int : float_to_int -> (unit -> 'a) -> 'a
(** Scoped like {!with_dot_accumulation}. *)

val with_dot_accumulation : dot_accumulation -> (unit -> 'a) -> 'a
(** Runs the thunk with the policy, restoring the previous one even if it
    raises. Scoped state: the engine is single-threaded. *)

include
  Semantics.TYPED_SEMANTICS
    with type 'a repr = 'a
     and type 'role index := 'role Dim.t
     and type input := Tensor.packed
     and type b := bool

(* [i64_load]'s [Bool] counterpart -- not part of [Semantics.TYPED_SEMANTICS]
   since only [Eval_direct]'s [To_copy(Long)] dispatch reads a Bool operand
   today; no [Symbolic] twin exists yet. *)
val bool_load : Tensor.packed -> Semantics.position Dim.t Vec6.t -> bool
