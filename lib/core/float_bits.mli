(** IEEE-754 representations used by structural identities.

    [exact] preserves every binary64 bit, including the sign of zero and NaN
    payloads. It is appropriate when the payload itself is observable.

    [portable] preserves every non-NaN representation but maps all NaNs to one
    quiet-NaN representation. JavaScript Numbers do not portably preserve NaN
    payloads, so use it for identities that must agree across native OCaml and
    JavaScript backends. *)

val exact : float -> int64
val portable : float -> int64
val compare_exact : float -> float -> int
val equal_exact : float -> float -> bool
val compare_portable : float -> float -> int
val equal_portable : float -> float -> bool
