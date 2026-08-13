(** Validate a JavaScript byte length before it is narrowed to an OCaml [int].
*)

val checked_length :
  float ->
  max:int64 ->
  (int, [> `Bad_length of float | `Over_limit of int64 ]) Err.t
(** Rejects non-finite, fractional, negative, and non-safe-integer values before
    comparing with [max] and [Me_limits.Hard.jsoo_safe_bytes]. *)
