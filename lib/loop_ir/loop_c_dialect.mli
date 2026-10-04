(** The spelling of the C the emitter writes. [Gnu] is the default: it includes
    the standard headers and uses their macros, for a host compiler.
    [Compcert_scalar] is self-contained C for an embedded CompCert: no
    preprocessor directive anywhere, the types and libc/libm functions it uses
    declared in the unit's own prelude, the macros ([INT64_C], [NAN],
    [INFINITY], [signbit]) written out. It has no vector form. Both compute the
    same bits. *)

type t = Compcert_scalar | Gnu

val i64 : t -> int64 -> string
(** An [int64_t] constant. *)

val u64 : t -> int64 -> string
(** A [uint64_t] constant. *)

val nan : t -> string
val infinity : t -> string

val signbit : t -> string -> string
(** [signbit] of a simple expression, which is evaluated more than once. *)

val isfinite : t -> string -> string
(** [isfinite] of a simple expression, which is evaluated more than once. *)

val host_symbols : string list
(** Every function a [Compcert_scalar] unit declares [extern]: the names to bind
    to the host when the unit is run in process. *)
