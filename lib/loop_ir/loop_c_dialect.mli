(** The spelling of the C the emitter writes. [Gnu] is the default: it includes
    the standard headers and uses their macros, for a host compiler.
    [Compcert_scalar] is self-contained C for an embedded CompCert: no
    preprocessor directive anywhere, the types and libc/libm functions it uses
    declared in the unit's own prelude, the macros ([INT64_C], [NAN],
    [INFINITY], [signbit]) written out. It has no vector form. [Compcert_ilp32]
    is the same for a 32-bit-pointer target, where [long] is 32 bits: [int64_t]
    is [long long] and its constants end in [LL]. All compute the same bits. *)

type t = Compcert_ilp32 | Compcert_scalar | Gnu

val i64 : t -> int64 -> string
(** An [int64_t] constant. *)

val u64 : t -> int64 -> string
(** A [uint64_t] constant. *)

val nan : t -> string
val infinity : t -> string
val nan_f32 : t -> string
val infinity_f32 : t -> string

val signbit_f32 : t -> string -> string
(** The binary32 forms: [f]-suffixed literals, so a [float] operand is never
    widened by a [double] constant. *)

val signbit : t -> string -> string
(** [signbit] of a simple expression, which is evaluated more than once. *)

val binary32_only : t -> bool
(** Whether a binary32 kernel in this dialect must take no binary64 step and
    convert between 64-bit integers and [float] without the compiler's
    conversion: [Compcert_ilp32], whose F-only riscv32 profile has neither. Its
    f16 and bf16 decoders and those conversions are the binary32 helpers then;
    every dialect computes the same bits. *)

val isfinite : t -> string -> string
(** [isfinite] of a simple expression, which is evaluated more than once. *)

val host_symbols : string list
(** Every function a [Compcert_scalar] unit declares [extern]: the names to bind
    to the host when the unit is run in process. *)
