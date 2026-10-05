(** The numerical policy of a generated kernel, kept apart from storage formats
    ({!Ssa_format}) and from the physical target ({!Ssa_target}). Working
    precision is chosen by role: general compute stays binary64, a kernel the
    vector planner takes runs in binary32. The policy names the permissions; a
    resolved plan records each kernel's {!Precision}.

    These are the same three presets, names and identities the Loop planner
    uses, restated here because this library sees no Loop type: the bridge suite
    checks the two agree. *)

module Precision : sig
  type t = F32 | F64

  val name : t -> string
end

type t =
  | Reference_f64
      (** every kernel binary64, no contraction, no reassociation: the default
          everywhere, the oracle and the debugging path *)
  | Simd_fp32_ordered
      (** vectorized kernels binary32, each output's operation order unchanged,
          contraction disabled: bitwise against the binary32 oracle *)
  | Simd_fp32_relaxed
      (** vectorized kernels binary32, reordered sums and contraction permitted
          where the plan records them *)

val all : t list
val name : t -> string
val of_name : string -> t option
val contraction_permitted : t -> bool
val reassociation_permitted : t -> bool

val identity : t -> string
(** A stable text for artifact identity: policy name, the permissions it grants
    and the working precision of the kernels it vectorizes. Different policies
    never share one. *)

(** The executors that can be asked to run a policy. *)
module Backend : sig
  type t = C | Interpreter | Javascript | Wasm

  val name : t -> string
end

type unsupported = [ `Unsupported_policy of Backend.t * t ]

val pp_unsupported : Format.formatter -> [< unsupported ] -> unit

val check : backend:Backend.t -> t -> (t, [> unsupported ]) Err.t
(** [Reference_f64] everywhere; a [Simd_fp32_*] policy only for C and Wasm. *)

(** Why a kernel stays binary64 under a [Simd_fp32_*] policy. *)
module Refusal : sig
  type t =
    | Wide_load of Ssa_id.Buffer.t * Ssa_format.Family.t
        (** a float read of a payload with no binary32 decode: a dequantizing
            read of an [i8] or [i16] buffer *)

  val pp : Format.formatter -> t -> unit
end

val admit : Ssa_program.t -> (unit, Refusal.t) result
(** Whether the whole program can run in binary32: every float read has a decode
    to a value binary32 holds or rounds once ([f32], [f16], [bf16], [bool],
    [f64], [i32], [i64]). The first refusal in program order. *)

val coverage :
  numerics:t ->
  f32_kernels:int ->
  kernels:int ->
  f32_invocations:int ->
  invocations:int ->
  refusals:(Refusal.t * int) list ->
  string
(** One line for a host to print: the policy, how many distinct kernels and
    invocations ran in binary32, and every invocation left binary64 because
    admission refused it, by reason. *)

val erf32 : float -> float
(** [Expr.Value]'s erf polynomial with a rounding after every operation and
    [exp] of the rounded argument rounded once: the sequence a binary32 kernel
    follows. *)

val fma32 : float -> float -> float -> float
(** [fma32 a b c] is binary32 [fmaf a b c] for binary32 operands held in
    binary64: [a * b + c] rounded once, never through a binary64 rounding of the
    sum first (which differs from [round32 (Float.fma a b c)] when that sum
    falls on a binary32 midpoint after its own rounding). *)
