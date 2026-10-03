(** The numerical policy of a generated kernel, kept apart from storage formats
    ([Tensor_sig.fmt]) and from the physical target ([Loop_target]).

    Working precision is chosen by role: general compute stays binary64, a
    kernel the performance planner vectorizes computes in binary32. The policy
    names the permissions; the resolved plan records each kernel's {!Precision}.
    Only generated C and Wasm accept a [Simd_fp32_*] policy. *)

(** The working precision of one kernel. *)
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
          contraction disabled: bitwise against the fp32 oracle *)
  | Simd_fp32_relaxed
      (** vectorized kernels binary32, reordered sums and contraction permitted
          where the plan records them *)

val all : t list
val name : t -> string
val of_name : string -> t option
val contraction_permitted : t -> bool
val reassociation_permitted : t -> bool

val identity : t -> string
(** A stable text for artifact identity and manifests: policy name, the
    permissions it grants, and the working precision of the kernels it
    vectorizes. Different policies never share one. *)

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
    | Wide_load of Tensor_id.t * string
        (** a float read of a payload with no binary32 decode: the format name
        *)

  val pp : Format.formatter -> t -> unit
end

val admit : Loop_program.t -> (unit, Refusal.t) result
(** Whether the whole program can run in binary32: every float read of a payload
    with a decode ([f32], [f16], [bf16], [bool], and [f64], [i32], [i64] rounded
    once), an int64 or an index converted to a float once. The first refusal in
    program order. *)

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
    admission refused it, by reason. A host reports it so mixed execution never
    passes for full fp32 coverage. *)

val round32 : float -> float
(** The binary32 value nearest [x] (round to nearest even), as a binary64. For
    [+ - * /] and [sqrt] of binary32 operands, [round32 (op a b)] is the
    correctly rounded binary32 result: binary64 carries [2p+2] significand bits,
    so rounding twice is innocuous. *)

val round32_of_i64 : int64 -> float
(** One rounding from an int64 to binary32, never through binary64 first. *)

val erf32 : float -> float
(** [Expr.Value]'s erf polynomial with a rounding after every operation and
    [exp] of the rounded argument rounded once: the sequence the generated C
    float helper transcribes. *)

val f32_literal : float -> string
(** A C [float] constant (hexadecimal, [f]-suffixed, so nothing promotes through
    [double]) reading back to exactly [round32 x]; [NAN] and [INFINITY] for the
    specials. *)

val fma32 : float -> float -> float -> float
(** [fma32 a b c] is binary32 [fmaf a b c] for binary32 operands held in
    binary64: [a * b + c] rounded once, never through a binary64 rounding of the
    sum first (which differs from [round32 (Float.fma a b c)] when that sum
    falls on a binary32 midpoint after its own rounding). Checked against the
    host's [fmaf] over a large sample in the C suite. *)
