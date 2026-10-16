(* Affine (asymmetric) quantization: real = scale * (q - zero_point).
   Granularity is per-tensor or per-channel along C. See
   .ai/native_tensor_design.md §3.

   ABSTRACT, and deliberately so. The representation holds two [float array]s,
   which a public variant would hand out by reference: a caller
   could then reach a validated [Tensor_sig.t] through [kernel.inputs] and
   mutate the semantic metadata after construction — or after a [Fusion_plan.t]
   was built over it. Copying inside [Kernel.create] would not help, because the
   copy stays reachable and mutable. The constructors below copy, and nothing
   here returns the stored arrays.

   The two per-channel arrays having EQUAL LENGTH is this module's invariant,
   established by [per_channel]. That the length equals a tensor's C extent is a
   different claim, belonging to whoever owns the shape; [channel_count] is what
   lets them check it. *)

type t

type error =
  [ `Quant_array_lengths of int * int  (** scale, zero_point *)
  | `Quant_not_binary32 of [ `Scale | `Zero_point ] * float
    (** a parameter that [round32] would change (or a non-finite one) *) ]

val pp_error : Format.formatter -> [< error ] -> unit

val round_scale : float -> float * bool
(** The import rule: a producer's binary64 scale rounded once to binary32, and
    whether that changed it. Applied at the boundary into {!t}, never in a
    kernel; the importer reports a change. *)

val per_tensor : scale:float -> zero_point:float -> t
(** Both parameters must be binary32 values ([round32 x = x], finite): a caller
    holding a producer's double rounds it with {!round_scale} first.
    [Invalid_argument] otherwise. *)

val per_tensor_checked :
  scale:float -> zero_point:float -> (t, [> error ]) Err.t
(** {!per_tensor} with the violation as a typed error. *)

val per_channel :
  scale:float array -> zero_point:float array -> (t, [> error ]) Err.t
(** Copies both arrays and rejects unequal lengths and any element that is not a
    binary32 value. Copying establishes ownership; it does not establish the
    invariant, which is why this is result-valued. *)

val channel_count : t -> int option
(** [None] for per-tensor. The only way to ask about granularity once the
    variant is hidden — and probing with [params] is not a substitute, since a
    per-tensor value accepts every channel while a short per-channel array
    raises at exactly the boundary a caller is trying to make safe. *)

val params : t -> c:Dim.index Dim.t -> float * float
(** scale, zero point *)

val dequantize : t -> c:Dim.index Dim.t -> q:int -> float
(** The binary64 reference: [scale * (q - zero_point)] over the binary32
    parameters, evaluated in binary64. *)

val dequantize32 : t -> c:Dim.index Dim.t -> q:int -> float
(** The binary32 meaning, as the explicit operations a binary32 kernel runs: [q]
    to float (exact), minus the zero point, times the scale, in that order, each
    rounded once to binary32. Equal to {!dequantize} whenever the zero point is
    an integer and the scale times the difference needs at most 24 bits, and
    otherwise within binary32 rounding of it. *)

val quantize : t -> c:Dim.index Dim.t -> qmin:int -> qmax:int -> float -> int
(** [round (x / scale + zero_point)], half to even, clamped to [[qmin, qmax]] in
    float before the conversion to an integer: an infinity or a value outside
    the range saturates, and a NaN is the level the zero point names, clamped.
    Never an unspecified conversion. *)

val equal : t -> t -> bool
(** Granularity, array lengths, every zero point, and every scale by
    [Core.Float_bits.exact] — never [Float.equal], which equates -0. with 0.
    Same rule as [Tensor.equal_bits]. *)

val pp : Format.formatter -> t -> unit
val jsont : t Jsont.t

val of_err_for_jsont : ('a, [< error ]) Err.t -> 'a
(** Carries a constructor failure into Jsont's error boundary as
    [Jsont.Error.msgf], so a malformed document becomes [decode_string]'s
    ordinary [Error _] instead of escaping as an exception. NAMED rather than
    inlined because it drops the [Err.Error.t] wrapper on purpose, and CLAUDE.md
    requires that to stay distinguishable from the defect. *)
