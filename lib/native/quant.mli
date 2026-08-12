(* Affine (asymmetric) quantization: real = scale * (q - zero_point).
   Granularity is per-tensor or per-channel along C. See
   .ai/native_tensor_design.md §3.

   ABSTRACT, and deliberately so. The representation holds a [float array] and
   an [int array], which a public variant would hand out by reference: a caller
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

type error = [ `Quant_array_lengths of int * int ]
(** scale, zero_point *)

val pp_error : Format.formatter -> [< error ] -> unit
val per_tensor : scale:float -> zero_point:int -> t

val per_channel : scale:float array -> zero_point:int array -> (t, error) Err.t
(** Copies both arrays and rejects unequal lengths. Copying establishes
    ownership; it does not establish the invariant, which is why this is
    result-valued and [per_tensor] is not. *)

val channel_count : t -> int option
(** [None] for per-tensor. The only way to ask about granularity once the
    variant is hidden — and probing with [params] is not a substitute, since a
    per-tensor value accepts every channel while a short per-channel array
    raises at exactly the boundary a caller is trying to make safe. *)

val params : t -> c:int -> float * int
val dequantize : t -> c:int -> q:int -> float
val quantize : t -> c:int -> qmin:int -> qmax:int -> float -> int

val equal : t -> t -> bool
(** Granularity, array lengths, every zero point, and every scale by
    [Int64.bits_of_float] — never [Float.equal], which equates -0. with 0. and
    any NaN with any other. Same rule as [Tensor.equal_bits]. *)

val pp : Format.formatter -> t -> unit
val jsont : t Jsont.t

val of_err_for_jsont : ('a, error) Err.t -> 'a
(** Carries a constructor failure into Jsont's error boundary as
    [Jsont.Error.msgf], so a malformed document becomes [decode_string]'s
    ordinary [Error _] instead of escaping as an exception. NAMED rather than
    inlined because it drops the [Err.Error.t] wrapper on purpose, and CLAUDE.md
    requires that to stay distinguishable from the defect. *)
