(** C emission for a [Loop_program.t]: one [static int] function over typed
    pointers, ordinary [for] loops, no allocation and no global state. Output is
    deterministic: emitting twice is byte-identical, and names depend on the
    position of first appearance, never on allocation ids.

    The function takes [struct model_error *err], a [double *local] scratch
    region (see {!t.local_doubles}), then one pointer per program buffer in
    program order ([b0], [b1], ...). It returns [0] on success or [1] after
    filling [*err] (see {!Loop_c_runtime}): a failure is a value, never an
    exception. The [invocation] field is left for the caller to fill. Emitted
    text needs {!Loop_c_runtime.prelude} and the helpers listed in {!t.helpers}
    before it.

    Only formats with a C implementation are admitted: [bf16], [bool], [f16],
    [f32], [f64], [i32] and [i64]. A quantized buffer is a typed refusal. *)

type error =
  [ `Unsupported_format of Tensor_id.t * string
  | `Unsupported_precision of Loop_numerics.Refusal.t
    (** [precision] was [F32] and {!Loop_numerics.admit} refused the program *)
  ]

val pp_error : Format.formatter -> [< error ] -> unit

type t = {
  source : string;  (** the function definition *)
  prelude : string;
      (** text the definition needs before it that is not a runtime helper (an
          emitter's own types and functions), made of blocks guarded so that
          several kernels' preludes can sit in one unit: empty for this module's
          own kernels *)
  helpers : Loop_c_runtime.Name.t list;  (** the helpers it calls *)
  local_doubles : int64;
      (** [double]s of scratch [local] must provide: the sum of the program's
          [Alloc] sizes (each gets its own region, so nested or repeated
          allocations can never overlap) *)
  precision : Loop_numerics.Precision.t;
      (** the working precision the kernel was emitted in *)
  refusal : Loop_numerics.Refusal.t option;
      (** why an fp32 policy left it binary64, when admission was the reason *)
  buffer_types : string list;
      (** the C cell type each buffer argument points to, in order *)
}

val kernel :
  ?vector:Loop_target.t ->
  ?numerics:Loop_numerics.t ->
  ?precision:Loop_numerics.Precision.t ->
  ?fuse_reductions:bool ->
  name:string ->
  Loop_program.t ->
  (t, [> error ]) Err.t
(** [vector] vectorizes the independent loops {!Loop_vectorize} finds for the
    target, under the strict contract, and emits them with GCC/Clang generic
    vectors; the text then needs {!Loop_c_runtime.Name}'s [Vector_prelude] (four
    binary64 lanes) or [Vector_prelude_f32] ({!Loop_c_runtime.f32_lanes}
    binary32 lanes).

    [numerics] ({!Loop_numerics.Reference_f64} by default) resolves the kernel's
    working precision through {!Loop_plan.resolve}: a [Simd_fp32_*] policy runs
    a kernel the planner vectorizes entirely in binary32 and leaves every other
    kernel binary64, byte-identical to [Reference_f64]'s.

    [precision] forces the working precision instead (for tests and scalar
    emission). [F32] computes the whole kernel in binary32: [float] temporaries,
    [f]-suffixed constants, [Round_f32] the identity, transcendentals the
    binary64 function on the widened argument rounded once, local arrays two
    cells to a scratch [double]. A program {!Loop_numerics.admit} refuses is a
    typed [`Unsupported_precision]. *)

val float_lit : float -> string
(** A C [double] constant that reads back to exactly the same bits (hexadecimal
    floating point; [NAN] and [INFINITY] for the specials). *)
