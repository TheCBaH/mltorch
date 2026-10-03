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

type error = [ `Unsupported_format of Tensor_id.t * string ]

val pp_error : Format.formatter -> [< error ] -> unit

type t = {
  source : string;  (** the function definition *)
  helpers : Loop_c_runtime.Name.t list;  (** the helpers it calls *)
  local_doubles : int64;
      (** [double]s of scratch [local] must provide: the sum of the program's
          [Alloc] sizes (each gets its own region, so nested or repeated
          allocations can never overlap) *)
  buffer_types : string list;
      (** the C cell type each buffer argument points to, in order *)
}

val kernel :
  ?vector:Loop_target.t ->
  name:string ->
  Loop_program.t ->
  (t, [> error ]) Err.t
(** [vector] vectorizes the independent loops {!Loop_vectorize} finds for the
    target (four binary64 lanes), under the strict contract, and emits them with
    GCC/Clang generic vectors; the text then needs {!Loop_c_runtime.Name}'s
    [Vector_prelude]. *)

val float_lit : float -> string
(** A C [double] constant that reads back to exactly the same bits (hexadecimal
    floating point; [NAN] and [INFINITY] for the specials). *)
