(** The registered codec operations a [Codec_operation] step may name. An
    operation is admitted by name and by implementation digest together: a step
    whose digest differs from the registered one is a different implementation
    and is refused. *)

open Inference_compare_schema

module Operation : sig
  type t = {
    name : Id.Codec_operation.t;
    implementation_digest : Digest.Implementation.t;
    arity : int;
    lossless : bool;  (** Declared by the registry, not by the artifact. *)
    infer : Tensor_spec.t list -> (Tensor_spec.t, Codec_error.t) result;
    apply :
      Logical_value.Context.t ->
      Logical_value.t list ->
      (Logical_value.t, Codec_error.t) Err.t;
  }
end

type t

val empty : t

val register :
  t ->
  Operation.t ->
  (t, [> `Duplicate_operation of Id.Codec_operation.t ]) Err.t

val find : t -> Id.Codec_operation.t -> Operation.t option
val capabilities : t -> Capability.Operation.t list
