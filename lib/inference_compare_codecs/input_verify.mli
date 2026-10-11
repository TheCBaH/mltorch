(** Exact verification of the inputs actually submitted to an engine.

    The imported handles are exported, read back as logical values, and run
    through the side's reconstruction recipe; the result must equal the common
    canonical value element for element, dtype and spec included. A digest is
    never consulted for equality: two different payloads can in principle share
    a digest, and only the bytes decide. *)

open Inference_compare_schema

type difference =
  | Spec_differs of { expected : Tensor_spec.t; actual : Tensor_spec.t }
  | Element_differs of { index : int64 }
      (** The first differing element in row-major order. *)

type outcome =
  | Equal of Recipe_eval.Lineage.t
  | Different of { difference : difference; lineage : Recipe_eval.Lineage.t }

val first_difference :
  Logical_value.Context.t ->
  expected:Logical_value.t ->
  actual:Logical_value.t ->
  (difference option, Codec_error.t) Err.t

val verify :
  Logical_value.Context.t ->
  Recipe_eval.env ->
  Recipe_eval.Checked.t ->
  imported:(Tensor_ref.t -> Logical_value.t option) ->
  expected:Logical_value.t ->
  (outcome, Codec_error.t) Err.t
(** Evaluates the reconstruction recipe over [imported] (its [Boundary]
    arguments are the engine-side tensors) and compares the result to
    [expected]. The reconstructed value is released before returning. *)
