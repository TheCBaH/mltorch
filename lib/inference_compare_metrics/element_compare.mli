(** Element-by-element comparison of two logical values in comparison
    coordinates, under a {!Comparison_policy.t}.

    Every element is inspected. Exactness (integers, Booleans), bitwise equality
    and the configured tolerance rule are separate counts. Exceptional operands
    (NaN, infinities, signed zeros) are classified before any arithmetic, and
    the tolerance rule [d <= atol + rtol * s] is evaluated in the policy's
    declared arithmetic with no implicit epsilon. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs

type error =
  | Codec of Codecs.Codec_error.t
  | Policy_codec_mismatch  (** The element rule does not fit the codec. *)
  | Spec_mismatch

val pp_error : Format.formatter -> error -> unit

val compare :
  ?max_samples:int ->
  Codecs.Logical_value.Context.t ->
  Comparison_policy.t ->
  left:Codecs.Logical_value.t ->
  right:Codecs.Logical_value.t ->
  (Element_result.t, error) Err.t
(** Both values must have equal specs. [max_samples] (default 8) bounds the
    retained difference samples. Statistics are binary64 and independent of the
    acceptance arithmetic. *)

val rule_passed : Element_result.t -> bool option
(** [None] when no element rule was configured. *)
