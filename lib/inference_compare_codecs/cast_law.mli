(** Which casts the codec layer admits, and which of them preserve every value.

    A recipe's own [lossless] flag is a claim; this table is the evidence. A
    cast is lossless only when every value of the source codec is represented
    exactly in the target, so a provider cannot launder a narrowing cast by
    declaring it lossless. *)

open Inference_compare_schema

type verdict = Lossless | Lossy | Unsupported

val classify : Element_codec.t -> Element_codec.t -> verdict
(** Same-codec casts are [Lossless]. *)

val rounding_admitted :
  source:Element_codec.t ->
  target:Element_codec.t ->
  Conversion_step.rounding ->
  bool
(** Integer targets from floats admit every rounding; float targets from
    integers admit only nearest-even. Casts between like families need none, and
    any rounding is accepted as an inert declaration there. *)

val overflow_admitted :
  source:Element_codec.t ->
  target:Element_codec.t ->
  Conversion_step.overflow ->
  bool
(** [Wrap] is admitted only between integer codecs. *)
