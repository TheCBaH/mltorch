(** Element-level kernels for the built-in conversion steps. Each takes
    validated inputs (checked by {!Recipe_eval.check}) and produces the output
    bytes; none reads outside its operands. Extents and strides here are [int]
    because the callers bounded every payload by [Sys.max_string_length] first.
*)

open Inference_compare_schema

val permute : width:int -> axes:int list -> shape:int list -> string -> string

val slice :
  width:int ->
  axis:int ->
  start:int ->
  length:int ->
  shape:int list ->
  string ->
  string

val concat : width:int -> axis:int -> (int list * string) list -> string

val cast :
  source:Element_codec.t ->
  target:Element_codec.t ->
  rounding:Conversion_step.rounding ->
  overflow:Conversion_step.overflow ->
  count:int ->
  string ->
  (string, Codec_error.t) result
(** [count] is the element count. *)
