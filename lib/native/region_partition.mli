module Axis_mode : sig
  type t = Singleton | Whole

  val pp : Format.formatter -> t -> unit
end

type t = private Axis_mode.t Vec6.t

type error =
  [ `Duplicate_axis of Expr.Axis.t | `Output_out_of_bounds of Vec6.coord ]

val pp_error : Format.formatter -> [< error ] -> unit
val singleton : t

val of_whole_axes :
  Expr.Axis.t list -> (t, [> `Duplicate_axis of Expr.Axis.t ]) Err.t

val mode : t -> Expr.Axis.t -> Axis_mode.t
val is_singleton : t -> bool
val whole_axes : t -> Expr.Axis.t list
val key_shape : output_shape:Vec6.shape -> t -> Vec6.shape

val key_of_output :
  output_shape:Vec6.shape -> t -> Vec6.coord -> (Vec6.coord, error) Err.t

val fold_keys :
  output_shape:Vec6.shape -> init:'a -> f:('a -> Vec6.coord -> 'a) -> t -> 'a

val fold_outputs :
  output_shape:Vec6.shape ->
  key:Vec6.coord ->
  init:'a ->
  f:('a -> Vec6.coord -> 'a) ->
  t ->
  'a

val pp : Format.formatter -> t -> unit
