(* Arithmetic that mixes [Dim] roles with [Op_config] quantities — a stride, a
   dilation, a group count meeting a position or an extent. It sits after
   [Op_config] because [Op_config] already depends on [Dim]; same-role
   arithmetic stays in [Dim]. *)

module Delta : sig
  (* [by * d]; a stride or dilation applied to a window position. *)
  val scale : by:Op_config.Pos.t -> Dim.delta Dim.t -> Dim.delta Dim.t

  (* Floor and ceiling division by a positive quantity, rounding toward
     negative and positive infinity respectively. *)
  val floor_div_pos : Dim.delta Dim.t -> by:Op_config.Pos.t -> Dim.delta Dim.t
  val ceil_div_pos : Dim.delta Dim.t -> by:Op_config.Pos.t -> Dim.delta Dim.t
end

module Extent : sig
  (* [e / by] when the group count [by] divides [e] exactly: the channels one
     group holds. *)
  val to_pos : Dim.extent Dim.t -> Op_config.Pos.t
  (** An extent used as a divisor or a stride: both are at least 1. *)

  val of_pos : Op_config.Pos.t -> Dim.extent Dim.t
  (** A count that is at least 1, taken as an extent: a requested output size.
  *)

  val div_exact :
    by:Op_config.Pos.t -> Dim.extent Dim.t -> Dim.extent Dim.t option

  (* [by * e], exclusive of [limit] — [Dim.product_bounded]'s contract, so a
     channel count times a group count cannot wrap a 32-bit [int]. *)
  val scale :
    limit:int64 ->
    by:Op_config.Pos.t ->
    Dim.extent Dim.t ->
    (Dim.extent Dim.t, [> `Product_over_limit of Dim.Product_witness.t ]) Err.t
end
