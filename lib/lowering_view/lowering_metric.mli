(** The numbers an overlay can colour a node by. The names are the wire's; the
    type is {!Lowering_key.Metric.t}, which the protocol shares. *)

type t = Lowering_key.Metric.t =
  | Coral_cycles  (** the Coral estimate of cycles *)
  | Scalar_loops  (** loops the planner tried and kept scalar *)
  | Vector_loops  (** loops the planner vectorized *)

val all : t list
val name : t -> string
val of_name : string -> t option
