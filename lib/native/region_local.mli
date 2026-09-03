module Shape : sig
  type t = private
    | Scalar
    | Vector of { extent : int; var : Expr.Reduce_var.t }

  val scalar : t
  val vector : extent:int -> var:Expr.Reduce_var.t -> t
  val slot_count : t -> int
  val pp : Format.formatter -> t -> unit
end

type t = private {
  id : Expr.Local_var.t;
  shape : Shape.t;
  value : Expr.Value.t;
}

val scalar : id:Expr.Local_var.t -> value:Expr.Value.t -> t

val vector :
  id:Expr.Local_var.t ->
  var:Expr.Reduce_var.t ->
  extent:int ->
  value:Expr.Value.t ->
  t
(** [value]'s body may mention [var] (via [Expr.Index.reduce var]) as its own
    per-element index -- free within [value], never bound by a nested reduction.
    A read at a computed index ([Expr.Value.local_at]) beta-reduces that binder
    against the read's own index during specialization. *)
