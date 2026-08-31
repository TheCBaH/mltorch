module Shape : sig
  type t

  val scalar : t
  val pp : Format.formatter -> t -> unit
end

type t = private {
  id : Expr.Local_var.t;
  shape : Shape.t;
  value : Expr.Value.t;
}

val scalar : id:Expr.Local_var.t -> value:Expr.Value.t -> t
