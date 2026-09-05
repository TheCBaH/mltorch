module Rhs : sig
  (* The right-hand side IS what a local is -- a scalar expression, or a
     vector body parameterized by its own per-element binder. [Shape.t] below
     is a projection of this, never a second source of truth: before this
     type existed, [t] carried [shape] and [value] as two independent
     fields that a smart constructor happened to keep in agreement, which
     cannot honestly extend to a third right-hand side whose value is not one
     [Expr.Value.t] (see the scan design record). *)
  type t = private
    | Scalar of Expr.Value.t
    | Vector of { extent : int; var : Expr.Reduce_var.t; body : Expr.Value.t }

  val scalar : Expr.Value.t -> t
  val vector : extent:int -> var:Expr.Reduce_var.t -> body:Expr.Value.t -> t
  val slot_count : t -> int

  val value : t -> Expr.Value.t
  (** The one [Expr.Value.t] every current right-hand side carries -- a scalar's
      own expression, or a vector's per-element body. Convenience for callers
      (rendering, folds) that only need to visit it, not dispatch on which case
      it came from. *)
end

module Shape : sig
  (* The declared read kind and storage extent, WITHOUT the expression body
     -- what a [Shape_mismatch] error reports, and all a shape-agreement
     check needs. Derived from [Rhs.t], never stored independently. *)
  type t = Scalar | Vector of { extent : int }

  val of_rhs : Rhs.t -> t
  val pp : Format.formatter -> t -> unit
end

type t = private { id : Expr.Local_var.t; rhs : Rhs.t }

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
