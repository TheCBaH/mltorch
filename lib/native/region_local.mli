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
    | Scan of Expr.Scan.t

  val scalar : Expr.Value.t -> t
  val vector : extent:int -> var:Expr.Reduce_var.t -> body:Expr.Value.t -> t
  val scan : Expr.Scan.t -> t
  val slot_count : t -> int

  val value : t -> Expr.Value.t
  (** The one [Expr.Value.t] a scalar/vector right-hand side carries -- a
      scalar's own expression, or a vector's per-element body. For a scan, a
      foldable stand-in wrapped as [Expr.Value.scan_at] at closed placeholder
      indices: convenience for callers (folds, scope/shape checks) that only
      need to visit the descriptor structurally, never for rendering it to a
      reader -- an unspecialized scan must show [init]/[update] directly (see
      the scan design record). *)
end

module Shape : sig
  (* The declared read kind and storage extent, WITHOUT the expression body
     -- what a [Shape_mismatch] error reports, and all a shape-agreement
     check needs. Derived from [Rhs.t], never stored independently. *)
  type t =
    | Scalar
    | Vector of { extent : int }
    | Scan of { width : int; steps : int }

  val of_rhs : Rhs.t -> t
  val pp : Format.formatter -> t -> unit
end

type t = private { id : Expr.Local_var.t; rhs : Rhs.t }

val scalar : id:Expr.Local_var.t -> value:Expr.Value.t -> t
val scan : id:Expr.Local_var.t -> scan:Expr.Scan.t -> t

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
