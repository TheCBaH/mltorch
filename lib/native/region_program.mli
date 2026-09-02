module Local_scope : sig
  type t = { local : Expr.Local_var.t; referenced : Expr.Local_var.t }
end

module Non_invariant : sig
  type t = { local : Expr.Local_var.t; axis : Expr.Axis.t }
end

type error =
  [ `Duplicate_local of Expr.Local_var.t
  | `Expr of Expr.Check.error
  | `Forward_local of Local_scope.t
  | `Local_list_too_large of int
  | `Non_invariant_local of Non_invariant.t
  | `Unknown_emitter_local of Expr.Local_var.t
  | `Unknown_local of Local_scope.t ]

type t = private {
  partition : Region_partition.t;
  locals : Region_local.t list;
  output : Expr.Value.t;
}

type program = t

val pixel : Expr.Value.t -> t
val partition : t -> Region_partition.t
val locals : t -> Region_local.t list
val output : t -> Expr.Value.t
val with_output : t -> Expr.Value.t -> t
val pixel_expression : t -> Expr.Value.t option

val specialize_pixel :
  max_size:int -> max_depth:int -> t -> (Expr.Value.t, error) Err.t
(** Symbolic expansion of a Region program into one Pixel expression. This is
    distinct from concrete scalar projection. *)

val reconstructs :
  max_size:int ->
  max_depth:int ->
  pixel:Expr.Value.t ->
  t ->
  (bool, error) Err.t

val create :
  max_size:int ->
  max_depth:int ->
  partition:Region_partition.t ->
  locals:Region_local.t list ->
  output:Expr.Value.t ->
  (t, error) Err.t

val check : max_size:int -> max_depth:int -> t -> (unit, error) Err.t
val pp_error : Format.formatter -> [< error ] -> unit
val pp : Format.formatter -> t -> unit

module Fold : sig
  val sources : t -> Expr.Source.Set.t

  val loads :
    t -> (Expr.Source.t * Expr.Role.Position.t Expr.Index.t Expr.Coord.t) list

  val intrinsic_sources : t -> Expr.Source.t list
  val binders : t -> Expr.Reduce_var.t list
  val intrinsics : t -> int
  val max_depth : t -> int
  val size : t -> int
end

module Builder : sig
  type 'a t

  val run : 'a t -> 'a
  val scalar : Expr.Value.t -> (Expr.Value.t -> 'a t) -> 'a t

  val finish :
    max_size:int ->
    max_depth:int ->
    partition:Region_partition.t ->
    output:Expr.Value.t ->
    (program, error) Err.t t
end
