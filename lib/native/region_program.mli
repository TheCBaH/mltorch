module Local_scope : sig
  type t = { local : Expr.Local_var.t; referenced : Expr.Local_var.t }
end

module Non_invariant : sig
  type t = { local : Expr.Local_var.t; axis : Expr.Axis.t }
end

module Shape_mismatch : sig
  type read = Scalar_read | Vector_read | Scan_read

  type t = {
    local : Expr.Local_var.t;
    read : read;
    declared : Region_local.Shape.t;
  }
end

type error =
  [ `Duplicate_local of Expr.Local_var.t
  | `Expr of Expr.Check.error
  | `Forward_local of Local_scope.t
  | `Local_list_too_large of int
  | `Local_words_over_limit of int
  | `Non_invariant_local of Non_invariant.t
  | `Scan of Expr.Scan.error
  | `Scan_updates_over_limit of int64
  | `Shape_mismatch of Shape_mismatch.t
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

val preflight :
  max_local_slots:int ->
  max_scan_state:int ->
  max_scan_updates:int64 ->
  output_shape:Vec6.shape ->
  t ->
  (unit, error) Err.t
(** The RESOURCE dimensions [check] does not cover: total local/trace storage
    against [max_local_slots], peak nested scan state against [max_scan_state],
    and one Region key's worst-case recurrence-update count (every local
    materialized once, a vector [extent] times, the emitter once per output
    sharing a key -- [outputs_per_key], derived from [output_shape] and the
    program's own partition) against [max_scan_updates]. A cost ESTIMATE and
    admission tool, not a runtime guarantee: call on an already-[check]ed
    program. *)

val scan_updates_total : output_shape:Vec6.shape -> t -> int64
(** [keys * per_key]'s worst-case recurrence-update count for this ONE program,
    where [keys] is the count of distinct Region keys (the product of
    [output_shape]'s extents over the partition's singleton axes). Summed across
    a Kernel's logical values, this is [max_scan_updates_total]'s own measure --
    a Kernel-scoped aggregate, deliberately not part of [preflight] itself. *)

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

  val vector :
    extent:int ->
    (Expr.Role.Position.t Expr.Index.t -> Expr.Value.t Expr.Builder.t) ->
    ((Expr.Role.Position.t Expr.Index.t -> Expr.Value.t) -> 'a t) ->
    'a t
  (** [value] builds the local's body from its own symbolic per-element index,
      the same shape [Expr.Builder.reduction]'s body callback has. [continue]
      receives a reader: [Expr.Value.local_at id] applied at whatever index its
      caller (typically an enclosing reduction's own bound variable) supplies.
  *)

  val scan :
    limits:Expr.Scan_limits.t ->
    width:int ->
    steps:int ->
    init:(lane:Expr.Role.Position.t Expr.Index.t -> Expr.Value.t Expr.Builder.t) ->
    update:
      (step:Expr.Role.Position.t Expr.Index.t ->
      lane:Expr.Role.Position.t Expr.Index.t ->
      previous_at:(Expr.Role.Position.t Expr.Index.t -> Expr.Value.t) ->
      Expr.Value.t Expr.Builder.t) ->
    ((row:Expr.Role.Position.t Expr.Index.t ->
     lane:Expr.Role.Position.t Expr.Index.t ->
     Expr.Value.t) ->
    (program, error) Err.t t) ->
    (program, error) Err.t t
  (** Declares a trace local via [Expr.Builder.scan] and hands [continue] a
      cached reader: [Expr.Value.local_scan_at id], applied at whatever row/lane
      its caller supplies -- the trace-local counterpart to [vector]'s
      per-element reader. Unlike [scalar]/[vector], this has a failure channel:
      [Expr.Builder.scan]'s own construction-time checks (steps/width sanity,
      [step]/[prev] not free in [init], the descriptor's worst case against
      [limits]) can reject the descriptor, in which case [continue] is never
      invoked and the whole chain short-circuits with [Err.fail (`Scan _)]. *)

  val finish :
    max_size:int ->
    max_depth:int ->
    partition:Region_partition.t ->
    output:Expr.Value.t ->
    (program, error) Err.t t
end
