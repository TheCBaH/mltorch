(* A validated, node-invocation-scoped bundle of shared Region locals plus
   several ordered emitters, each reading those locals at its own physical
   output coordinate through a checked canonical-to-physical axis mapping.
   See _ai_/shared_multi_output_impl.md §3 (project step 19): LSTM's three
   outputs (output/h_n/c_n) build one shared recurrence trace once and read
   it three different ways, rather than each output rebuilding its own
   independent copy of every layer/direction's scan. This module owns the
   representation and the checked construction/projection; it knows nothing
   about LSTM or any other specific operation. *)

(* Which emitter of a group: the position in its ordered emitter list. Not a
   node's output ordinal, though the two coincide for the callers today. *)
module Ordinal : Core.Tagged_int.S

module Emitter : sig
  type t = {
    output_shape : Vec6.shape;
    partition : Region_partition.t;
    key_axes : (Expr.Axis.t * Expr.Axis.t) list;
        (** [(canonical_axis, physical_axis)] pairs: which of this emitter's own
            [Region_partition.Axis_mode.Singleton] axes carries the canonical
            group's key coordinate declared at that [canonical_axis]. *)
    output : float Expr.Value.t;
        (** In this emitter's OWN physical output coordinates -- never
            rewritten; only the shared locals it reads are projected per emitter
            (see [project]). *)
  }
end

module Extent_mismatch : sig
  type t = { canonical : Expr.Axis.t; physical : Expr.Axis.t }
end

type mapping_error =
  | Duplicate_canonical_axis of Expr.Axis.t
  | Duplicate_target_axis of Expr.Axis.t
  | Extent_mismatch of Extent_mismatch.t
  | Target_not_singleton of Expr.Axis.t
  | Uncovered_canonical_axis of Expr.Axis.t
  | Uncovered_physical_axis of Expr.Axis.t

type error =
  [ `Empty_emitters
  | `Mapping of Ordinal.t * mapping_error
  | `Program of Region_program.error
  | `Scan of Expr.Scan.error
  | `Unknown_emitter of Ordinal.t ]

type t
(** Private: canonical shape/partition, the shared locals, and the ordered
    emitter list. Grouping identity is scoped to one call to [create] -- two
    structurally identical invocations produce two distinct [t] values, never
    merged or cached (see the design record's "grouping identity is scoped to
    one node invocation"). *)

val create :
  max_size:int ->
  max_depth:int ->
  canonical_shape:Vec6.shape ->
  locals:Region_local.t list ->
  emitters:Emitter.t list ->
  (t, error) Err.t
(** Validates every emitter's axis mapping structurally, validates the shared
    locals against the canonical partition alone (a shared local may only vary
    over the declared canonical key axes -- reuses [Region_program.check]'s own
    [Non_invariant_local] rule for this), and validates every emitter's full
    projected program (reuses [project]). Nothing is retained unless all three
    pass for every emitter. *)

val sources : t -> Ordinal.t -> Expr.Source.Set.t option
(** [None] only for an out-of-range ordinal; otherwise the union of
    [Expr.Fold.sources] over every shared local's raw (unprojected) RHS plus
    that one emitter's own raw output expression. Sources don't depend on axis
    substitution, so this needs no [project]. *)

val max_depth : t -> Ordinal.t -> int option
(** [None] only for an out-of-range ordinal; otherwise the max [Expr.Fold.depth]
    over every shared local's raw RHS and that one emitter's own raw output.
    [project]'s substitution replaces one [Output] leaf with another, never
    restructuring the tree, so this equals [Region_program.Fold.max_depth] of
    the projected program -- computed directly on the raw group, no [project]
    needed. *)

val intrinsic_sources : t -> Ordinal.t -> Expr.Source.t list option
val binders : t -> Ordinal.t -> Expr.Reduce_var.t list option

val intrinsics : t -> Ordinal.t -> int option
(** Same "no [project] needed" reasoning as [sources]/[max_depth]: none of these
    depend on which axis a leaf's [Output] variable names. *)

val canonical_shape : t -> Vec6.shape

val canonical_partition : t -> Region_partition.t
(** Singleton at exactly the axes some emitter declares as a [key_axes]
    canonical side; [Whole] everywhere else. Exposed so a caller can
    [Region_partition.fold_keys] over [canonical_shape] without re-deriving this
    split. *)

val locals : t -> Region_local.t list
val emitters : t -> Emitter.t list
val emitter : t -> Ordinal.t -> Emitter.t option

val indexed_emitters : t -> (Ordinal.t * Emitter.t) list
(** Every emitter with its own ordinal, in order. *)

val project :
  max_size:int ->
  max_depth:int ->
  t ->
  Ordinal.t ->
  (Region_program.t, error) Err.t
(** The one substitute-then-check operation: rewrites the shared locals'
    output-axis references from canonical to emitter [ordinal]'s own physical
    axes (via [Expr.Rewrite.substitute_output], never touching reducer/local
    identities), leaves the emitter's own output expression untouched, and
    re-validates the result as an ordinary [Region_program.t] under that
    emitter's own partition. Used both by [create] (once per emitter, to prove
    every projection is well-formed) and by any later caller needing an
    executable single-output program for one emitter (e.g. Kernel's scalar
    [value_at] fallback). *)

val finish :
  max_size:int ->
  max_depth:int ->
  canonical_shape:Vec6.shape ->
  emitters:Emitter.t list ->
  (t, error) Err.t Region_program.Builder.t
(** Same shape as [Region_program.Builder.finish], so it slots directly into a
    [Region_program.Builder.t]'s CPS-threaded [finish] callback (a group builder
    shares the exact same [scalar]/[vector]/[scan] combinators, only the final
    step differs -- see [Region_program.Builder.scan]'s own polymorphic result,
    which is what makes this possible without a second scan builder). Lives here
    rather than in [region_program.ml] to avoid a [Region_program] <->
    [Region_group] module cycle: this module already depends on [Region_program]
    for [Region_program.t]/[check]/[error]. *)

val pp : t Fmt.t
val pp_error : error Fmt.t

type region_group = t
(** An alias for this module's own [t], usable unqualified inside [Ref] below,
    where [t] instead names [Ref.t]. *)

(* A single stage/value's own computation: either a standalone program, or a
   reference into one ordinal of a shared group. Stage_program.Stage.t and
   Kernel.Value.t both carry this instead of a bare [Region_program.t], so
   several sibling stages/values can share one physically-identical
   [Region_group.t] instance rather than each rebuilding an independent copy
   -- see _ai_/shared_multi_output_impl.md §3.1: "do not keep an
   independently editable per-output program plus an optional sharing tag." *)
module Ref : sig
  type t = Grouped of region_group * Ordinal.t | Solo of Region_program.t

  val sources : t -> Expr.Source.Set.t
  val max_depth : t -> int
  val intrinsic_sources : t -> Expr.Source.t list
  val binders : t -> Expr.Reduce_var.t list
  val intrinsics : t -> int

  val project :
    max_size:int -> max_depth:int -> t -> (Region_program.t, error) Err.t
  (** [Solo p] re-[Region_program.check]s [p] and returns it unchanged -- [p]
      may have reached this [t] via [Region_program.with_output] (a raw record
      update with no check of its own) or a hand-built, untrusted
      [Kernel.t]/[Stage_program.t], so this cannot skip the check an
      always-freshly-[create]d program would not need.
      [Grouped (g,i) -> Region_group.project g i]. *)

  val check : max_size:int -> max_depth:int -> t -> (unit, error) Err.t
  (** [project], discarding the resulting program -- for a caller that only
      wants validation. *)

  val pixel_expression : t -> float Expr.Value.t option
  (** [Solo p -> Region_program.pixel_expression p]; [Grouped _ -> None] -- a
      grouped emitter is never a bare pixel expression; the shared executor is
      its only execution path (design record §5.4). *)

  val pp : t Fmt.t
  (** Never fabricates a projection: a [Grouped] value prints its group (shared
      locals once, every emitter's own raw output), the same
      no-fabricated-projection rule [Region_group.pp]/[Expr.Pp.scan_open]
      already follow. *)
end

module Run : sig
  type 'a t = Solo of 'a | Group of region_group * (Ordinal.t * 'a) list

  val duplicate_ordinal : 'a t -> Ordinal.t option
  (** [Some ordinal] iff [ordinal] appears more than once in a [Group] run's
      members -- never true for a run [runs] produces from either symbolic
      builder's own output (which always assigns one stage per ordinal), but
      [runs] itself does not reject a malformed/hand-built items list, so a
      caller executing a run must check this before trusting
      [List.assoc ordinal members] to name a unique member (design record §5.2,
      "selected ordinals cannot repeat"). [None] for [Solo]. *)
end

val runs : computation:('a -> Ref.t) -> 'a list -> 'a Run.t list
(** Partitions [items] (in order) into maximal runs of consecutive items whose
    [~computation] is [Ref.Grouped] into the SAME physical [Region_group.t]
    (compared with [==]), versus solitary [Ref.Solo]/differently-grouped items.
    A [Group] run's members are in encounter order -- ascending emitter ordinal
    for every real caller, since both symbolic builders emit one node's sibling
    stages/values contiguously that way, though this function itself assumes
    nothing about ordinal order. Shared by [Stage_program.ground] and
    [Kernel_eval.execute], so the two cannot detect sharing two different ways.
*)
