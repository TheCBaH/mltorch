(* The whole-graph symbolic form: a stage DAG (one stage per node) layered over the
   unchanged [Expr]. Each stage's [body] is the node's per-pixel expression, whose
   [Load]s carry the producing edges' signatures — so a downstream stage references
   an upstream stage purely through the signature it loads (the "source = Input |
   Stage" distinction is recovered by membership, not an [Expr] variant). [ground]
   evaluates the stages in topo order, feeding each result into the next stage's
   binding, which is what makes symbolic execution extend through the whole graph.
   See .ai/native_symbolic_language.md and .ai/native_graph_design.md. *)

open Graph_ir

module Stage : sig
  type t = {
    id : Tensor_id.t; (* the producing edge id (this stage's identity) *)
    sg : Tensor_sig.t; (* the stage's output signature *)
    computation : Region_group.Ref.t;
        (* The single structural computation: either a standalone program
           (a Pixel-authored stage embeds its original expression
           mechanically with [Region_program.pixel]) or a reference to one
           ordinal of a group several sibling stages share (project step
           19). *)
  }

  val computation : t -> Region_group.Ref.t
  (** The stage's single structural computation. A Pixel stage retains its
      original expression object inside [Region_program.pixel]. *)

  val sources : t -> Expr.Source.Set.t

  type pixel_body_error = [ Region_program.error | `Not_a_pixel_program ]

  val pp_pixel_body_error : Format.formatter -> [< pixel_body_error ] -> unit

  val pixel_body :
    max_size:int ->
    max_depth:int ->
    scan_limits:Expr.Scan_limits.t ->
    t ->
    (float Expr.Value.t, pixel_body_error) Err.t
  (** The symbolic Pixel view used only by consumers that cannot yet traverse
      Region locals. A group emitter is never a bare pixel expression -- the
      shared executor is its only execution path (design record §5.4) -- so this
      fails with [`Not_a_pixel_program] on a [Region_group.Ref.Grouped] stage
      rather than attempting to specialize it. *)

  val check :
    max_size:int -> max_depth:int -> t -> (unit, Region_group.error) Err.t
end

(* The int64 twin of [Stage.t], restricted to a bare pixel body -- no
   [Region_group.Ref]/locals -- mirroring [Kernel.Value_i64.t]'s own
   deliberately narrow "Pixel-only" shape (see the implementation tracker's
   D09/D10 notes). Added ALONGSIDE [Stage.t], never folded into it: a stage's
   [computation] field cannot represent an [int64 Expr.Value.t] body without
   widening [Region_group.Ref.t]/[Region_program.t] themselves, which is out
   of this slice's scope. [Stage_program.ground] and every OTHER existing
   consumer of [t.stages] (grounding/verification/export) does not read
   [stages_i64]; only [Kernel_adapt] does. *)
module Stage_i64 : sig
  type t = { id : Tensor_id.t; sg : Tensor_sig.t; pixel : int64 Expr.Value.t }
end

type t = {
  inputs : (Tensor_id.t * Tensor_sig.t) list;
      (* graph inputs (Load "sources") *)
  input_kinds : Input.kind Tensor_id.Map.t;
      (* same source classification as the originating graph *)
  consts : (Tensor_sig.t * float) list; (* synthetic constant-filled operands *)
  stages : Stage.t list; (* topo-ordered *)
  stages_i64 : Stage_i64.t list;
      (* int64-carrier pixel-only stages, alongside [stages]; see
         [Stage_i64]'s own doc comment *)
  outputs : Tensor_id.t list;
}

val pp : Format.formatter -> t -> unit

type error =
  [ Region_group.error
  | Region_program.error
  | Region_eval.error
  | `Duplicate_group_ordinal of int ]

val pp_error : Format.formatter -> [< error ] -> unit

(* Chain-ground the stages: [bind] supplies a tensor for each graph input id; the
   result maps every stage's edge id to its grounded tensor (the intermediates),
   so the graph outputs are looked up by id. Int64 stages ([stages_i64]) are
   grounded too, on demand and memoised, so an int64 stage may read a float
   stage and a float stage may read an int64 one; their tensors appear in the
   result alongside the float ones. Stages are grouped into maximal
   runs of consecutive [Region_group.Ref.Grouped] members sharing one
   physically-identical [Region_group.t] (project step 19's Section C
   milestone 2); each such run shares ONE evaluation of its group's locals per
   canonical key across every member ([Region_execution.lower_group]/
   [materialize_group]), installing every member's tensor atomically before
   the fold advances. An ordinary [Solo] stage, or a run of length one, takes
   the unchanged single-program path ([Region_program.check]/[preflight] via
   [Region_execution.lower]). Every run is preflighted before the first one is
   materialized. [?limits] defaults to [Kernel.Limits.default]. [?region_counters],
   if given, maps a stage's edge id to the (possibly shared) counters record its
   caller bound it to -- mirroring [Eval_direct.run]'s own convention -- and a
   [Group_run] resolves ONE counters record from its first member, so shared
   work is charged exactly once per group, never once per sibling. *)
val ground :
  ?limits:Kernel.Limits.t ->
  ?region_counters:Region_execution.counters Tensor_id.Map.t ->
  t ->
  bind:(Tensor_id.t -> Tensor.packed) ->
  (Tensor.packed Tensor_id.Map.t, error) Err.t
