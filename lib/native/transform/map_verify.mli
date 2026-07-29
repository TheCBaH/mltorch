(* Checking that a [Graph_map.t]'s value claims actually hold, WITHOUT payload
   data for the graph inputs — so a passing result is a statement about every
   input, not about one sample.

   Every value cluster [Graph_map.clusters_over] synthesises is checked
   independently: its members' per-pixel expressions are ground (indices
   evaluated at a concrete coordinate, payloads left as free cells) and
   compared structurally. Corresponding graph INPUTS are the hypothesis — they
   are fed the same data — and everything else is an obligation.

   Three properties shape the whole thing, and are why the verdicts are not
   simply pass/fail:

   - Proof is sound under over-approximation; refutation is not. Two terms equal
     as functions of their free cells are equal under every assignment,
     including the constrained ones a truncated frontier admits. A DISAGREEMENT
     between them proves nothing, because internal cells are constrained by
     their producers. So a failed comparison is [Unproved], never [Refuted]
     (Stage 2 adds a probe that can produce a real counterexample).
   - [Identical] is about bits, so comparison is bitwise throughout and [Round]
     nodes — the f32 materialization every stage performs — are part of the
     term rather than an idealisation to be erased.
   - A cluster is a pair of SETS. Comparing one representative per side would
     say nothing about the others, which for [{t0,t1} -> {t0}] is exactly the
     trimmed edge the map claims is identical.

   See .ai/native_transform_verify.md. *)

open Graph_ir

module Member : sig
  (* Side-tagged, because [{src=t0} <-> {dst=t0}] holds two distinct members
     that share a raw id, and because a comparison can legitimately run between
     two SOURCE members.

     Only the ERASED form is public. During checking a member carries its graph
     version and there is no [id] accessor at all — reaching an id means going
     through the eliminator that pairs a member with its own side, which is the
     mistake fixed by hand at map_verify.ml:316. Reports drop the version: the
     hierarchy below is unparameterized and escapes into [Pass.outcome] and the
     interpreter's result record, and a report is read, not computed with. *)
  module Erased : sig
    type t = { id : Tensor_id.t; side : [ `Dst | `Src ] }

    val compare : t -> t -> int
    val pp : Format.formatter -> t -> unit
  end
end

module Budget : sig
  (* Counted, never timed, so a verdict is deterministic and a golden is
     stable. [max_coords] is checked against [Vec6.numel] BEFORE any work,
     which is what stops this being pointed at a real model's activations. *)
  type t = {
    max_coords : int;
    max_nodes : int;
    max_rounds : int;
    sample : int option;
  }

  val cumulative : t (* composed maps: deeper frontier, more rounds *)
  val default : t (* one rewrite step *)

  (* For a real model. [max_coords] still refuses whole activation tensors
     outright, so what this widens is the constant-shaped clusters — folded
     weights and biases, which is where fold_const and fold_batch_norm act. *)
  val release : t
  val pp : Format.formatter -> t -> unit
end

(* A named point on the cost/coverage curve, so a caller can pick how hard to
   look without knowing what a coord or a round is. Effort drives the probe
   count as well as the budget: a deeper frontier is worth more draws. *)
module Effort : sig
  type t = Quick | Standard | Thorough

  val all : t list
  val budget : t -> Budget.t
  val of_string : string -> (t, [> `Unknown_effort of string ]) Stdlib.result
  val probe : t -> int
  val to_string : t -> string
  val pp : Format.formatter -> t -> unit
end

module Coverage : sig
  (* [Not_applicable] is the honest answer for a cluster that returns before any
     coordinate is examined. Keeping it a constructor rather than an option
     forces every early exit to say which it is. *)
  type t = Exhaustive | Not_applicable | Sampled of int

  val pp : Format.formatter -> t -> unit
end

module Strength : sig
  type proof =
    | Constants
      (* structural once the graph's own constant payloads are substituted:
           holds for every INPUT, for these constants. Strictly weaker than
           [Structural], which is why it is only attempted when that fails. *)
    | Structural (* structural over free cells: holds for every payload *)

  type test =
    | Agrees of float
      (* the two sides have the same polynomial in their free cells, with
           coefficients within this tolerance: evidence that they differ only by
           rounding, and never a proof — coefficients [1] and [1 + eps] pass a
           tolerance while being neither exactly equal nor boundedly close for
           an unbounded free cell *)
    | Disagrees of Ground_expr.Valuation.t
  (* a probe found an assignment the two sides evaluate differently, but
           the claim is not [Identical], so rounding alone could explain it and
           this is evidence rather than a refutation *)

  (* No [proves : proof -> relation -> bool]. There used to be one, answering
     [false] for [Unverifiable] and [true] for everything else; nothing ever
     called it, and its answer is now wrong — structural equality discharges an
     [Unverifiable] cluster too, because it is not a claim about values at all.
     A predicate no caller consults cannot go stale visibly, which is what makes
     dead API that states the wrong rule worse than none. *)
  val pp_proof : Format.formatter -> proof -> unit
  val pp_test : Format.formatter -> test -> unit
end

module Refutation : sig
  (* Tagged rather than a fixed src/dst/coord triple: a shape mismatch has no
     meaningful common coordinate, and either side of a comparison may be a
     source member. *)
  type t =
    | Shape of {
        lhs : Member.Erased.t;
        lhs_shape : Vec6.shape;
        rhs : Member.Erased.t;
        rhs_shape : Vec6.shape;
      }
    | Value of {
        coord : Vec6.coord;
        lhs : Member.Erased.t;
        rhs : Member.Erased.t;
        valuation : Ground_expr.Valuation.t;
      }
  (* A reproducible counterexample: replaying [valuation] through
           [Ground_expr.eval] on both terms gives different bits. Only ever
           built for an [Identical] claim — see the note on [run]. *)

  val pp : Format.formatter -> t -> unit
end

module Unproved : sig
  type t =
    | Eval of Ground_eval.error
    | Exhausted of {
        coord : Vec6.coord;
        lhs : Member.Erased.t;
        rhs : Member.Erased.t;
      }
      (* frontier fully expanded, terms still differ *)
    | Max_nodes of int
    | Max_rounds
      (* cells still expandable: the frontier never reached the inputs *)
    | Out_of_bounds of Ground_expr.Cell.t
    | Too_large of int
    | Unbound_constant of Ground_expr.Cell.t
      (* a model constant whose payload nobody supplied. Not [Refuted]: the two
           graphs' constants are distinct cells because sigma covers user data
           only, so a probe would "separate" a pair that may hold identical
           bytes *)
    | Unsupported_format of {
        blocked : Ground_expr.Cell.t;
        member : Member.Erased.t;
      }
    | Unsupported_relation of Correspondence.relation

  val pp : Format.formatter -> t -> unit
end

module Verdict : sig
  type t =
    | Proved of Strength.proof
    | Refuted of Refutation.t
    | Tested of Strength.test (* evidence, never a proof *)
    | Unproved of Unproved.t
    | Vacuous (* a creation or a deletion: the cluster claims nothing *)

  val pp : Format.formatter -> t -> unit

  (* The WEAKER of two verdicts about the same cluster, since a cluster is only
     as verified as its least-verified coordinate — so a coordinate that needed
     bound constants is not overwritten by a later one that proved structurally.
     [Refuted] dominates: one counterexample settles the cluster however many
     other coordinates agreed. Ties keep the earlier verdict, so a report names
     the first coordinate that failed rather than the last. *)
  val join : t -> t -> t

  (* How weak a verdict is; higher wins a join. Exposed so a caller aggregating
     OUTCOMES can pick the whole weaker one — verdict and coverage together —
     rather than joining the two independently and pairing a verdict with
     coverage from a different outcome. *)
  val rank : t -> int

  (* Outcome and reason, with ids, coordinates and valuations dropped, so
     verdicts can be counted. The payloads belong in [Report.pp_verdicts]. *)
  val label : t -> string
end

module Outcome : sig
  type t = { coverage : Coverage.t; verdict : Verdict.t }

  (* The weaker outcome ENTIRE. Joining verdict and coverage independently pairs
     a verdict with coverage from a different outcome — an [Unproved Too_large],
     whose coverage is [Not_applicable] because nothing was examined, could come
     out marked [sampled n] from a sibling edge. *)
  val join : t -> t -> t
  val pp : Format.formatter -> t -> unit
end

(* Where a cluster sits in the destination's structural hierarchy. The IR's
   groups are what a reader recognises — "layer1.0", "features.3" — so a report
   over a real model is only legible when its clusters are attributed to them.
   The root is the empty path. *)
module Group_path : sig
  type t = string list

  val compare : t -> t -> int
  val to_string : t -> string
  val pp : Format.formatter -> t -> unit
end

module Entry : sig
  type t = {
    (* Erased, deliberately: a report escapes into [Pass.outcome] and the
       interpreter's result record, and parameterising the whole hierarchy by
       ['src] and ['dst] would buy nothing a reader can use. *)
    cluster : Correspondence.Cluster.Erased.t;
    group : Group_path.t;
    outcome : Outcome.t;
  }
end

(* Counts by outcome label, deterministically ordered. What a summary is for:
   which clusters came back proved, and for the rest, WHY. A sampled verdict
   gets its own label rather than being folded in with the exhaustive ones —
   counting a sampled proof as "proved" is exactly the overstatement coverage
   exists to prevent. *)
module Tally : sig
  type t

  val bindings : t -> (string * int) list
  val of_entries : Entry.t list -> t
  val total : t -> int
  val pp : Format.formatter -> t -> unit
end

module Report : sig
  type t = { entries : Entry.t list }

  (* Strict: every outcome is [Proved] with [Exhaustive] coverage, or [Vacuous]
     (whose coverage is [Not_applicable] — it claims nothing, so there is
     nothing to cover). *)
  val proved : t -> bool
  val refuted : t -> bool
  val by_group : t -> (Group_path.t * Tally.t) list
  val tally : t -> Tally.t
  val summary : t -> string
  val pp : Format.formatter -> t -> unit
  val pp_groups : Format.formatter -> t -> unit
  val pp_summary : Format.formatter -> t -> unit
  val pp_verdicts : Format.formatter -> t -> unit
end

module Policy : sig
  (* What a caller treats as failure. [Reject_refuted] is the release bar: only
     an actual counterexample stops the build, so [Unproved] from a budget or a
     tier that cannot reach a pass is tolerated. [Require_proved] is the
     development bar, where an unproved rewrite is a rewrite nobody has
     justified. *)
  type t = Reject_refuted | Require_proved

  val accepts : t -> Report.t -> bool
  val pp : Format.formatter -> t -> unit
end

type error = [ Graph_map.error | `Missing_signature of Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit

(* Endpoints are checked by [Graph_map.create], so nothing here re-validates
   them. Claim closure is different: [Graph_map.compose] takes no snapshots and
   so cannot re-establish it, and a cumulative run is handed exactly such a
   composed map — so [check_claim_closure] runs first and is not optional.

   [probe] is the number of valuations tried when the terms differ, and it is a
   REFUTATION engine, not a weak proof: no number of agreeing draws produces a
   [Proved]. Two conditions bound it.

   It runs only once the frontier has reached the graph inputs on both sides
   ([Ground_eval.expandable] false). Cells left at a truncated frontier are
   internal stage results constrained by their producers, so assigning them
   independently could manufacture a counterexample no input can realise.

   And it formally refutes only [Identical]. [Equivalent] permits rounding to
   differ, so a disagreement between rounded terms cannot contradict it, and
   [Approximate] would need declared input ranges and an error model that
   [Precision.Set.t] does not carry. For those, a disagreement is reported as
   [Tested (Disagrees _)].

   Constants are PER-GRAPH: [Rewrite.apply] filters payloads to live
   destination ids, so one a fold consumed and deleted exists only on the
   source side. *)
(* The tolerance [Coeff_form] runs at when no caller supplies one. 1e-5, the
   same number fold_batch_norm_test.ml's output-level check uses — shared for
   familiarity, NOT because the two are the same bar. Per-coefficient and
   whole-tensor agreement are incomparable in general, which is why the verifier
   is adopted alongside that check rather than in place of it. The tolerance
   actually used is recorded in [Tested (Agrees _)], so a golden always shows
   which bar produced the verdict. *)
val default_coefficient_tolerance : float

val run :
  ?budget:Budget.t ->
  ?coefficient_tolerance:float ->
  ?probe:int ->
  ?src_constants:Tensor.packed Tensor_id.Map.t ->
  ?dst_constants:Tensor.packed Tensor_id.Map.t ->
  ('src, 'dst) Graph_map.t ->
  src:'src Snapshot.t ->
  dst:'dst Snapshot.t ->
  (Report.t, error) Core.result

(* Reads [Rewrite.constants] from each state separately, for the reason above.
   Works unchanged on a composed pipeline step, which is what makes cumulative
   verification free at the API level. *)
val step :
  ?budget:Budget.t ->
  ?coefficient_tolerance:float ->
  ?probe:int ->
  'v Rewrite.t ->
  'v Rewrite.step ->
  (Report.t, error) Core.result
