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
     two SOURCE members. *)
  type t = Dst of Tensor_id.t | Src of Tensor_id.t

  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit
end

module Budget : sig
  (* Counted, never timed, so a verdict is deterministic and a golden is
     stable. [max_coords] is checked against [Vec6.numel] BEFORE any work,
     which is what stops this being pointed at a real model's activations. *)
  type t = { max_coords : int; max_nodes : int; max_rounds : int }

  val cumulative : t (* composed maps: deeper frontier, more rounds *)
  val default : t (* one rewrite step *)
  val pp : Format.formatter -> t -> unit
end

module Coverage : sig
  (* [Not_applicable] is the honest answer for a cluster that returns before any
     coordinate is examined. Keeping it a constructor rather than an option
     forces every early exit to say which it is. *)
  type t = Exhaustive | Not_applicable

  val pp : Format.formatter -> t -> unit
end

module Strength : sig
  type proof = Structural

  (* Whether a proof of this strength discharges a claim of that relation. *)
  val proves : proof -> Correspondence.relation -> bool
  val pp_proof : Format.formatter -> proof -> unit
end

module Refutation : sig
  (* Tagged rather than a fixed src/dst/coord triple: a shape mismatch has no
     meaningful common coordinate, and either side of a comparison may be a
     source member. *)
  type t =
    | Shape of {
        lhs : Member.t;
        lhs_shape : Vec6.shape;
        rhs : Member.t;
        rhs_shape : Vec6.shape;
      }

  val pp : Format.formatter -> t -> unit
end

module Unproved : sig
  type t =
    | Eval of Ground_eval.error
    | Exhausted of { coord : Vec6.coord; lhs : Member.t; rhs : Member.t }
      (* frontier fully expanded, terms still differ *)
    | Max_nodes of int
    | Max_rounds
      (* cells still expandable: the frontier never reached the inputs *)
    | Out_of_bounds of Ground_expr.Cell.t
    | Too_large of int
    | Unsupported_format of { blocked : Ground_expr.Cell.t; member : Member.t }
    | Unsupported_relation of Correspondence.relation

  val pp : Format.formatter -> t -> unit
end

module Verdict : sig
  type t =
    | Proved of Strength.proof
    | Refuted of Refutation.t
    | Unproved of Unproved.t
    | Vacuous (* a creation or a deletion: the cluster claims nothing *)

  val pp : Format.formatter -> t -> unit
end

module Outcome : sig
  type t = { coverage : Coverage.t; verdict : Verdict.t }

  val pp : Format.formatter -> t -> unit
end

module Report : sig
  type t = { clusters : (Correspondence.Cluster.t * Outcome.t) list }

  (* Strict: every outcome is [Proved] with [Exhaustive] coverage, or [Vacuous]
     (whose coverage is [Not_applicable] — it claims nothing, so there is
     nothing to cover). *)
  val proved : t -> bool
  val refuted : t -> bool
  val summary : t -> string
  val pp : Format.formatter -> t -> unit
  val pp_verdicts : Format.formatter -> t -> unit
end

type error = [ Graph_map.error | `Missing_signature of Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit

(* [Graph_map.validate] runs first and is not optional: the phantom tags cannot
   tie a map to two PARTICULAR graphs, so a well-typed map may name ids that
   exist in neither. *)
val run :
  ?budget:Budget.t ->
  ('a, 'b) Graph_map.t ->
  src:graph ->
  dst:graph ->
  (Report.t, error) Core.result

(* Works unchanged on a composed pipeline step, which is what makes cumulative
   verification free at the API level. *)
val step :
  ?budget:Budget.t ->
  'v Rewrite.t ->
  'v Rewrite.step ->
  (Report.t, error) Core.result
