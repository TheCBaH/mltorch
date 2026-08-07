(* Symbolic verification results, placed in the rendered hierarchy and counted.

   {b Why the exporter places them itself.} [Map_verify.Group_path] appends a
   group's LABEL alone, while the importer labels every group with the PT2
   node's target — so sibling [convolution] groups collapse into one path, and
   that path cannot key the ID-qualified namespace [Me_build] emits. And
   [Map_verify]'s own attribution takes the FIRST destination edge that
   resolves, which is set order rather than a common group.

   So placement is recomputed here, over the namespaces the projection actually
   used: a cluster's destination members map to their producing nodes,
   producerless members are IGNORED (a graph input or a constant belongs to no
   group), and the placement is the longest common namespace prefix of the rest,
   falling back to the root when none remain. That is
   [Graph_view.common_group]'s rule expressed in namespace space, which is the
   space the answer has to be in.

   {b Where a group fact can live at all.} [groupNodeAttributes] is the only
   part of the wire format keyed by namespace rather than by node id, so it is
   where a per-group rollup belongs. Per-node verdicts go through the
   node-keyed [Node_data_set] instead, and the two say different things: the
   node datum is one node's weakest outcome, the group attribute is its
   subtree's tally.

   See .ai/model_explorer_design.md. *)

type error = [ Me_ids.error | `Over_limit of string * int | Pass.count_error ]
(** [Me_ids.error] because a placement is a NAMESPACE, spelled by the same
    encoder the projection uses. *)

val pp_error : Format.formatter -> [< error ] -> unit

val summary :
  Map_verify.Report.t -> (Pass.Outcome_counts.t, [> error ]) Core.result
(** The [Feature Verification] payload: counts over the COMPOSED whole-pipeline
    report, whose clusters are in the final graph's ids. *)

val audit_status : Pass.Audit_log.t -> Me_session.Capability.Pass_audit_status.t
(** The [Feature Pass_audits] payload. Separate from {!summary} because the log
    is bounded and the composed report is not: mapping truncation to one
    [Verification -> Unavailable Over_limit] would hide a result that is still
    available, while [Available] alone would conceal that the detail is
    incomplete. Two keys say both things. *)

val placement :
  namespace_of:(Graph_ir.Node_id.t -> string) ->
  producer:(Graph_ir.Tensor_id.t -> Graph_ir.Node_id.t option) ->
  Map_verify.Entry.t ->
  string
(** Where one cluster sits in the rendered hierarchy — the longest common
    namespace prefix of its destination members' producers, [""] (the root) when
    none of them has one. Exposed so the rule can be tested on the two cases a
    real model never happens to show side by side: a cluster inside one group,
    and a cluster spanning two. *)

val by_namespace :
  limits:Me_limits.Limits.t ->
  Graph_ir.graph ->
  Map_verify.Report.t ->
  ((string * (string * string) list) list, [> error ]) Core.result
(** The rollup, ready for {!Me_build.Make.graph}'s [group_attrs]: one entry per
    namespace that holds a cluster, its attributes being the outcome counts.
    Ordered by namespace, and the counts within an entry by
    [Outcome_counts.bindings]'s canonical order, so two runs agree byte for
    byte. *)

val node_data :
  limits:Me_limits.Limits.t ->
  graph:string ->
  Graph_ir.graph ->
  Map_verify.Report.t ->
  (Me_session.Node_data_set.t, [> error ]) Core.result
(** Per node, [Map_verify.Outcome.join] over its outputs, since a node is only
    as verified as its least-verified result. The value is
    [Map_verify.Verdict.rank] and the label is [Outcome.label] — the one place
    the ["<verdict> [sampled n]"] spelling is written.

    The rank is HOW WEAK a verdict is, so a gradient over it runs proved (1, 2)
    through tested (3, 4) to unproved (5) and refuted (6). [Vacuous] is 0, BELOW
    proved: a creation or a deletion claims nothing, and the ordering exists so
    that a claimless cluster cannot mask a real verdict when a node's outputs
    are joined — it is not a statement that a vacuous cluster is better verified
    than a proved one.

    [Outcome.join], not two independent joins: an [Unproved Too_large] examined
    nothing, so its coverage is [Not_applicable], and joining the fields
    separately would let it come out marked [sampled n] borrowed from a sibling
    output. *)
