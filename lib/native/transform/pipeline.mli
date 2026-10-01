(* The canonical Native pass pipeline: one documented ordering, in the library
   rather than in a CLI. See .ai/native_transform_design.md §12h.

   It lived in bin/native_graph.ml until Native4D needed it. A dialect
   conversion is only ever meaningful on a canonical graph — the PT2 importer
   emits conv weights right-aligned from ATen's [out, in, kh, kw], which lands on
   D/H/W/C, so an unpipelined resnet18 has a non-unit D on every conv weight and
   is outside the four-axis dialect entirely. Leaving the definition of
   "canonical" in a CLI would have meant two callers agreeing by coincidence. *)

(* The ordered stages [canonical_with_trace] composes into one [Pass.sequence]:
   reshape_to_permute, relayout (an outer fixpoint of nine inner fixpoints),
   prune, the first fold_const fixpoint, fold_batch_norm, the second fold_const
   fixpoint. Exposed so a benchmark or test can run and time each stage
   separately (chaining the state through [Pass.run_all], one stage at a time)
   while still agreeing with [canonical]/[canonical_with_trace]'s own
   single-composite execution — [canonical_with_trace] is built FROM this list,
   so there is one authoritative pass-order definition, not two that could
   drift apart. Separately executing a stage is not guaranteed to reproduce
   [canonical_with_trace]'s own overhead exactly (verification/trace scope
   differ per call); compare outputs and maps against [canonical_with_trace]
   itself before trusting a staged measurement. *)
val canonical_stages :
  on_materialized_fold:(Fold_const.Trace.event -> unit) -> Pass.t list

(* [fold] no longer controls any pass. Const-SSA folding is part of the one
   canonical graph, whether captures have materialized payloads or not.

   Idempotent: running the result twice produces the same graph as running it
   once. *)
val canonical_with_trace :
  on_materialized_fold:(Fold_const.Trace.event -> unit) -> fold:bool -> Pass.t

val canonical : fold:bool -> Pass.t
(** [fold] is retained for source compatibility and has no effect on the
    canonical graph. Payloads must be materialized explicitly after the symbolic
    transform. *)
