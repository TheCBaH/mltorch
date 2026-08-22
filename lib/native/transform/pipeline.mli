(* The canonical Native pass pipeline: one documented ordering, in the library
   rather than in a CLI. See .ai/native_transform_design.md §12h.

   It lived in bin/native_graph.ml until Native4D needed it. A dialect
   conversion is only ever meaningful on a canonical graph — the PT2 importer
   emits conv weights right-aligned from ATen's [out, in, kh, kw], which lands on
   D/H/W/C, so an unpipelined resnet18 has a non-unit D on every conv weight and
   is outside the four-axis dialect entirely. Leaving the definition of
   "canonical" in a CLI would have meant two callers agreeing by coincidence. *)

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
