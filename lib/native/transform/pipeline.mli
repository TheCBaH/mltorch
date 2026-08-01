(* The canonical Native pass pipeline: one documented ordering, in the library
   rather than in a CLI. See .ai/native_transform_design.md §12h.

   It lived in bin/native_graph.ml until Native4D needed it. A dialect
   conversion is only ever meaningful on a canonical graph — the PT2 importer
   emits conv weights right-aligned from ATen's [out, in, kh, kw], which lands on
   D/H/W/C, so an unpipelined resnet18 has a non-unit D on every conv weight and
   is outside the four-axis dialect entirely. Leaving the definition of
   "canonical" in a CLI would have meant two callers agreeing by coincidence. *)

(* [fold] is not "run the extra passes at the end": both branches run
   [Fold_batch_norm], and they differ only in the [Fold_const] rounds around it.
   Structural callers with no payloads bound pass [~fold:false], since with
   nothing to evaluate [Fold_const] would decline every node and including it
   would suggest otherwise.

   Idempotent: running the result twice produces the same graph as running it
   once, on either branch. *)
val canonical : fold:bool -> Pass.t
