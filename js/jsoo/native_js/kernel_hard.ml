(* Checked-in override of lib/native/kernel_hard.ml, following the
   convention js/melange/walk_core/walk_core.ml established: a hand-written
   file with the SAME module name as one this mirror's copy_files stanzas
   deliberately do not copy (see js/jsoo/native_js/dune's root stanza),
   rather than a fork of the origin file (tail-call plan, Stage 7; see
   .ai/). [Kernel_hard_shared] is still copied from lib/native verbatim and
   [include]d here unchanged -- only [eval_depth] differs.

   Measured against the full modules/devcontainer.pytorch-image-models
   corpus (100 models, payload-free, via bin/pt2_json_model_support.exe):
   with Kernel.create's [Eval_too_deep] guard temporarily disabled, the
   corpus-wide maximum combined per-value eval depth (the same [e] kernel.ml
   computes) is 6238, reached by csatv2 (mobilenetv5_base is next at 5313;
   every other model in the corpus is under 4800). 12288 is ~2x that
   measured maximum -- the same headroom multiple lib/native/kernel_hard.ml's
   own comment used over resnet18's ~770 to reach native's original 1536,
   since re-measured down to 1280 -- and it
   is a small, deliberate fraction of the 20,000-deep smoke case Stage 6
   already proved the installed [eval_hybrid ~cutoff:50] driver survives on
   both JS backends, so this is not the adversarial ceiling, just a
   production-shaped one with room for the corpus to grow. Native's own
   [Hard.eval_depth] (in lib/native/kernel_hard.ml) is unaffected -- native's
   [Eval.value] is still ordinary recursion, genuinely stack-bound, and out
   of this override's scope. *)

include Kernel_hard_shared

let eval_depth = 12288
