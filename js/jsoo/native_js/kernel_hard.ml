(* Checked-in override of lib/native/kernel_hard.ml, following the
   convention js/melange/walk_core/walk_core.ml established: a hand-written
   file with the SAME module name as one this mirror's copy_files stanzas
   deliberately do not copy (see js/jsoo/native_js/dune's root stanza),
   rather than a fork of the origin file (tail-call plan, Stage 7; see
   .ai/). [Kernel_hard_shared] is still copied from lib/native verbatim and
   [include]d here unchanged -- only [eval_depth] differs.

   [eval_depth] is checked by [Kernel_eval] where a recursion through virtual
   edges starts, NOT by [Kernel.create], and is NOT a stack bound: the
   buffer-based [Kernel_eval.run] never recurses through producers, and the
   recursive [value_at] path is bounded at runtime by [eval_stack_budget]
   below, which counts what a producer transition actually costs. Do not read
   this ceiling as the depth the evaluator survives.

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
   both JS backends for ONE [Expr.Eval.value] call, so this is not the
   adversarial ceiling, just a production-shaped one with room for the corpus
   to grow. Native's own
   [Hard.eval_depth] (in lib/native/kernel_hard.ml) is unaffected -- native's
   [Eval.value] is still ordinary recursion, genuinely stack-bound, and out
   of this override's scope. *)

include Kernel_hard_shared

let eval_depth = 12288

(* [Kernel_eval.value_at] nests real JS stack per producer transition: the
   [Env.load] callback is synchronous, so the hybrid evaluator's heap-resident
   machine only helps INSIDE one [Expr.Eval.value] call. A transition costs
   [transition_base] levels of fixed frames plus the body's direct segment,
   which is capped by the hybrid [cutoff] ([Expr.Eval.cutoff], 50) because the
   rest of the body runs on the heap.

   Measured under node with a chain of [n] producers, each a body of depth [d]
   (jsoo, both guards lifted; unit = [transition_cost] per transition, [d + 2]
   being the converted body's depth): the frontier is 1134..2002 for every [d]
   from 1 to 250, lowest where the direct segment saturates at the cutoff. 672
   is 96 * 7, so it keeps the shared [eval_recursion] (96) boundary for the
   depth-3 bodies that figure was measured with, and sits at ~0.6 of the
   lowest frontier. The frontier is unstable near the edge; do not tune this
   to it. *)
let transition_base = 4
let direct_cutoff = 50
let eval_stack_budget = 672
let transition_cost ~body_depth = transition_base + min body_depth direct_cutoff
