(* See pipeline.mli. Moved here from bin/native_graph.ml, comments and all,
   because Native4D needs the same definition of "canonical". *)

(* The relayout group. The lowering emits an inverse permute pair at every op
   boundary (.ai/native_transform_design.md §1) and cancelling them is the whole
   point. [Sink_permute] (§12d) catches the case where an elementwise op (Relu,
   Add, ...) sits between the pair — Chain/Trim only cancel ADJACENT permutes, so
   sinking has to make them adjacent first; a second Chain/Trim round then
   collapses what it exposes. *)
(* [Sink_permute_mean] transports a permutation through a [keepdim=true] [Mean]
   (§12f) rather than sinking it unchanged — [Mean] is intentionally absent from
   [Sink_permute]'s allowlist, since reducing the same axis names after removing
   its input permutation would reduce the wrong dimensions. It runs right after
   the initial chain/trim cleanup: transport can expose an adjacent permute pair
   on either side, which [Sink_permute] and a later [Chain_permute]/
   [Trim_permute] round then pick up. *)
(* [Reuse_permute] and [Bypass_permute] complement [Sink_permute]: see
   .ai/native_layout_reuse_plan.md. [Reuse_permute] turns a mixed elementwise
   operand set uniform by reusing an alternate-layout edge the graph already
   computes, which [Sink_permute] can then move downstream; [Bypass_permute] then
   removes the inverse consumers that move exposes, without requiring the whole
   run to be interior the way [Trim_permute] does. These unlock one another —
   bypassing one residual block's inverse permutes can make the next block's skip
   edge interior, exposing another sink/trim opportunity — so the whole group
   runs under one outer fixed point rather than a single pass over each. *)
let relayout =
  Pass.fixpoint
    (Pass.sequence ~name:"relayout"
       [
         Pass.fixpoint Chain_permute.pass;
         Pass.fixpoint Trim_permute.pass;
         Pass.fixpoint Sink_permute_mean.pass;
         Pass.fixpoint Sink_permute.pass;
         Pass.fixpoint Reuse_permute.pass;
         Pass.fixpoint Sink_permute.pass;
         Pass.fixpoint Bypass_permute.pass;
         Pass.fixpoint Chain_permute.pass;
         Pass.fixpoint Trim_permute.pass;
       ])

(* Pruning, in the one order that works. [Dce] first, to remove the [Discard]
   sink that is holding a max-pool index edge in use; then
   [Drop_pool_indices], which can only narrow the op once that edge is
   genuinely dead; then [Dce] again, because narrowing the op can strand
   whatever the index edge fed. Neither is wrapped in [Pass.fixpoint]: [Dce]'s
   predicate is global, so one sweep removes everything unreachable, and
   [Drop_pool_indices] cannot create new work for itself. *)
let prune =
  Pass.sequence ~name:"prune" [ Dce.pass; Drop_pool_indices.pass; Dce.pass ]

(* Order is load-bearing. The importer emits every conv weight behind a relayout
   permute, so the weight is a NODE OUTPUT until folding materialises it — and
   batch-norm folding requires constant parameters. So constant folding runs
   first to make the weights constant, then the batch-norm fold, then constant
   folding again to collapse the parameter arithmetic that fold emits. Without
   the first pass the batch-norm fold matches nothing at all.

   [fold:false] still folds batch norm. It is not a prefix of [fold:true] — a
   structural caller with no payloads bound gets the same batch-norm treatment,
   just without the [Fold_const] rounds that would decline every node anyway. *)
let canonical ~fold =
  Pass.sequence ~name:"canonical"
    ([ Reshape_to_permute.pass; relayout; prune ]
    @
    if fold then
      [
        Pass.fixpoint Fold_const.pass;
        Fold_batch_norm.pass;
        Pass.fixpoint Fold_const.pass;
      ]
    else [ Fold_batch_norm.pass ])
