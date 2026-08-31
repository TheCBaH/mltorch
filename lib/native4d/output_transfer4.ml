(* How a value claim survives a Native4D op, per output. The dialect's own
   table, because §9.3 requires claim closure to use the DESTINATION dialect's:
   applying Native's to a Native4D graph would be asking the wrong question.

   Exhaustive with NO default arm, the same discipline and the same reason as
   Native's: a defaulting classifier silently mis-transfers the next op someone
   adds, and mis-transferring means a verifier asserting a guarantee the graph
   does not make. Adding an op means deciding its answer here — the site is
   listed in .ai/native4d_add_op.md. *)

open Op

(* Simpler than Native's for pooling: there is no argmax-pool, so no POOL op
   here answers [Discontinuous]. The pooled VALUE is branch-selecting but
   continuous — the branches agree at the boundary — and only a genuine argmax
   would not be. If [ArgMaxPool] is ever added (design §8 lists it as the
   smallest honest extension for live indices), it would be a second
   [Discontinuous] pooling arm alongside [To_copy]'s below. *)
let classify (op : Op.t) ~output:_ =
  match op with
  (* Data movement: every output element is COPIED from an input element with no
     arithmetic, so an incoming [Approximate] claim crosses unchanged rather than
     being downgraded to [Unverifiable] the way continuity would downgrade it.
     [Permute4]/[Reshape4] are the total case, a permutation; [Unbind] and
     [Slice4] are the partial one, each output a slice of the operand. Both qualify, because the claim is
     per-element and copying preserves it — the rule is not "the value multiset
     is unchanged", which holds only of the total case. See
     .ai/native_transform_design.md §8. *)
  | Add _ | Add_scalar _ | Adaptive_avg_pool2d _ | Avg_pool2d _ | Clamp _ ->
      Output_transfer.Continuous
  | Concat4 _ -> Output_transfer.Reindexing
  (* The pure-broadcast case, the same argument Native's own [Output_transfer]
     makes for [Expand]: every output reads exactly one input element, a
     broadcast axis read repeatedly and a non-broadcast axis read once each --
     no arithmetic on any of them. *)
  | Expand4 _ -> Output_transfer.Reindexing
  | Conv2d _ | Depthwise_conv2d _ | Div _ | Div_scalar _ | Gelu _
  | Batch_norm_no_stats _ | Group_norm4 _ | Grouped_conv2d _ | Hardsigmoid _
  | Hardswish _ | Hardtanh _ | Layer_norm _ | Leaky_relu _ | Max_keepdims _
  | Max_pool2d _ | Mean_keepdims _ | Mul _ | Mul_scalar _ | Pad4 _ ->
      Output_transfer.Continuous
  | Permute4 _ -> Output_transfer.Reindexing
  | Pow _ | Relu _ -> Output_transfer.Continuous
  (* Data movement, the same argument [Expand4] above makes: every output
     element is COPIED from an input element with no arithmetic -- taken
     modulo ([Repeat4]) or by floor-divided position ([RepeatInterleave4])
     rather than clamped to 0, but still a pure gather. *)
  | Repeat4 _ | RepeatInterleave4 _ -> Output_transfer.Reindexing
  | Reshape4 _ -> Output_transfer.Reindexing
  | Rms_norm _ -> Output_transfer.Continuous
  | Rsub_scalar _ -> Output_transfer.Continuous
  (* Data movement, the same argument as [Unbind]/[Slice4] above: every output
     element is COPIED from an input element with no arithmetic -- [Select4]
     drops the axis instead of narrowing or enumerating it, but the copy
     argument does not care which of the three a partial selection is. *)
  | Select4 _ -> Output_transfer.Reindexing
  (* Every output is copied from ONE of [self]/[src] with no arithmetic, and
     the choice is a structural fact about the output coordinate -- the same
     argument Native's own [Output_transfer] makes for [Select_scatter]. *)
  | Select_scatter4 _ -> Output_transfer.Reindexing
  | Sigmoid _ | Silu _ -> Output_transfer.Continuous
  | Slice4 _ -> Output_transfer.Reindexing
  | Split_with_sizes4 _ -> Output_transfer.Reindexing
  (* Data movement, the same argument as [Concat4]/[Select4] above: every
     output element is COPIED from an input element with no arithmetic --
     [Stack4] selects which operand by index rather than by within-segment
     offset, but the copy argument does not care which of the two a join is. *)
  | Stack4 _ -> Output_transfer.Reindexing
  | Sqrt _ | Sub _ | Sum_keepdims _ | Transposed_conv2d _ ->
      Output_transfer.Continuous
  (* Same per-op-not-per-target conservative answer as Native's own
     [Output_transfer]: the long/bool targets can each flip their result from
     an arbitrarily small input change, so the whole op answers
     [Discontinuous] rather than reading the payload to special-case the
     float target. *)
  | To_copy _ -> Output_transfer.Discontinuous
  | Unbind _ -> Output_transfer.Reindexing
  | Upsample_bilinear2d _ -> Output_transfer.Continuous
  (* A gather, not a blend, unlike [Upsample_bilinear2d] just above -- see
     [Output_transfer]'s own [Upsample_nearest2d] arm for the full argument. *)
  | Upsample_nearest2d _ -> Output_transfer.Reindexing
  | Vector_norm_keepdims _ -> Output_transfer.Continuous
  | Arange4 _ | Zeros4 _ -> Output_transfer.Continuous

module Transfer = Output_transfer.Make (struct
  type nonrec op = Op.t

  let operands = Op.operands
  let classify = classify
end)

let propagate = Transfer.propagate
