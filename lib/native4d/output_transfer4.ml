(* How a value claim survives a Native4D op, per output. The dialect's own
   table, because §9.3 requires claim closure to use the DESTINATION dialect's:
   applying Native's to a Native4D graph would be asking the wrong question.

   Exhaustive with NO default arm, the same discipline and the same reason as
   Native's: a defaulting classifier silently mis-transfers the next op someone
   adds, and mis-transferring means a verifier asserting a guarantee the graph
   does not make. Adding an op means deciding its answer here — the site is
   listed in .ai/native4d_add_op.md. *)

open Op

(* Simpler than Native's, and the simplification is the dialect's doing: there
   is no argmax-pool, so nothing here is [Discontinuous]. The pooled VALUE is
   branch-selecting but continuous — the branches agree at the boundary — and
   only a genuine argmax would not be. If [ArgMaxPool] is ever added (design §8
   lists it as the smallest honest extension for live indices), it is the one op
   whose index output must answer [Discontinuous]. *)
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
  | Permute4 _ | Reshape4 _ | Slice4 _ | Unbind _ -> Output_transfer.Reindexing
  | Add _ | Add_scalar _ | Avg_pool2d _ | Clamp _ | Conv2d _
  | Depthwise_conv2d _ | Div _ | Div_scalar _ | Hardsigmoid _ | Hardswish _
  | Hardtanh _ | Layer_norm _ | Max_pool2d _ | Mean_keepdims _ | Mul _ | Pad4 _
  | Relu _ | Rms_norm _ | Silu _ | Sqrt _ | Sub _ | Transposed_conv2d _ ->
      Output_transfer.Continuous

module Transfer = Output_transfer.Make (struct
  type nonrec op = Op.t

  let operands = Op.operands
  let classify = classify
end)

let propagate = Transfer.propagate
