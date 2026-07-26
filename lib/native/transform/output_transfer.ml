(* How a value claim survives an op, per output. Consulted by [Rewrite]'s claim
   propagation; see .ai/native_transform_design.md §8.

   Exhaustive on purpose, with NO default arm: a defaulting classifier would
   silently mis-transfer the next op someone adds, and mis-transferring means a
   verifier asserting something the graph does not guarantee. Adding an op means
   deciding its answer here — the site is listed in .ai/native_add_op.md. *)

open Graph_ir

type t =
  | Continuous  (** small input change, small output change *)
  | Discontinuous  (** an arbitrarily small change can switch the result *)
  | Reindexing  (** the output is a permutation of the input's values *)

let pp fmt = function
  | Continuous -> Fmt.string fmt "continuous"
  | Discontinuous -> Fmt.string fmt "discontinuous"
  | Reindexing -> Fmt.string fmt "reindexing"

(* [Relu] and the pooled *value* are branch-selecting but continuous — the
   branches agree at the boundary — so only a genuine argmax is discontinuous. *)
let classify (op : op) ~output =
  match op with
  | Permute _ | Reshape _ -> Reindexing
  | Max_pool2d_with_indices _ ->
      if output = 0 then Continuous else Discontinuous
  (* No outputs at all, so this is unreachable from propagation; answer
     conservatively rather than inventing a guarantee. *)
  | Discard _ -> Discontinuous
  | Add _ | Avg_pool2d _ | Batch_norm _ | Bmm _ | Conv2d _ | Conv2d_padding _
  | Convolution _ | Div _ | Linear _ | Max_pool2d _ | Mean _ | Mul _ | Relu _
  | Rms_norm _ | Sqrt _ | Sub _ ->
      Continuous

(* [Identical] survives everything, evaluation being deterministic. [Equivalent]
   is a claim about exact arithmetic, so it survives any continuous output but
   not a discontinuous one: a rounding difference does not nudge an argmax, it
   selects a different element, and the checker compares computed values.
   [Approximate] dies at any actual computation — continuity gives no error
   BOUND, since multiplication amplifies by its other operand, reductions
   accumulate, sqrt is unbounded near zero, and quantized saturation is only
   piecewise continuous — so only a proven reindexing carries it. *)
let transfer (claim : Correspondence.relation) = function
  | Reindexing -> claim
  | Continuous -> (
      match claim with
      | Correspondence.Approximate _ -> Correspondence.Unverifiable
      | c -> c)
  | Discontinuous -> (
      match claim with
      | Correspondence.Identical -> Correspondence.Identical
      | _ -> Correspondence.Unverifiable)
