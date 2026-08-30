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
  | Reindexing
      (** value ROUTING: every output element is copied from some input element
          with no arithmetic, so it carries its source element's claim
          unchanged. A permutation is the total case ([Permute], [Reshape],
          [Clone]); a SELECTION is the partial one ([Unbind], whose every output
          is one slice). Both are sound for the same reason — the claim is
          per-element, and copying preserves it — which is why the rule is
          "copied without arithmetic" rather than "the value multiset is
          unchanged". The latter is true only of the total case. *)

let pp fmt = function
  | Continuous -> Fmt.string fmt "continuous"
  | Discontinuous -> Fmt.string fmt "discontinuous"
  | Reindexing -> Fmt.string fmt "reindexing"

(* [Relu] and the pooled *value* are branch-selecting but continuous — the
   branches agree at the boundary — so only a genuine argmax is discontinuous. *)
let classify (op : op) ~output =
  match op with
  (* [Clone] is the identity, and identity is the degenerate permutation — so it
     is reindexing, not merely continuous. The difference is real: continuity
     downgrades an incoming [Approximate] claim to [Unverifiable], while
     reindexing carries it across unchanged, which is the right answer for an op
     that moves no value at all. *)
  | Clone _ | Permute _ | Reshape _ -> Reindexing
  (* Value routing, but a SELECTION rather than a permutation: each output is
     one slice of the operand, so the outputs' value multisets partition the
     input's rather than each reproducing it. [Slice] is the single-output form
     of the same argument: it keeps a strided SUBSET of the elements, adding no
     arithmetic to any of them. That is still [Reindexing] — the
     class is "copied without arithmetic", and an [Approximate] bound is
     per-element, so a slice carries it exactly as a permutation does. *)
  (* [Concat] joins several operands' elements with no arithmetic either — the
     N-input dual of [Unbind]'s N-output selection, same reasoning. *)
  (* [Upsample_nearest2d] is a third shape, neither permutation nor partition:
     a GATHER that may read one input element for several output positions
     (upsampling) or never read others, per [Resize.Nearest_axis]'s per-axis
     nearest-index computation. The soundness argument does not care which
     shape the routing takes — "copied without arithmetic" holds regardless of
     whether the map is injective or surjective — unlike [Upsample_bilinear2d]
     just below, whose output is a weighted BLEND of up to four input elements
     and is therefore [Continuous], not [Reindexing]. *)
  | Concat _ | Select _ | Slice _ | Split_with_sizes _ | Stack _ | Unbind _
  | Upsample_nearest2d _ ->
      Reindexing
  (* [Pad] is NOT reindexing, and the mode is why the honest answer is one class
     rather than two. In [Constant] mode the padded cells are a synthesized fill
     that is a copy of no input element, so a per-element [Approximate] claim
     about the input says nothing about them. In [Reflect] mode every output IS
     a copied element and [Reindexing] would be sound — but classification here
     is per OP, not per payload, and a class that changed with a field would be
     one refactor away from being read off the wrong one. [Continuous] is true
     of both: a small input change moves the copied cells slightly and leaves
     any constants exactly where they were. *)
  | Pad _ -> Continuous
  | Max_pool2d_with_indices _ ->
      if output = 0 then Continuous else Discontinuous
  (* No outputs at all, so this is unreachable from propagation; answer
     conservatively rather than inventing a guarantee. *)
  | Discard _ -> Discontinuous
  | Add _ | Add_scalar _ | Adaptive_avg_pool2d _ | Amax _ | Avg_pool2d _
  | Batch_norm _ | Bmm _ | Clamp _ | Conv2d _ | Conv2d_padding _ | Convolution _
  | Div _ | Div_scalar _ | Gelu _ | Group_norm _ | Hardsigmoid _ | Hardswish _
  | Hardtanh _ | Layer_norm _ | Linear _ | Max_pool2d _ | Mean _ | Mul _
  | Mul_scalar _ | Pow _ | Relu _ | Rms_norm _ | Sdpa _ | Sigmoid _ | Silu _
  | Softmax _ | Sqrt _ | Sub _ | Sum _ | Upsample_bilinear2d _ | Vector_norm _
    ->
      Continuous

(* [Identical] survives everything, evaluation being deterministic. [Equivalent]
   is a claim about exact arithmetic, so it survives any continuous output but
   not a discontinuous one: a rounding difference does not nudge an argmax, it
   selects a different element, and the checker compares computed values.
   [Approximate] dies at any actual computation — continuity gives no error
   BOUND, since multiplication amplifies by its other operand, reductions
   accumulate, sqrt is unbounded near zero, and quantized saturation is only
   piecewise continuous — so only a proven reindexing carries it. What makes
   that sound is that the claim is PER-ELEMENT and reindexing only copies
   elements; it does not depend on the whole multiset surviving, which is why
   a selection like [Unbind] qualifies as much as a permutation does. *)
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

(* The forward closure of the table above over a graph: after a fold declares its
   boundary [Equivalent], every edge downstream of it is too, and leaving those
   implicitly [Identical] would have a verifier assert bit-equality on the
   model's final output.

   [explicit] seeds the claims a map states outright, keyed by destination id;
   [preserved] answers "does the source graph have this id too", since an id
   present in only one version has no counterpart to claim anything about.
   Entries equal to [Identical] are omitted, so the result names exactly the
   edges a map must mention. The graph must be in topological order, which is
   what [Graph_view.of_graph] checks.

   One implementation, two callers with opposite purposes: [Rewrite.apply] uses
   it to LABEL the map it is building, [Graph_map.create] to REJECT a map whose
   labels are not closed. They have to agree, and sharing the code is how. *)
(* PARAMETERISED over an inline reduced signature, deliberately NOT over
   [Dialect.S]: [Dialect.S] names [Output_transfer.t] as [classify]'s return
   type, so depending on it here would close a compilation-unit cycle. Any
   [Dialect.S] satisfies this structurally, so the constraint costs nothing at
   the use site and keeps the dependency running one way. *)
module type OPS = sig
  type op

  val operands : op -> Tensor_id.t list
  val classify : op -> output:int -> t
end

module Make (D : OPS) = struct
  let propagate ~explicit ~preserved (g : D.op Graph_common.Graph.t) =
    List.fold_left
      (fun acc (n : D.op Graph_common.Node.t) ->
        let operand_claim id =
          Option.value
            (Tensor_id.Map.find_opt id acc)
            ~default:Correspondence.Identical
        in
        let incoming =
          List.fold_left
            (fun c id -> Correspondence.join c (operand_claim id))
            Correspondence.Identical (D.operands n.Node.op)
        in
        List.fold_left
          (fun (acc, i) out ->
            let acc =
              if Tensor_id.Map.mem out acc then acc
              else if not (preserved out) then acc
              else
                let claim =
                  transfer incoming (D.classify n.Node.op ~output:i)
                in
                if claim = Correspondence.Identical then acc
                else Tensor_id.Map.add out claim acc
            in
            (acc, i + 1))
          (acc, 0) n.Node.outputs
        |> fst)
      explicit g.Graph_common.Graph.nodes
end

(* The Native specialization, so every existing caller is unchanged. *)
include Make (struct
  type nonrec op = op

  let operands = Graph_ir.operands
  let classify = classify
end)
