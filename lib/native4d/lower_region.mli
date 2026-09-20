(* Multi-node regions the per-node lowerer cannot express, and how each is
   emitted. A region is a run of Native nodes whose INTERNAL tensors lie outside
   the four-axis domain but whose boundary tensors (the input, and every output)
   are inside it, so the whole run can be replaced by four-axis data movement
   without any dialect ever naming T or D.

   Two shapes, both [Reshape (-> Clone) -> Permute] over an in-domain operand:
   - the reshape target is out of domain but the permute's result is in it
     (a convolution weight relayout: [W=64 C=147] -> [D=64 H=3 W=7 C=7] ->
     [N=64 H=7 W=7 C=3]);
   - the permute's result is out of domain too, and is only ever selected or
     unbound along one axis (the qkv split of an attention block).

   A [Clone] between the reshape and the permute is absorbed. A third shape,
   found when neither of those applies, is a run of reshape, clone and permute
   nodes between two in-domain tensors with T or D on the interior: one index
   permutation of the flat data, planned by [Wide_permute] as a few
   [Reshape4]/[Permute4] steps.

   A fourth shape is a [Stack] on an in-frame axis read only by a [Reshape] back
   into the frame (a sin/cos interleave): each operand reshaped onto the
   stacked value's non-unit axes, a [Concat4] on the stacked one, and the final
   reshape.

   Every step is data movement, so each output keeps its source id and the claim
   is [Identical]; internal tensors have no destination and are deleted. The
   Native4D design record under [.ai/] describes the region relation. *)

open Graph_ir

type t = {
  trigger : Node_id.t;
      (** The member that emits the whole region: the permute when it has one
          consumer shape, otherwise the topologically first select or unbind, so
          every output exists before anything reads it. *)
  members : Node_id.t list;
      (** Every absorbed source node, [trigger] included. They form ONE cluster
          in the node map; the others emit nothing. *)
  internal : Tensor_id.t list;
      (** Source tensors with no destination: outside the domain by
          construction, and consumed only inside the region. *)
  input : Tensor_id.t;
      (** The region's operand; the first one when there are several. *)
  interleave : interleave option;
      (** Set for a stack read back into the frame, whose operands are all
          inputs and whose [outputs] steps start after the concatenation. *)
  outputs : output list;
}

and interleave
and output

val find : Graph_view.t -> t list
(** Regions a plain per-node lowering would reject. A region is reported only
    where its result fits the four-axis frame; anything else is left to the
    ordinary path, which names the real blocker. *)

val emit_region : Lower_engine_acc.acc -> t -> Lower_engine_acc.acc
(** Emits every output's chain at [trigger]. *)
