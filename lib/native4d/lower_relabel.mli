(* A closed group of Native nodes whose values carry an extent on T or D can be
   lowered with N, T and D fused into N. The three are adjacent in the frame's
   order, so fusing them keeps every tensor's flat layout: for a tensor whose
   only such axis is D it is reading D as N. Most of the ops involved (elementwise
   arithmetic, batched matmul, convolution, attention, reshape, expand, sum and
   softmax over the axis, stack, split) mean the same thing on the fused axis; a
   permutation that moves T or D is planned as data movement of the flat layout,
   and an unbind or a reduction that re-packs its survivors is a slice or a
   reduction followed by a reshape. This is the
   split-attention pattern of ResNeSt and SK-Net (a reshape or stack introduces
   the axis, a softmax and a weighted sum consume it, and only its two ends touch
   the four-axis frame), the outlook attention of VOLO, the heads on T and D
   of a relative-position bias, and the window batch of a windowed attention.

   The result is a NATIVE graph in which the group's internal tensors are
   renamed and reshaped onto N, so the ordinary lowering applies to it
   unchanged. The internal source tensors have no destination and are deleted;
   the renamed ones are created; every boundary tensor keeps its id, so its
   claim is [Identical]. A permutation that needs several steps is replaced by
   a chain of Native nodes, the last keeping the member's id. The relabelled graph
   is validated, which catches any op whose shapes would not survive the fusion
   (a broadcast on T alone, say): the group is then left alone. A group that
   meets an op outside the supported set, a graph input or output with such an
   extent, or a tensor a region already owns is left alone too: the ordinary path
   then names the blocker. *)

open Graph_ir

type t = {
  members : Node_id.t list;  (** Source nodes of the group, in graph order. *)
  internal : Tensor_id.t list;  (** Source tensors with an extent on D. *)
  fresh : Tensor_id.t list;
      (** The relabelled twin of each [internal] tensor, pairwise. *)
  extra : Tensor_id.t list;
      (** Tensors created between the nodes a member is replaced by. *)
  extra_nodes : Node_id.t list;
      (** Nodes created beside the members, which keep their ids. *)
  nodes : (Node_id.t * node list) list;
      (** Each member with its replacement, in order, ending on the member's own
          id. *)
  sigs : Tensor_sig.t list;  (** Signatures of [fresh] and [extra]. *)
}

val find : Graph_view.t -> watermark:int -> avoid:Tensor_id.Set.t -> t list
(** Fresh tensor ids are allocated from [watermark] upward. A group that meets a
    tensor of [avoid] is left alone (the tensors a region already owns). *)

val apply : graph -> t list -> graph
(** The graph the ordinary lowering runs on. *)
