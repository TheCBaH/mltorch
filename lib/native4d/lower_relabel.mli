(* A closed group of Native nodes whose values carry their extent on D (with N
   and T unit) can be lowered by reading D as N. The two axes are adjacent in
   the frame's order with only the unit T between them, so relabelling keeps
   every tensor's flat layout, and the ops involved (elementwise arithmetic,
   reshape, sum and softmax over the axis, stack) mean the same thing on either
   name. This is the split-attention pattern of ResNeSt and SK-Net: a reshape or
   stack introduces the axis, a softmax and a weighted sum consume it, and only
   its two ends touch the four-axis frame.

   The result is a NATIVE graph in which the group's internal tensors are
   renamed and reshaped onto N, so the ordinary lowering applies to it
   unchanged. The internal source tensors have no destination and are deleted;
   the renamed ones are created; every boundary tensor keeps its id, so its
   claim is [Identical]. A group that meets an op outside the supported set, a
   graph input or output with an extent on D, or a tensor with extent on N or T
   as well, is left alone: the ordinary path then names the blocker. *)

open Graph_ir

type t = {
  members : Node_id.t list;  (** Source nodes of the group, in graph order. *)
  internal : Tensor_id.t list;  (** Source tensors with an extent on D. *)
  fresh : Tensor_id.t list;
      (** The relabelled twin of each [internal] tensor, pairwise. *)
  nodes : node list;  (** The members with D read as N, same ids. *)
  sigs : Tensor_sig.t list;  (** Signatures of [fresh]. *)
}

val find : Graph_view.t -> watermark:int -> t list
(** Fresh ids are allocated from [watermark] upward. *)

val apply : graph -> t list -> graph
(** The graph the ordinary lowering runs on. *)
