(* Bridge between rank-bearing ATen shapes and the fixed 6-axis frame, right-
   aligned and positional: a rank-r ATen shape fills the innermost r axes of
   [N T D H W C]. Mechanical only — no batch-on-[N] or NCHW<->NHWC remap. See
   .ai/native_tensor_design.md §1d. *)

(* The innermost [rank] frame axes, in canonical order. [rank] must be in [0,6]. *)
val used_axes : rank:int -> Axis.t list

(* Right-align an ATen shape into the frame; the outer [6 - rank] axes are
   extent 1. Raises if [Array.length dims > 6]. *)
val of_aten : int array -> Vec6.shape

(* The ATen shape of the innermost [rank] axes — the inverse of [of_aten] given
   the original rank ([to_aten ~rank:(length a) (of_aten a) = a]). The rank is
   required: the frame alone cannot recover it. *)
val to_aten : rank:int -> Vec6.shape -> int array

(* The frame axis an ATen [dim] index addresses for a tensor of [rank] dims;
   negative [dim] counts from the end (PyTorch convention). *)
val axis_of_dim : rank:int -> int -> Axis.t
