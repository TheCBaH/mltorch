(* Bridge between rank-bearing ATen shapes and the fixed 6-axis frame, right-
   aligned and positional: a rank-r ATen shape fills the innermost r axes of
   [N T D H W C]. Mechanical only — no batch-on-[N] or NCHW<->NHWC remap. See
   .ai/native_tensor_design.md §1d. *)

(* The innermost [rank] frame axes, in canonical order. [rank] must be in [0,6]. *)
val used_axes : rank:int -> Axis.t list

(* Dropping axes re-packs the survivors right-aligned — the same rule
   [used_axes] states, so it lives here rather than being restated by each op
   that removes an axis ([Reduce.Mean] with keepdim=false, [Split.Unbind]).
   Returns (surviving INPUT axis, OUTPUT axis carrying its data) pairs, in
   canonical order. *)
val repack_dropped : dropped:Axis.t list -> (Axis.t * Axis.t) list

(* Error set owned by this module: its own rank check unioned with [Dim.error]
   (from validating each dim). [pp_error] delegates to [Dim.pp_error]. *)
type rank_bound = { rank : int; lo : int; hi : int }
type error = [ `Rank_out_of_range of rank_bound | Dim.error ]

val pp_error : Format.formatter -> error -> unit

(* Right-align an ATen shape into the frame; the outer [6 - rank] axes are
   extent 1. [Error] if the rank exceeds 6 or any dim is negative. *)
val of_aten : int array -> (Vec6.shape, error) Err.t

(* The ATen shape of the innermost [rank] axes — the inverse of [of_aten] given
   the original rank ([to_aten ~rank:(length a) (of_aten a) = a]). The rank is
   required: the frame alone cannot recover it. *)
val to_aten : rank:int -> Vec6.shape -> int array

(* The frame axis an ATen [dim] index addresses for a tensor of [rank] dims;
   negative [dim] counts from the end (PyTorch convention). *)
val axis_of_dim : rank:int -> int -> Axis.t
