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

(* The [-1]-inference convention shared by [view.default]/[_unsafe_view.default]:
   a request to reshape into [size], validated against the source's element
   count. [numel] is the field a reader needs alongside the rejected [size],
   and [int64] because it is bounded below [Kernel.Limits.Hard.numel] by the
   caller ([Vec6.numel_bounded]) rather than trusted. *)
module View_size : sig
  type t = {
    size : int list;
    numel : int64;
    fault : [ `Multiple_inferred | `Not_divisible | `Count_mismatch ];
  }

  val pp : Format.formatter -> t -> unit
end

(* Error set owned by this module: its own rank check and the [-1] convention's
   faults, unioned with [Dim.error] (from validating each dim/size entry).
   [pp_error] delegates to [Dim.pp_error] and [View_size.pp]. *)
type rank_bound = { rank : int; lo : int; hi : int }

type error =
  [ `Rank_out_of_range of rank_bound | `View_size of View_size.t | Dim.error ]

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

(* Resolve a [view.default]/[_unsafe_view.default] target [size] (which may
   carry one [-1], PyTorch's "infer this dimension" convention) against the
   source's element count [numel]. [numel] is the caller's responsibility to
   bound (below [Kernel.Limits.Hard.numel], via [Vec6.numel_bounded]) before
   calling: this function trusts it and therefore never multiplies past it. *)
val resolve_view_size : numel:int64 -> int list -> (int list, [> error ]) Err.t
