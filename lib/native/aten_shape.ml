(* The bridge between rank-bearing ATen/PyTorch shapes and the fixed 6-axis
   frame. Purely positional and RIGHT-ALIGNED: a rank-r ATen shape occupies the
   innermost r axes of [N T D H W C], the outer 6-r axes are size 1. ATen dim [i]
   maps to frame position [6 - r + i]. The reverse ([to_aten]) needs the rank,
   because the frame alone can't recover it (an [N=1] axis is ambiguous between
   "unused" and "batch of 1"). See .ai/native_tensor_design.md §1d.

   This is mechanical, not semantic: it does NOT place batch on [N] or remap
   NCHW<->NHWC — a rank-4 [n;h;w;c] lands with [n] on [D]. Semantic placement is
   a separate load-time concern. *)

(* The innermost [rank] axes, in canonical [N T D H W C] order. *)
let used_axes ~rank =
  if rank < 0 || rank > 6 then
    invalid_arg "Aten_shape.used_axes: rank out of [0,6]";
  let drop = List.length Axis.all - rank in
  List.filteri (fun i _ -> i >= drop) Axis.all

(* Error set owned by this module: its own rank check unioned with [Dim.error]
   (from validating each untrusted dim). The printer delegates to [Dim]. *)
type rank_bound = { rank : int; lo : int; hi : int }
type error = [ `Rank_out_of_range of rank_bound | Dim.error ]

let pp_error ppf : error -> unit = function
  | `Rank_out_of_range { rank; lo; hi } ->
      Format.fprintf ppf "rank %d out of [%d, %d]" rank lo hi
  | #Dim.error as e -> Dim.pp_error ppf e

(* Right-align an ATen shape into the frame; outer axes default to extent 1.
   Validates the untrusted rank and each dim, so it returns a [result]. *)
let of_aten (dims : int array) =
  let open Err.Syntax in
  let r = Array.length dims in
  let* () =
    if r > 6 then Err.fail (`Rank_out_of_range { rank = r; lo = 0; hi = 6 })
    else Err.return ()
  in
  Err.List.fold_left
    (fun s (ax, v) ->
      let+ e = Dim.extent_checked v in
      Vec6.set s ax e)
    (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
    (List.combine (used_axes ~rank:r) (Array.to_list dims))

(* Read back the innermost [rank] axes as an ATen shape. Inverse of [of_aten]
   given the original rank: [to_aten ~rank:(length a) (of_aten a) = a]. *)
let to_aten ~rank (s : Vec6.shape) =
  Array.of_list (List.map (fun ax -> (Vec6.get s ax :> int)) (used_axes ~rank))

(* The frame axis an ATen dim index refers to, for a tensor of the given rank.
   Negative [dim] counts from the end, as in PyTorch. *)
let axis_of_dim ~rank dim =
  if rank < 1 || rank > 6 then
    invalid_arg "Aten_shape.axis_of_dim: rank out of [1,6]";
  let d = if dim < 0 then dim + rank else dim in
  if d < 0 || d >= rank then
    invalid_arg "Aten_shape.axis_of_dim: dim out of range";
  List.nth (used_axes ~rank) d
