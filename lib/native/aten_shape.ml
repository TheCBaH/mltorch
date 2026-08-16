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

(* Dropping axes from the frame re-packs the survivors right-aligned, which is
   the same rule [used_axes] states — so it belongs here rather than restated by
   each op that removes an axis. Returns (surviving INPUT axis, OUTPUT axis
   carrying its data) pairs, in canonical order.

   Two ops need exactly this: [Reduce.Mean] with keepdim=false, and
   [Split.Unbind], which drops the axis it selects along. Restating it per op
   would be a second definition free to drift from the compute that reads it. *)
let repack_dropped ~dropped =
  let survivors = List.filter (fun a -> not (List.mem a dropped)) Axis.all in
  List.combine survivors (used_axes ~rank:(List.length survivors))

module View_size = struct
  type t = {
    size : int list;
    numel : int64;
    fault : [ `Multiple_inferred | `Not_divisible | `Count_mismatch ];
  }

  let pp_size ppf size =
    Fmt.pf ppf "[%a]" (Fmt.list ~sep:(Fmt.any ", ") Fmt.int) size

  let pp ppf { size; numel; fault } =
    match fault with
    | `Multiple_inferred ->
        Fmt.pf ppf "view size %a has more than one inferred (-1) dimension"
          pp_size size
    | `Not_divisible ->
        Fmt.pf ppf "view size %a does not divide %Ld elements" pp_size size
          numel
    | `Count_mismatch ->
        Fmt.pf ppf "view size %a does not match %Ld elements" pp_size size numel
end

(* Error set owned by this module: its own rank check and the [-1] convention's
   faults, unioned with [Dim.error] (from validating each untrusted dim/size
   entry). The printer delegates to [Dim]/[View_size]. *)
type rank_bound = { rank : int; lo : int; hi : int }

type error =
  [ `Rank_out_of_range of rank_bound | `View_size of View_size.t | Dim.error ]

let pp_error ppf : error -> unit = function
  | `Rank_out_of_range { rank; lo; hi } ->
      Format.fprintf ppf "rank %d out of [%d, %d]" rank lo hi
  | `View_size e -> View_size.pp ppf e
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

(* The order of these four steps IS the specification, because three of them
   divide: an earlier revision of this rule treated the rejections as a SET
   rather than a sequence, which is how a division by zero survives review --
   [size = [-1; 0]] folds a naive [known] to 0 and the very next operation is
   [numel / known].

   1. SCAN for [-1]: more than one is [`Multiple_inferred]. Every OTHER entry
      goes through [Dim.extent_checked] ([n >= 1]), so a zero or negative
      entry is already [`Non_positive_extent] (via [Dim.error]) before any
      division runs -- there is no separate zero rule to state.
   2. FOLD [known] in [int64] with a divide-first guard against [numel] itself,
      not [Kernel.Limits.Hard.numel]: that is the STRONGER bound and it is
      free. A size list that is a genuine view of [numel] has [known] dividing
      it, so [known > numel] is already the [`Count_mismatch] rejection --
      reached here, before the multiply that would produce it, rather than
      after. The caller has already bounded [numel] below [Hard.numel], so
      this fold never multiplies past 2^31 and cannot overflow [int64].
   3. Only now DIVIDE. With every contributing factor >= 1, [known] is never
      zero: a [-1] is resolved as [numel / known] after
      [Int64.rem numel known = 0L] ([`Not_divisible] otherwise); with no [-1],
      require [known = numel] ([`Count_mismatch] otherwise).
   4. NARROW last. The resolved entry is bounded by [numel < Hard.numel], so
      narrowing it to [int] happens after the bound, not before it. *)
let resolve_view_size ~numel size =
  let open Err.Syntax in
  let fail fault = Err.fail (`View_size { View_size.size; numel; fault }) in
  let n_inferred = List.length (List.filter (Int.equal (-1)) size) in
  let* () = if n_inferred > 1 then fail `Multiple_inferred else Err.return () in
  let* known =
    Err.List.fold_left
      (fun known d ->
        if d = -1 then Err.return known
        else
          let* e = Dim.extent_checked d in
          let e64 = Int64.of_int (Dim.to_int e) in
          if Int64.compare known (Int64.div numel e64) > 0 then
            fail `Count_mismatch
          else Err.return (Int64.mul known e64))
      1L size
  in
  let* inferred =
    if n_inferred = 0 then
      if Int64.equal known numel then Err.return None else fail `Count_mismatch
    else if not (Int64.equal (Int64.rem numel known) 0L) then
      fail `Not_divisible
    else Err.return (Some (Int64.div numel known))
  in
  Err.List.map
    (fun d ->
      Err.return (if d = -1 then Int64.to_int (Option.get inferred) else d))
    size
