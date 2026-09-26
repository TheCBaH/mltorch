(* Permutation constants and shape/rank-resolution helpers for
   Native_interp, split out of native_interp_decode.ml once that file
   crossed the tracked 1000-line ceiling (scripts/check-file-size.sh).
   Depends on the generic argument-decoding helpers there (malformed,
   shape_of_sizes, ...), reached through the [open] below -- consumers of
   both open this module alongside it, the same convention
   native_interp_decode_conv.ml already established. *)

open Pytorch_types
open Schema_runtime
open Native_interp_error
open Native_interp_decode

let perm_nchw_to_nhwc =
  let open Axis in
  [ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ]

let perm_nhwc_to_nchw =
  let open Axis in
  [ (N, N); (T, T); (D, D); (H, C); (W, H); (C, W) ]

(* ATen batch normalization always names dimension 1 as channels, while the
   frame right-aligns an arbitrary ATen rank.  The fixed NCHW permutation above
   is its rank-4 instance; this pair also covers the corpus's rank-3 [N,C,L]
   activations without treating their batch extent as a channel. *)
let batch_norm_channel_perms ~rank =
  match Aten_shape.used_axes ~rank with
  | first :: channel :: rest ->
      let before_last xs = List.rev xs |> List.tl |> List.rev in
      let destinations =
        match rest with
        | [] -> [ first; Axis.C ]
        | _ -> first :: Axis.C :: channel :: before_last rest
      in
      let pairs = List.combine destinations (first :: channel :: rest) in
      let inverse_pairs = List.map (fun (dst, src) -> (src, dst)) pairs in
      let complete pairs =
        List.map
          (fun dst ->
            (dst, Option.value (List.assoc_opt dst pairs) ~default:dst))
          Axis.all
      in
      (complete pairs, complete inverse_pairs)
  | _ -> invalid_arg "batch_norm_channel_perms: ATen rank must be at least 2"

let perm_oihw_to_conv_weight =
  let open Axis in
  [ (N, D); (T, T); (D, N); (H, W); (W, C); (C, H) ]

(* [Op_bridge.perm_conv1d]'s mirror: right-aligned rank-3 [aten.conv1d.default]
   operands ([N,C,L] activation, [Cout,Cin/groups,K] weight) both land the
   same way under [of_aten] -- [role0, channel, spatial] -- so one permutation
   moves role0 onto native [N] (the axis [Conv2d.output_shape] reads the
   weight's real [Cout] from) and channel/spatial onto [C]/[W]. A pure product
   of two disjoint transpositions, hence its own inverse: it relayouts [x]/
   [weight] in and the raw output back to the generic [N,C,L] convention. *)
let perm_conv1d =
  let open Axis in
  [ (N, H); (T, T); (D, D); (H, N); (W, C); (C, W) ]

(* [Op_bridge.perm_conv3d]'s mirror: right-aligned rank-5
   [aten.conv3d.default] operands ([N,C,D,H,W] activation,
   [Cout,Cin/groups,Kd,Kh,Kw] weight) both land the same way under [of_aten]
   -- [role0, channel, spatial, spatial, spatial] -- so one permutation moves
   role0 onto native [N] and the three ATen spatial axes onto [D]/[H]/[W] in
   the SAME order, leaving channel on [C]. A genuine 4-cycle on [D,H,W,C], not
   its own inverse (unlike [perm_conv1d]'s pair of disjoint transpositions) --
   see [perm_conv3d_inv] for the relayout back to the generic [N,C,D,H,W]
   convention. *)
let perm_conv3d =
  let open Axis in
  [ (N, T); (T, N); (D, H); (H, W); (W, C); (C, D) ]

let perm_conv3d_inv =
  let open Axis in
  [ (N, T); (T, N); (D, C); (H, D); (W, H); (C, W) ]

(* Rank-2 addmm weight [In,Out] (W=In, C=Out) -> native [N=Out, C=In]. *)
let perm_addmm_weight =
  let open Axis in
  [ (N, C); (T, T); (D, D); (H, H); (W, N); (C, W) ]

(* Rank-2 linear weight [Out,In] (W=Out, C=In) -> native [N=Out, C=In]. NOT the
   permutation above, and not a rename of it: `addmm`'s [mat2] is the transpose
   of `linear`'s [weight], so an arm that reused one for the other would build a
   weight whose output and input axes are swapped. Both spellings exist in
   [Op_bridge] (op_bridge.ml:221,227) for the same reason. *)
let perm_linear_weight =
  let open Axis in
  [ (N, W); (T, T); (D, D); (H, H); (W, N); (C, C) ]

(* The [tensor_values] lookup, open-coded at five sites with the same three
   steps and a different role label each. Three functions rather than one
   because the sites want different depths: [mean.dim], [permute.default] and
   [unbind.int] need only the RANK, which a symbolic dimension does not
   prevent, while a conv weight needs the extents themselves.

   [role] stays a parameter so each caller keeps its own diagnostic. Sharing one
   role across two arms would make the row ambiguous about which one failed,
   which is the property that made these worth typing in the first place. *)
let tensor_meta esc (graph : Pytorch_types.Graph.t) ~ssa ~role =
  match String_map.find_opt ssa graph.tensor_values with
  | Some x -> x
  | None -> malformed esc (`Missing_metadata { ssa; role })

let meta_rank (meta : TensorMeta.t) = Rank.of_list meta.TensorMeta.sizes

(* [shape_of_sizes] RIGHT-ALIGNS a declared size list into the six-axis frame,
   so [C] and [1,C] land on exactly the same extents. [Graph_shape]'s operand
   check compares those frames and therefore cannot tell the two apart -- but
   ATen can, and refuses a bias that is not 1-D. The declared RANK exists only
   on this side of the conversion, so no shared native rule can cover it and
   each importer has to check its own. *)
let require_rank esc (graph : Pytorch_types.Graph.t) ~ssa ~role ~expected =
  let got = meta_rank (tensor_meta esc graph ~ssa ~role) in
  let expected = Rank.of_int expected in
  if not (Rank.equal got expected) then
    malformed esc
      (`Bad_dimension { tensor = ssa; fault = `Expected_rank { expected; got } })

let static_sizes esc ~tensor (meta : TensorMeta.t) =
  List.map
    (function
      | SymInt.Int i -> i
      | SymInt.Expr _ ->
          malformed esc (`Bad_dimension { tensor; fault = `Symbolic }))
    meta.TensorMeta.sizes

let sizes_rank_5 esc ~tensor = function
  | [ a; b; c; d; e ] -> (a, b, c, d, e)
  | sizes ->
      malformed esc
        (`Bad_dimension
           {
             tensor;
             fault =
               `Expected_rank
                 { expected = Rank.of_int 5; got = Rank.of_list sizes };
           })

let sizes_rank_4 esc ~tensor = function
  | [ a; b; c; d ] -> (a, b, c, d)
  | sizes ->
      malformed esc
        (`Bad_dimension
           {
             tensor;
             fault =
               `Expected_rank
                 { expected = Rank.of_int 4; got = Rank.of_list sizes };
           })

let sizes_rank_3 esc ~tensor = function
  | [ a; b; c ] -> (a, b, c)
  | sizes ->
      malformed esc
        (`Bad_dimension
           {
             tensor;
             fault =
               `Expected_rank
                 { expected = Rank.of_int 3; got = Rank.of_list sizes };
           })

let sizes_rank_2 esc ~tensor = function
  | [ a; b ] -> (a, b)
  | sizes ->
      malformed esc
        (`Bad_dimension
           {
             tensor;
             fault =
               `Expected_rank
                 { expected = Rank.of_int 2; got = Rank.of_list sizes };
           })

(* [used] is the innermost [rank] frame axes, so it has SIX entries once rank
   exceeds six — and then [d >= rank] admits d = 6 and [List.nth] raises
   [Failure "nth"]. The rank comes from a node's [tensor_values] metadata, which
   is untrusted model data and is NOT covered by [shape_of_sizes]'s own
   rank check: that one runs over graph inputs and captured tensors, not over an
   edge some node produced. Guarding here covers every caller
   (mean.dim, permute.default, unbind.int) rather than each arm separately, and
   reports the same row [shape_of_sizes] would for the same condition. *)
let used_axes_for esc ~tensor (rank : Rank.t) =
  if (rank :> int) > 6 then
    malformed esc (`Bad_dimension { tensor; fault = `Rank_over_six })
  else Aten_shape.used_axes ~rank

(* A dim number judged against a rank: negative counts from the end, and
   anything still outside [0, rank) is [`Axis_out_of_range], reported as
   written. The position it names. *)
let normalize_dim esc ~(rank : Rank.t) (dim : Aten_int.Dim.t) =
  let rank_n = (rank :> int) in
  let n = (dim :> int) in
  let n = if n < 0 then n + rank_n else n in
  if n < 0 || n >= rank_n then
    malformed esc (`Axis_out_of_range { axis = dim; rank })
  else n

(* Like [normalize_dim], for an op that INSERTS an axis (stack, unsqueeze): the
   valid positions are [0, rank], one more than an existing tensor has. The
   error still reports the operand's own rank. *)
let normalize_insert_dim esc ~(rank : Rank.t) (dim : Aten_int.Dim.t) =
  let rank_n = (rank :> int) in
  let n = (dim :> int) in
  let n = if n < 0 then n + rank_n + 1 else n in
  if n < 0 || n > rank_n then
    malformed esc (`Axis_out_of_range { axis = dim; rank })
  else n

(* The declared sizes of [shape]'s innermost [rank] axes as plain ints, for the
   arms that rebuild a size list to hand back to [shape_of_sizes]. *)
let aten_sizes ~rank shape =
  List.map
    (fun (s : Aten_int.Size.t) -> (s :> int))
    (Array.to_list (Aten_shape.to_aten ~rank shape))

(* The [normalized_shape] of layer_norm/rms_norm against the input's declared
   [sizes]: its length must lie in [1, rank] and it must equal the trailing
   sizes. Returns the trailing axes it normalizes over. *)
let normalized_axes esc ~tensor ~op sizes (normalized : Aten_int.Size.t list) =
  let rank = List.length sizes in
  let k = List.length normalized in
  if k < 1 || k > rank then
    malformed esc (`Normalized_rank { op; rank = Rank.of_int rank; got = k });
  let trailing l = List.filteri (fun i _ -> i >= rank - k) l in
  let expected = List.map Aten_int.Size.of_int (trailing sizes) in
  if expected <> normalized then
    malformed esc (`Normalized_shape { op; expected; got = normalized });
  trailing (used_axes_for esc ~tensor (Rank.of_int rank))

let axes_for_rank esc ~tensor (rank : Rank.t) (dims : Aten_int.Dim.t list) =
  let used = used_axes_for esc ~tensor rank in
  let rank_n = (rank :> int) in
  List.map
    (fun (d : Aten_int.Dim.t) ->
      let n = (d :> int) in
      let n = if n < 0 then n + rank_n else n in
      if n < 0 || n >= rank_n then
        malformed esc (`Axis_out_of_range { axis = d; rank })
      else List.nth used n)
    dims

let native_perm esc ~tensor ~(rank : Rank.t) (dims : Aten_int.Dim.t list) =
  let used = used_axes_for esc ~tensor rank in
  let rank_n = (rank :> int) in
  let outer = List.filter (fun a -> not (List.mem a used)) Axis.all in
  List.map (fun a -> (a, a)) outer
  @ List.mapi
      (fun i (d : Aten_int.Dim.t) ->
        let n = (d :> int) in
        let n = if n < 0 then n + rank_n else n in
        if n < 0 || n >= rank_n then
          malformed esc (`Axis_out_of_range { axis = d; rank });
        (List.nth used i, List.nth used n))
      dims

(* Shares [Aten_shape.resolve_view_size] with [Op_bridge] rather than
   re-deriving the [-1] convention: op3-impl.md F1 found this resolver
   accepted an invalid target silently (two [-1]s, a numel mismatch, a
   non-divisible inference) and F8 found its diagnostic named a tensor called
   "view" that never existed. Composed through [Err.Escape.or_throw], which
   exists precisely so a recursive walk can call an ordinary result-returning
   function without threading results through its own arms
   ([conv_in_channels] above is the same pattern: a bounded [int64] count
   inside the escape walk, reported as a typed row). *)
let resolve_view esc ~tensor shape size =
  let bad_view fault : error = `Bad_view { Bad_view.size; fault } in
  let numel =
    Err.Escape.or_throw esc
      (Err.map_error bad_view
         (Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel shape))
  in
  let resolved =
    Err.Escape.or_throw esc
      (Err.map_error
         (fun e -> bad_view (`Aten_shape e))
         (Aten_shape.resolve_view_size ~numel size))
  in
  shape_of_sizes esc tensor
    (List.map (fun (x : Aten_int.Size.t) -> SymInt.Int (x :> int)) resolved)

(* [expand.default]'s [size], resolved against [self_dims] ([self]'s own
   ATen rank -- see [Aten_shape.resolve_expand_size]'s comment for why this,
   not [shape], is what the [-1] convention needs). Mirrors [resolve_view]:
   composed through [Err.Escape.or_throw] so this recursive-walk caller can
   call an ordinary result-returning function directly, then handed to
   [shape_of_sizes] for the same right-alignment every other importer arm
   uses. *)
let resolve_expand esc ~tensor ~self_dims size =
  let bad_expand fault : error = `Bad_expand { Bad_expand.size; fault } in
  let resolved =
    Err.Escape.or_throw esc
      (Err.map_error
         (fun e -> bad_expand (`Aten_shape e))
         (Aten_shape.resolve_expand_size ~self_dims ~size))
  in
  shape_of_sizes esc tensor
    (List.map (fun (x : Aten_int.Size.t) -> SymInt.Int (x :> int)) resolved)

(* [repeat.default]'s [repeats], checked against [self_dims] ([self]'s own
   ATen rank -- see [Aten_shape.resolve_repeat_size]'s comment for why this,
   not [shape], is what the length check needs). Mirrors [resolve_expand]:
   composed through [Err.Escape.or_throw], then handed to [shape_of_sizes]
   for the same right-alignment every other importer arm uses -- unlike
   [resolve_expand] there is no [-1] substitution first, since [repeats] has
   no such convention. *)
let resolve_repeat esc ~tensor ~self_dims repeats =
  let bad_repeat fault : error = `Bad_repeat { Bad_repeat.repeats; fault } in
  let resolved =
    Err.Escape.or_throw esc
      (Err.map_error
         (fun e -> bad_repeat (`Aten_shape e))
         (Aten_shape.resolve_repeat_size ~self_dims ~repeats))
  in
  shape_of_sizes esc tensor
    (List.map (fun (x : Aten_int.Size.t) -> SymInt.Int (x :> int)) resolved)

(* [tile.default]'s own resolution: [Aten_shape.resolve_tile_size]'s left-pad
   rule, the reverse of [resolve_repeat]'s -- otherwise identical, down to
   the [shape_of_sizes] right-alignment. Its own [Bad_tile] row, not
   [resolve_repeat]'s [Bad_repeat], so a failure prints "tile", not
   "repeat" -- see [Bad_tile]'s own comment for why that fault is actually
   unreachable here. *)
let resolve_tile esc ~tensor ~self_dims dims =
  let bad_tile fault : error = `Bad_tile { Bad_tile.dims; fault } in
  let resolved =
    Err.Escape.or_throw esc
      (Err.map_error
         (fun e -> bad_tile (`Aten_shape e))
         (Aten_shape.resolve_tile_size ~self_dims ~dims))
  in
  shape_of_sizes esc tensor
    (List.map (fun (x : Aten_int.Size.t) -> SymInt.Int (x :> int)) resolved)

(* Shared by [upsample_bilinear2d.vec]/[upsample_nearest2d.vec]'s arms: both
   schemas are `(Tensor input, SymInt[]? output_size, ..., float[]?
   scale_factors)`, and ATen's own `compute_output_size` accepts exactly one
   of the two, never both, never neither. Same resolution [Op_bridge]'s
   [resolve_upsample_size] performs, restated here only because this importer
   reads serialized metadata where that one reads a live tensor. [op] is
   [node.target] (the FULL name), matching [Bad_upsample_size]'s own field. *)
let resolve_upsample_size esc ~op ~in_h ~in_w output_size scale_factors =
  match (output_size, scale_factors) with
  | [ h; w ], [] -> (h, w)
  | [], [ sh; sw ] ->
      ( int_of_float (float_of_int in_h *. sh),
        int_of_float (float_of_int in_w *. sw) )
  | [], [] ->
      malformed esc
        (`Bad_upsample_size
           { Bad_upsample_size.op; fault = Bad_upsample_size.Neither })
  | (_ :: _ :: _ | [ _ ]), [] ->
      malformed esc
        (`Bad_arity
           { Bad_arity.param = `Output_size; got = List.length output_size })
  | [], _ ->
      malformed esc
        (`Bad_upsample_size
           {
             Bad_upsample_size.op;
             fault =
               Bad_upsample_size.Bad_scale_arity (List.length scale_factors);
           })
  | _ :: _, _ :: _ ->
      malformed esc
        (`Bad_upsample_size
           { Bad_upsample_size.op; fault = Bad_upsample_size.Both })

(* The shared resolver, wrapped in this module's own row. Beside [resolve_view]
   and for its reason: the arm that calls it runs inside the builder monad,
   where the ambient error type is [Graph_builder.error], so the widening has to
   happen out here where [error] is what a row can be. *)
let resolve_slice_arg esc ~extent ~start ~stop ~step =
  Err.Escape.or_throw esc
    (Err.map_error
       (fun e : error ->
         `Bad_slice { Bad_slice.start; stop; step; fault = `Aten_shape e })
       (Aten_shape.resolve_slice ~extent ~start ~stop ~step))

(* [aten.select.int]'s index, shared with [Op_bridge] the same way
   [resolve_slice_arg] shares [Aten_shape.resolve_slice]: ATen REJECTS an
   out-of-range index rather than clamping it, so this cannot reuse
   [resolve_slice_arg]'s bound. *)
let resolve_select_index esc ~extent ~index =
  Err.Escape.or_throw esc
    (Err.map_error
       (fun e : error ->
         `Bad_select { Bad_select.index; fault = `Aten_shape e })
       (Aten_shape.resolve_index ~extent ~index))
