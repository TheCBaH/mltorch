(* Join and split family (cat/stack/meshgrid/split/unbind) op-dispatch arms for [Op_bridge], split from op_bridge_shape.ml. See
   op_bridge.ml for the [dispatch] facade that folds every family module
   (including this one) into one lookup. *)

open Pytorch_types
open Op_bridge_error
open Op_bridge_decode
module D = Interp_decode

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  match node.target with
  | "torch.ops.aten.cat.default" ->
      Some
        (let* aten_xs = tensors_arg aten_env node "tensors" in
         let* () = Err.List.iter (require_f32 "tensors") aten_xs in
         let* dim = dim_arg ~default:0 node "dim" in
         match aten_xs with
         | [] -> fail (`Concat_no_tensors "cat.default")
         | aten_x0 :: _ ->
             let rank = aten_rank aten_x0 in
             let* () =
               Err.List.iter
                 (fun t ->
                   let got = aten_rank t in
                   if Rank.equal got rank then return ()
                   else
                     fail
                       (`Concat_rank_mismatch
                          {
                            Concat_rank_mismatch.op = "cat.default";
                            first = rank;
                            other = got;
                          }))
                 aten_xs
             in
             let* d = norm_dim ~op:"cat.default" ~rank dim in
             let axis = Aten_shape.axis_of_dim ~rank (Aten_int.Dim.of_int d) in
             let* xs = Err.List.map (native_of_aten "tensors") aten_xs in
             build_g ~name:"cat" xs (fun ids ->
                 let open Graph_builder in
                 let+ y = concat { Concat.Concat.axis } ids in
                 [ y ]))
  (* One [Stack] node: inserts a size-1 axis per operand at [axis], then joins
     them — ATen's own definition is exactly
     `cat([t.unsqueeze(dim) for t in tensors], dim)`, which [Stack]'s shape
     rule/[Compute] reuse from [Concat]/[Split.Select]'s implementations
     rather than materializing as separate [Reshape] nodes. [dim] is judged
     against rank+1 valid positions, the OUTPUT rank, same reasoning as
     [norm_unsqueeze_dim]. *)
  | "torch.ops.aten.stack.default" ->
      Some
        (let* aten_xs = tensors_arg aten_env node "tensors" in
         let* () = Err.List.iter (require_f32 "tensors") aten_xs in
         let* dim = dim_arg ~default:0 node "dim" in
         match aten_xs with
         | [] -> fail (`Concat_no_tensors "stack.default")
         | aten_x0 :: _ ->
             let rank = aten_rank aten_x0 in
             let* () =
               Err.List.iter
                 (fun t ->
                   let got = aten_rank t in
                   if Rank.equal got rank then return ()
                   else
                     fail
                       (`Concat_rank_mismatch
                          {
                            Concat_rank_mismatch.op = "stack.default";
                            first = rank;
                            other = got;
                          }))
                 aten_xs
             in
             let* d = norm_unsqueeze_dim ~op:"stack.default" ~rank dim in
             let* xs = Err.List.map (native_of_aten "tensors") aten_xs in
             let axis =
               Aten_shape.axis_of_dim
                 ~rank:(Rank.of_int ((rank :> int) + 1))
                 (Aten_int.Dim.of_int d)
             in
             build_g ~name:"stack" xs (fun ids ->
                 let open Graph_builder in
                 let+ y = stack { Concat.Stack.axis } ids in
                 [ y ]))
  (* [meshgrid.indexing(Tensor[] tensors, *, str indexing) -> Tensor[]],
     restricted to the corpus's own shape -- see [Meshgrid.Meshgrid]'s own
     comment: every input rank-1, [indexing="ij"] only ("xy" swaps the first
     two OUTPUT axes for every output, a different rule with no corpus
     evidence). Rank itself is [Graph_shape]'s own check
     ([Meshgrid.Meshgrid.output_shapes]), not restated here. *)
  | "torch.ops.aten.meshgrid.indexing" ->
      Some
        (let* aten_xs = tensors_arg aten_env node "tensors" in
         let* indexing = string_arg ~default:"ij" node "indexing" in
         let* () =
           if String.equal indexing "ij" then return ()
           else
             fail
               (`Validation_failure
                  (Printf.sprintf
                     "meshgrid.indexing: indexing=%S is not supported (only \
                      \"ij\")"
                     indexing))
         in
         let* xs = Err.List.map (native_of_aten "tensors") aten_xs in
         build_g ~name:"meshgrid" xs (fun ids ->
             let open Graph_builder in
             meshgrid ids))
  (* One [Select] node: picks index [idx] along [axis] and drops it. Unlike
     [slice.Tensor], ATen REJECTS an out-of-range [index] rather than clamping
     it -- [Aten_shape.resolve_index], not [resolve_slice]. *)
  | "torch.ops.aten.unbind.int" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         (* Rank comes from the ORIGINAL ATen tensor: [of_aten] right-aligns
            into the six-axis frame, after which the rank is not recoverable. *)
         let rank = aten_rank aten_x in
         let* dim = dim_arg ~default:0 node "dim" in
         (* Convert BEFORE judging the dim, so a rank Native cannot represent is
            reported as the rank fault it is rather than as a bad dimension. *)
         let* x = native_of_aten "self" aten_x in
         let* axis = dim_axis ~op:"unbind.int" ~rank dim in
         build_g ~name:"unbind" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               unbind { Split.Unbind.axis } x_id
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* `unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)`:
     a sliding-window view. [dimension] resolves against [self]'s OWN rank
     exactly like [unbind.int]'s [dim] above, but names the axis the window
     COUNT lands on in the OUTPUT ([Unfold.Unfold.dest_of]), not the axis of
     [dimension] itself -- see unfold.ml's own header for why appending an
     ATen axis relabels every axis already in the frame. [dest_of] raises
     only when [dimension] resolves to the frame's own [N] (self already
     uses all six axes at its outermost position), which the six-axis frame
     categorically cannot grow past; every other rejection (self's [N] not
     already unit) is [Unfold.output_shape]'s own typed check. *)
  | "torch.ops.aten.split_with_sizes.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let rank = aten_rank aten_x in
         let* dim = dim_arg ~default:0 node "dim" in
         let* sizes = sizes_arg node "split_sizes" in
         let* x = native_of_aten "self" aten_x in
         let* axis = dim_axis ~op:"split_with_sizes.default" ~rank dim in
         let* params =
           Split.Split_with_sizes.of_aten ~axis
             ~in_extent:(Vec6.get (packed_shape x) axis)
             sizes
           |> Err.map_error (fun e -> `Build e)
         in
         build_g ~name:"split_with_sizes" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               split_with_sizes params x_id
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* `split.Tensor(self, split_size, dim)` -- the equal-chunk-size sibling of
     [split_with_sizes.default] just above. Binds to the *existing*
     [Split_with_sizes] node: [chunk_sizes] derives the same sizes list ATen
     itself would split into, so this is "translate params, still one node",
     not a decomposition. *)
  | "torch.ops.aten.split.Tensor" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let rank = aten_rank aten_x in
         let* dim = dim_arg ~default:0 node "dim" in
         let* split_size = int_arg node "split_size" in
         let* x = native_of_aten "self" aten_x in
         let* axis = dim_axis ~op:"split.Tensor" ~rank dim in
         let* split_size =
           pos ~op:"split.Tensor" ~param:`Split_size split_size
         in
         let extent = (Vec6.get (packed_shape x) axis :> int) in
         let sizes =
           List.map Dim.extent
             (chunk_sizes ~extent ~split_size:(split_size :> int))
         in
         build_g ~name:"split" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               split_with_sizes { Split.Split_with_sizes.axis; sizes } x_id
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* Both overloads share this body: after commit 1's checked resolver the body
     is three calls, and two copies of it is exactly the drift risk this repo
     keeps warning about. Still exact-target dispatch in op3.md's sense -- both
     targets are named, there is no fallthrough, and [native_of_aten]/every
     diagnostic below reads [node.target], so a failure still says which
     overload it was (op3-impl.md Part IV #2). [Identical] AFTER
     materialization, not alias-identical: ATen may return a view over the
     same storage, native graph edges are values. See
     .ai/native_aten_bridge_layout.md. *)
  (* Schema: `upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size,
     bool align_corners, float[]? scale_factors)`. Exactly one of
     [output_size]/[scale_factors] is ever given (ATen's own
     `compute_output_size` rejects both-or-neither); a [scale_factors] graph
     is resolved to an explicit size here, at import time, using ATen's own
     `floor(input_size * scale_factor)` -- the Native op only ever sees a
     concrete size, since Native shapes are static. *)
  (* Schema: `upsample_bicubic2d.vec(Tensor input, SymInt[]? output_size, bool
     align_corners, float[]? scale_factors)` -- same shape as
     [upsample_bilinear2d.vec] just below, including the output_size/
     scale_factors resolution; only the Native op differs. *)
  | _ -> None
