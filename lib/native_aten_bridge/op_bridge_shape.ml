(* Shape and view family (permute/pad/slice/squeeze/view/expand/repeat...)
   op-dispatch arms for [Op_bridge], split from op_bridge.ml. See op_bridge.ml
   for the [dispatch] facade that folds every family module (including this
   one) into one lookup. *)

open Pytorch_types
open Op_bridge_error
open Op_bridge_decode
module D = Interp_decode

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  match node.target with
  | "torch.ops.aten.permute.default" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let rank = aten_rank t in
         let* dims = ints_arg node "dims" in
         let* perm =
           native_perm_of_aten ~op:"permute.default" ~rank
             (List.map Aten_int.Dim.of_int dims)
         in
         let* x = native_of_aten "self" t in
         build_g ~name:"permute" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = permute perm x_id in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* Reuses [native_perm_of_aten], the same permute machinery
     [permute.default] builds on, rather than a new builder -- outer padding
     axes stay identity because that helper already does that. Equal dims are
     a real identity transpose, not special-cased away: [List.init] produces
     the identity list, and it lowers like any other permutation. *)
  | "torch.ops.aten.transpose.int" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let rank = aten_rank t in
         let* dim0 = dim_arg node "dim0" in
         let* dim1 = dim_arg node "dim1" in
         let* d0 = norm_dim ~op:"transpose.int" ~rank dim0 in
         let* d1 = norm_dim ~op:"transpose.int" ~rank dim1 in
         let dims =
           List.init
             (rank :> int)
             (fun i ->
               Aten_int.Dim.of_int
                 (if i = d0 then d1 else if i = d1 then d0 else i))
         in
         let* perm = native_perm_of_aten ~op:"transpose.int" ~rank dims in
         let* x = native_of_aten "self" t in
         build_g ~name:"transpose" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = permute perm x_id in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* The only arm returning a variable number of outputs, and the only one whose
     count is fixed by the OPERAND rather than the op. Every slice is exposed:
     unlike a fixed tuple's dead output, there is nothing here to drop, and
     [Verify.verify_node] requires exact cardinality for a dynamic list for
     precisely that reason.

     This must NOT call ATen's own unbind. ATen is the oracle
     [Interp_verify]/[Verify.verify_node] runs; the native side has to execute
     [Graph_ir.Unbind] through [Eval_direct] or the comparison is vacuous. *)
  | "torch.ops.aten.pad.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* () = require_f32 "self" aten_x in
         (* Rank from the ORIGINAL ATen tensor, before [of_aten] right-aligns it
            into the frame and the rank stops being recoverable -- the same
            ordering [unbind.int] uses, and here it is load-bearing twice over:
            the pad list is indexed FROM the innermost dimension, so a wrong rank
            silently pads the wrong axes. *)
         let rank = aten_rank aten_x in
         let* pad = ints_arg node "pad" in
         let* mode = string_arg ~default:"constant" node "mode" in
         let* value = float_opt_arg node "value" in
         (* No [Err.map_error]: [params_of_aten] already fails with this
            module's own [`Bad_pad_list] row, so the detection origin is the
            check that found the fault rather than this call site. *)
         let* params = Pad.Pad.params_of_aten ~rank ~pad ~mode ~value in
         let* x = native_of_aten "self" aten_x in
         build_g ~name:"pad" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = pad params x_id in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* The bounds come from the LIVE tensor's extent, which is the whole reason
     this arm and the importer's differ at all: everything after
     [Aten_shape.resolve_slice] is shared, and the extent is the one thing the
     two paths learn differently. Rank before [of_aten], as [pad.default] and
     [unbind.int] do -- the frame does not carry it. *)
  | "torch.ops.aten.slice.Tensor" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* () = require_f32 "self" aten_x in
         let rank = aten_rank aten_x in
         let* dim = dim_arg ~default:0 node "dim" in
         let* start = int_opt_arg node "start" in
         let* stop = int_opt_arg node "end" in
         let* step = int_arg ~default:1 node "step" in
         let start = Option.map Aten_int.Index.of_int start in
         let stop = Option.map Aten_int.Index.of_int stop in
         let step = Aten_int.Step.of_int step in
         let* x = native_of_aten "self" aten_x in
         let* axis = dim_axis ~op:"slice.Tensor" ~rank dim in
         let extent = Vec6.get (packed_shape x) axis in
         let* bounds =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.resolve_slice ~extent ~start ~stop ~step)
         in
         build_g ~name:"slice" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y =
                 slice
                   {
                     Split.Slice.axis;
                     start = bounds.Aten_shape.Slice_bounds.start;
                     stop = bounds.Aten_shape.Slice_bounds.stop;
                     step = bounds.Aten_shape.Slice_bounds.step;
                   }
                   x_id
               in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* The variadic Native [Concat] op, direct: every operand keeps its rank,
     [dim] names an existing axis. ATen requires every tensor to share one
     rank (no broadcasting across the list), checked explicitly rather than
     left to [Concat.output_shape]'s own axis-agreement rule — see
     [Concat_rank_mismatch]'s doc comment for why that rule would report the
     wrong fault. *)
  | "torch.ops.aten.squeeze.dims" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let rank = aten_rank aten_x in
         let* dims = ints_arg node "dim" in
         let* x = native_of_aten "self" aten_x in
         let* ds =
           Err.List.map
             (fun d ->
               norm_dim ~op:"squeeze.dims" ~rank (Aten_int.Dim.of_int d))
             dims
         in
         let shape = packed_shape x in
         let aten_list = aten_sizes ~rank shape in
         let out_list =
           List.filteri
             (fun i _ -> not (List.mem i ds && List.nth aten_list i = 1))
             aten_list
         in
         let* target =
           Err.map_error (fun e -> `Aten_shape e) (of_sizes out_list)
         in
         build_g ~name:"squeeze" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = reshape { Reshape.Reshape.shape = target } x_id in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* [squeeze.dim(self, dim)]: drops the axis at [dim] when its LIVE extent is
     1. Real ATen leaves [self] unchanged (same rank) rather than raising when
     the extent is not 1 -- a genuine per-shape branch, decided here against
     the same live tensor [select.int]'s [resolve_index] reads, not a runtime
     value. Either branch keeps the linearized data order (dropping a unit
     axis is the exact inverse of [unsqueeze.default]'s insertion just below),
     so this legalizes to [Reshape] alone, the same "no new node" treatment. *)
  | "torch.ops.aten.squeeze.dim" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let rank = aten_rank aten_x in
         let* dim = dim_arg node "dim" in
         let* x = native_of_aten "self" aten_x in
         let* d = norm_dim ~op:"squeeze.dim" ~rank dim in
         let axis = Aten_shape.axis_of_dim ~rank (Aten_int.Dim.of_int d) in
         let extent = Vec6.get (packed_shape x) axis in
         let aten_list = aten_sizes ~rank (packed_shape x) in
         let out_list =
           if Dim.to_int extent = 1 then
             List.filteri (fun i _ -> i <> d) aten_list
           else aten_list
         in
         let* target =
           Err.map_error (fun e -> `Aten_shape e) (of_sizes out_list)
         in
         build_g ~name:"squeeze" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = reshape { Reshape.Reshape.shape = target } x_id in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* Legalized to [Reshape] alone: inserting a size-1 axis never changes the
     linearized data order, so no [Slice] is needed, unlike [select.int]'s
     axis removal. The target is [Aten_shape.to_aten]'s list with a bare [1]
     spliced in at the normalized position -- the same round-trip
     [select.int]/[view.default] use. *)
  | "torch.ops.aten.unsqueeze.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let rank = aten_rank aten_x in
         let* dim = dim_arg node "dim" in
         let* x = native_of_aten "self" aten_x in
         let* d = norm_unsqueeze_dim ~op:"unsqueeze.default" ~rank dim in
         let aten_list = aten_sizes ~rank (packed_shape x) in
         let front = List.filteri (fun i _ -> i < d) aten_list in
         let back = List.filteri (fun i _ -> i >= d) aten_list in
         let* target =
           Err.map_error
             (fun e -> `Aten_shape e)
             (of_sizes (front @ [ 1 ] @ back))
         in
         build_g ~name:"unsqueeze" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = reshape { Reshape.Reshape.shape = target } x_id in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  | "torch.ops.aten.unfold.default" ->
      Some
        (let op = "unfold.default" in
         let* aten_x = tensor_arg aten_env node "self" in
         let rank = aten_rank aten_x in
         let* dimension = dim_arg node "dimension" in
         let* source = dim_axis ~op ~rank dimension in
         let* size_int = int_arg node "size" in
         let* step_int = int_arg node "step" in
         let* x = native_of_aten "self" aten_x in
         try
           let* size = extent ~op ~param:`Kernel_size size_int in
           let* step = pos ~op ~param:`Stride step_int in
           let axis = Unfold.Unfold.dest_of source in
           build_g ~name:"unfold" [ x ] (function
             | [ x_id ] ->
                 let open Graph_builder in
                 let+ y = unfold { Unfold.Unfold.axis; size; step } x_id in
                 [ y ]
             | operands -> arity_mismatch ~op:node.Node.target operands)
         with Invalid_argument msg -> fail (`Validation_failure msg))
  (* One [Split_with_sizes] node: divides [axis] into contiguous windows of
     [split_sizes], KEEPING the axis in every output, unlike [unbind.int]
     which drops it. [split_sizes] is a required arg (no schema default), the
     same as [view.default]'s [size]; [Split.Split_with_sizes.output_shapes]
     is what checks the sizes are positive and sum to the axis's extent, not
     this arm -- the same division of labor [cat.default]'s rank check and
     [Concat.output_shape]'s axis-agreement check follow. Preserves the
     input's own dtype like [unbind.int], not [require_f32]'d like
     [cat.default]/[stack.default]. *)
  | "torch.ops.aten.view.default" | "torch.ops.aten._unsafe_view.default" ->
      Some
        ((* Contiguous reshape. [of_aten] inputs are already ATen-row-major, so
            the native reshape needs no surrounding permutes; the target native
            shape is [size] right-aligned (a single -1 resolved against numel). *)
         let* aten_x = tensor_arg aten_env node "self" in
         let* size = sizes_arg node "size" in
         let* x = native_of_aten "self" aten_x in
         let (Tensor.Tensor r) = x in
         (* [x]'s shape already cleared [Tensor_bridge.of_aten]'s numel
            preflight, so this count is < [Kernel.Limits.Hard.numel] and the
            [int64] conversion below cannot itself be the overflow this design
            guards against. *)
         let numel = Int64.of_int (Dim.to_int (Vec6.numel r.shape)) in
         let* resolved =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.resolve_view_size ~numel size)
         in
         let* target =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.of_aten (Array.of_list resolved))
         in
         let params = { Reshape.Reshape.shape = target } in
         build_g ~name:"view" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = reshape params x_id in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* [alias(self) -> Tensor] is a pure identity view (same storage, same
     size/strides) -- ATen's own schema takes no argument beyond [self]. Binds
     to the *existing* [Clone] node: paramless, [output_shape] the identity,
     and already the node this repo uses for "same value, no change" (see
     [clone.default]'s own arm) -- Native4D already elides it entirely
     (the Native4D design in `.ai/`, §7.1), so this legalization costs nothing
     there either. *)
  | "torch.ops.aten.expand.default" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let* size = sizes_arg node "size" in
         let* (_ : bool) = bool_arg node "implicit" in
         let* resolved =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.resolve_expand_size ~self_dims:(aten_dims t) ~size)
         in
         let* target =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.of_aten (Array.of_list resolved))
         in
         let* x = native_of_aten "self" t in
         build_g ~name:"expand" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = expand { Pointwise.Expand.size = target } x_id in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* `repeat(Tensor self, SymInt[] repeats) -> Tensor`. [repeats] is checked
     against [self]'s own ATen rank ([Aten_shape.resolve_repeat_size], the
     same split [expand.default]'s arm above makes for [size]) then
     right-aligned into the frame with [of_aten] -- unlike [expand]'s [size],
     [repeats] has no [-1] convention to resolve first, so there is no
     translation step between the two. Binds to the genuinely new
     [Repeat.Repeat]: each output element is the source element at the
     output coordinate MODULO the source extent, independently per axis
     ([Repeat.Repeat.Compute.pixel]), not a broadcast. *)
  | "torch.ops.aten.repeat.default" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let* repeats = sizes_arg node "repeats" in
         let* resolved =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.resolve_repeat_size ~self_dims:(aten_dims t) ~repeats)
         in
         let* target =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.of_aten (Array.of_list resolved))
         in
         let* x = native_of_aten "self" t in
         build_g ~name:"repeat" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = repeat { Repeat.Repeat.repeats = target } x_id in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* `tile(Tensor self, SymInt[] dims) -> Tensor`: legalizes onto the SAME
     [Repeat.Repeat] node [repeat.default] above builds, the only difference
     being [dims]'s own left-pad rule ([Aten_shape.resolve_tile_size]) --
     genuinely no new op, since ATen itself defines [tile] as [repeat] after
     that padding. *)
  | "torch.ops.aten.tile.default" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let* dims = sizes_arg node "dims" in
         let* resolved =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.resolve_tile_size ~self_dims:(aten_dims t) ~dims)
         in
         let* target =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.of_aten (Array.of_list resolved))
         in
         let* x = native_of_aten "self" t in
         build_g ~name:"tile" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = repeat { Repeat.Repeat.repeats = target } x_id in
               [ y ]
           | operands -> arity_mismatch ~op:node.Node.target operands))
  (* `repeat_interleave.self_int(Tensor self, SymInt repeats, int? dim=None,
     *, SymInt? output_size=None) -> Tensor`, restricted to an explicit
     [dim] -- the [dim=None] flatten-first form is a genuinely different
     reshape+repeat composition, deferred until a model demonstrates it (see
     [Repeat.RepeatInterleave]'s own comment). [output_size] is
     read-and-discarded, the same treatment [implicit] gets in the
     [expand.default] arm above: it exists only so a Tensor-valued [repeats]
     can skip a device sync, and this is the SymInt-scalar overload. *)
  | _ -> None
